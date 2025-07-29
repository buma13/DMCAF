import os
import sqlite3
import numpy as np
from typing import List
from datetime import datetime

from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import torch
from scipy.linalg import sqrtm
from .object_count import ObjectCounter
from .object_detection import ObjectDetector
from .object_color_classification import ObjectColorClassifier


class Evaluator:
    def __init__(self, conditioning_db_path: str, output_db_path: str, metrics_db_path: str):
        self.conditioning_conn = sqlite3.connect(conditioning_db_path)
        self.output_conn = sqlite3.connect(output_db_path)
        self.evaluation_conn = sqlite3.connect(metrics_db_path)
        self._create_evaluation_table()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _create_evaluation_table(self):
        cursor = self.evaluation_conn.cursor()

        # Table for per-image/per-condition raw metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                condition_id INTEGER,
                model_name TEXT,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                metric_type TEXT,
                metric_name TEXT,
                value REAL,
                image_path TEXT,
                timestamp TEXT
            )
        """)

        # Table specific for count evaluation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_count_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                condition_id INTEGER,
                model_name TEXT,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                metric_type TEXT,
                metric_name TEXT,
                value REAL,
                expected_number INTEGER,
                detected_count INTEGER,
                image_path TEXT,
                timestamp TEXT
            )
        """)

                # Table specific for count evaluation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_color_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                condition_id INTEGER,
                model_name TEXT,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                metric_type TEXT,
                metric_name TEXT,
                value REAL,
                expected_color TEXT,
                detected_color TEXT,
                image_path TEXT,
                timestamp TEXT
            )
        """)

        # Table for experiment-wide aggregated metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                model_name TEXT,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                metric_type TEXT,
                metric_name TEXT,
                mean_value REAL,
                std_value REAL,
                min_value REAL,
                max_value REAL,
                count INTEGER,
                timestamp TEXT
            )
        """)

        self.evaluation_conn.commit()

    def evaluate_outputs(self, experiment_id: str, metrics: List[str]):
        if "FID" in metrics:
            self._evaluate_fid(experiment_id)
        if "YOLO Count" in metrics:
            self.evaluate_object_count(experiment_id)
        if "YOLO Composition" in metrics:
            self.evaluate_object_composition(experiment_id)
        if "Color Classification" in metrics:
            self.evaluate_color_classification(experiment_id)

    def _evaluate_fid(self, experiment_id: str):
        cursor = self.output_conn.cursor()
        cursor.execute("""
            SELECT model_name, guidance_scale, num_inference_steps, image_path
            FROM dm_outputs
            WHERE experiment_id = ?
        """, (experiment_id,))
        rows = cursor.fetchall()

        config_to_images = {}
        for model_name, guidance_scale, num_steps, image_path in rows:
            key = (model_name, guidance_scale, num_steps)
            config_to_images.setdefault(key, []).append(image_path)

        for (model_name, guidance_scale, num_steps), image_paths in config_to_images.items():
            print(f"Evaluating FID for {model_name} (guidance={guidance_scale}, steps={num_steps})")
            activations = self._compute_activations(image_paths)
            mu_gen, sigma_gen = activations.mean(axis=0), np.cov(activations, rowvar=False)

            # Simulated real stats (for now, use offset generated stats)
            mu_real, sigma_real = mu_gen + 0.1, sigma_gen + 0.01

            fid_score = self._calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
            self._log_experiment_metric(
                experiment_id, model_name,
                guidance_scale, num_steps,
                "Quality", "FID", fid_score
            )

    def evaluate_object_count(self, experiment_id: str):
        """
        Evaluates if the number of generated objects matches the number in the prompt.
        """
        print(f"Evaluating Object Count Accuracy for experiment: {experiment_id}")
        obj_counter = ObjectCounter()
        output_cursor = self.output_conn.cursor()
        cond_cursor = self.conditioning_conn.cursor()

        output_cursor.execute("""
            SELECT condition_id, image_path, model_name, guidance_scale, num_inference_steps
            FROM dm_outputs
            WHERE experiment_id = ?
        """, (experiment_id,))

        for cond_id, image_path, model, gs, steps in output_cursor.fetchall():
            cond_cursor.execute("SELECT number, object FROM conditions WHERE id = ?", (cond_id,))
            cond_result = cond_cursor.fetchone()
            if not cond_result:
                continue

            expected_number, target_object = cond_result
            if not os.path.exists(image_path):
                print(f"Skipping non-existent image: {image_path}")
                continue

            detected_count = obj_counter.count_objects_in_image(image_path, target_object)

            # Accuracy is 1 if counts match, 0 otherwise.
            accuracy = 1.0 if detected_count == expected_number else 0.0

            print(f"Image: {os.path.basename(image_path)}, Expected: {expected_number} {target_object}, Found: {detected_count}, Accuracy: {accuracy}")

            self._log_raw_metric(
                experiment_id, cond_id, model, gs, steps,
                "Accuracy", f"ObjectCountAccuracy_{target_object}", accuracy, image_path
            )
            self._log_raw_count_metric(
                experiment_id, cond_id, model, gs, steps,
                "Accuracy", f"ObjectCountAccuracy_{target_object}", accuracy, expected_number, detected_count, image_path
            )
        self._compute_experiment_aggregates(experiment_id)

    def evaluate_color_classification(self, experiment_id: str):
        """
        Evaluates if the color of objects in generated images matches the expected color.
        """
        print(f"Evaluating Object Color Classification Accuracy for experiment: {experiment_id}")
        color_classifier = ObjectColorClassifier()
        output_cursor = self.output_conn.cursor()
        cond_cursor = self.conditioning_conn.cursor()

        output_cursor.execute("""
            SELECT condition_id, image_path, model_name, guidance_scale, num_inference_steps
            FROM dm_outputs
            WHERE experiment_id = ?
        """, (experiment_id,))

        for cond_id, image_path, model, gs, steps in output_cursor.fetchall():
            cond_cursor.execute("SELECT color1, object FROM conditions WHERE id = ?", (cond_id,))
            cond_result = cond_cursor.fetchone()
            if not cond_result:
                continue

            expected_color, target_object = cond_result
            if not os.path.exists(image_path):
                print(f"Skipping non-existent image: {image_path}")
                continue

            detected_color, confidence = color_classifier.classify_color(image_path, target_object)

            # Accuracy is 1 if colors match, 0 otherwise.
            accuracy = 1.0 if detected_color == expected_color else 0.0

            print(f"Image: {os.path.basename(image_path)}, Expected: {expected_color} {target_object}, Found: {detected_color}, Accuracy: {accuracy}")

            self._log_raw_metric(
                experiment_id, cond_id, model, gs, steps,
                "Accuracy", f"ColorClassificationAccuracy_{target_object}", accuracy, image_path
            )
            self._log_raw_color_metric(
                experiment_id, cond_id, model, gs, steps,
                "Accuracy", f"ColorClassificationAccuracy_{target_object}", accuracy,
                expected_color, detected_color, image_path
            )

        # After processing all images, compute experiment-wide aggregates
        self._compute_experiment_aggregates(experiment_id)
    def evaluate_object_composition(self, experiment_id: str):
        """
        Evaluates if the spatial relationship between objects in generated images
        matches the compositional prompt by analyzing bounding boxes.
        """
        print(f"Evaluating Object Composition for experiment: {experiment_id}")
        obj_detector = ObjectDetector()
        output_cursor = self.output_conn.cursor()
        cond_cursor = self.conditioning_conn.cursor()

        # First, get all outputs for the experiment from the output database.
        output_cursor.execute("""
            SELECT condition_id, image_path, model_name, guidance_scale, num_inference_steps
            FROM dm_outputs
            WHERE experiment_id = ?
        """, (experiment_id,))

        for cond_id, image_path, model, gs, steps in output_cursor.fetchall():
            # For each output, check the condition type from the conditioning database.
            cond_cursor.execute("""
                SELECT type, object, number, relationship, object2, number2
                FROM conditions WHERE id = ?
            """, (cond_id,))
            cond_result = cond_cursor.fetchone()

            if not cond_result or cond_result[0] != 'compositional_prompt':
                continue

            _, obj1_name, num1, relation, obj2_name, num2 = cond_result
            obj1_singular = obj1_name.rstrip('s')
            obj2_singular = obj2_name.rstrip('s')

            if not os.path.exists(image_path):
                print(f"Skipping non-existent image: {image_path}")
                continue
            detected_objects = obj_detector.detect_objects_with_boxes(image_path)

            obj1_boxes = [d['box'] for d in detected_objects if d['class_name'] == obj1_singular]
            obj2_boxes = [d['box'] for d in detected_objects if d['class_name'] == obj2_singular]

            # Basic check: did we detect at least one of each required object?
            if not obj1_boxes or not obj2_boxes:
                accuracy = 0.0
            else:
                # If objects are detected, check if any pair satisfies the spatial relationship.
                accuracy = 0.0 # Assume failure until a correct pair is found
                for b1 in obj1_boxes:
                    for b2 in obj2_boxes:
                        if self._check_spatial_relation(b1, b2, relation):
                            accuracy = 1.0
                            break # Found a valid pair, no need to check others
                    if accuracy == 1.0:
                        break

            metric_name = f"CompositionalAccuracy_{obj1_singular}_{relation}_{obj2_singular}"
            print(f"Image: {os.path.basename(image_path)}, Relation: '{relation}', Accuracy: {accuracy}")

            self._log_raw_metric(
                experiment_id, cond_id, model, gs, steps,
                "Accuracy", metric_name, accuracy, image_path
            )

        # After processing all images, compute experiment-wide aggregates
        self._compute_experiment_aggregates(experiment_id)

    def _check_spatial_relation(self, box1: List[float], box2: List[float], relation: str) -> bool:
        """Checks if box1 is in the correct spatial relation to box2."""
        # Get center points
        c1_x, c1_y = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        c2_x, c2_y = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2

        # Get heights and widths for proximity checks
        h1, w1 = box1[3] - box1[1], box1[2] - box1[0]
        h2, w2 = box2[3] - box2[1], box2[2] - box2[0]

        is_correct = False
        if relation in ['above', 'on top of']:
            # Center of box1 is above center of box2
            # and they are vertically aligned (overlap horizontally)
            is_correct = c1_y < c2_y and max(box1[0], box2[0]) < min(box1[2], box2[2])
        elif relation == 'below':
            is_correct = c1_y > c2_y and max(box1[0], box2[0]) < min(box1[2], box2[2])
        elif relation == 'to the left of':
            # Center of box1 is left of center of box2
            # and they are horizontally aligned (overlap vertically)
            is_correct = c1_x < c2_x and max(box1[1], box2[1]) < min(box1[3], box2[3])
        elif relation == 'to the right of':
            is_correct = c1_x > c2_x and max(box1[1], box2[1]) < min(box1[3], box2[3])
        elif relation == 'next to':
            # Check for horizontal or vertical proximity.
            is_horizontally_aligned = max(box1[1], box2[1]) < min(box1[3], box2[3])
            is_horizontally_close = abs(c1_x - c2_x) < (w1 + w2) # Allow some gap

            is_vertically_aligned = max(box1[0], box2[0]) < min(box1[2], box2[2])
            is_vertically_close = abs(c1_y - c2_y) < (h1 + h2) # Allow some gap

            is_correct = (is_horizontally_aligned and is_horizontally_close) or \
                         (is_vertically_aligned and is_vertically_close)

        return is_correct

    def _compute_activations(self, image_paths: List[str]) -> np.ndarray:
        batch = []
        for path in image_paths:
            if not os.path.exists(path):
                continue
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            batch.append(tensor)

        if not batch:
            return np.empty((0, 2048))

        batch_tensor = torch.cat(batch)
        with torch.no_grad():
            features = self.model(batch_tensor)
        return features.cpu().numpy()

    def _calculate_fid(self, mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        covmean = sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def _log_raw_metric(self, experiment_id: str, condition_id: int, model_name: str,
                        guidance_scale: float, num_steps: int, metric_type: str,
                        metric_name: str, value: float, image_path: str):
        cursor = self.evaluation_conn.cursor()
        cursor.execute("""
            INSERT INTO raw_metrics (
                experiment_id, condition_id, model_name, guidance_scale,
                num_inference_steps, metric_type, metric_name, value, image_path, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, condition_id, model_name, guidance_scale,
            num_steps, metric_type, metric_name, value, image_path, datetime.now().isoformat()
        ))
        self.evaluation_conn.commit()

    def _log_raw_count_metric(self, experiment_id: str, condition_id: int, model_name: str,
                        guidance_scale: float, num_steps: int, metric_type: str,
                        metric_name: str, value: float, expected_number: int, detected_count: int,
                        image_path: str):
        cursor = self.evaluation_conn.cursor()
        cursor.execute("""
            INSERT INTO raw_count_metrics (
                experiment_id, condition_id, model_name, guidance_scale,
                num_inference_steps, metric_type, metric_name, value, expected_number, detected_count, image_path,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, condition_id, model_name, guidance_scale, num_steps, metric_type,
            metric_name, value, expected_number, detected_count, image_path, datetime.now().isoformat()
        ))
        self.evaluation_conn.commit()

    def _log_raw_color_metric(self, experiment_id: str, condition_id: int, model_name: str,
                        guidance_scale: float, num_steps: int, metric_type: str,
                        metric_name: str, value: float, expected_color: str, detected_color: str, image_path: str):
        cursor = self.evaluation_conn.cursor()
        cursor.execute("""
            INSERT INTO raw_color_metrics (
                experiment_id, condition_id, model_name, guidance_scale,
                num_inference_steps, metric_type, metric_name, value, expected_color, detected_color, image_path, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, condition_id, model_name, guidance_scale,
            num_steps, metric_type, metric_name, value, expected_color, detected_color, image_path, datetime.now().isoformat()
        ))
        self.evaluation_conn.commit()

    def _log_experiment_metric(self, experiment_id: str, model_name: str,
                               guidance_scale: float, num_steps: int, metric_type: str,
                               metric_name: str, value: float):
        cursor = self.evaluation_conn.cursor()
        cursor.execute("""
            INSERT INTO experiment_metrics (
                experiment_id, model_name, guidance_scale, num_inference_steps,
                metric_type, metric_name, mean_value, std_value, min_value, max_value, count, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, model_name, guidance_scale, num_steps, metric_type, metric_name,
            value, None, None, None, None, datetime.now().isoformat()
        ))
        self.evaluation_conn.commit()

    def _compute_experiment_aggregates(self, experiment_id: str):
        """Compute and store experiment-wide statistics from raw metrics."""
        cursor = self.evaluation_conn.cursor()

        # Get all unique configurations for this experiment
        cursor.execute("""
            SELECT DISTINCT model_name, guidance_scale, num_inference_steps, metric_type, metric_name
            FROM raw_metrics
            WHERE experiment_id = ?
        """, (experiment_id,))

        for model, gs, steps, metric_type, metric_name in cursor.fetchall():
            # Calculate aggregates for this configuration and metric
            cursor.execute("""
                SELECT value FROM raw_metrics
                WHERE experiment_id = ? AND model_name = ? AND guidance_scale = ?
                AND num_inference_steps = ? AND metric_type = ? AND metric_name = ?
            """, (experiment_id, model, gs, steps, metric_type, metric_name))

            values = [row[0] for row in cursor.fetchall()]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                count = len(values)

                # Insert aggregated metrics
                cursor.execute("""
                    INSERT INTO experiment_metrics (
                        experiment_id, model_name, guidance_scale, num_inference_steps,
                        metric_type, metric_name, mean_value, std_value, min_value, max_value, count, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (experiment_id, model, gs, steps, metric_type, metric_name,
                      mean_val, std_val, min_val, max_val, count, datetime.now().isoformat()))

        self.evaluation_conn.commit()

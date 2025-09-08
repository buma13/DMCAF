import os
import sqlite3
import cv2
import numpy as np
import json
from typing import List, Dict, Tuple, Any
from datetime import datetime

from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
import torch
from scipy.linalg import sqrtm
from ultralytics import YOLO
from .object_count import ObjectCounter
from .object_detection import ObjectDetector
from .object_color_classification import ObjectColorClassifier
from .object_segmentation import ObjectSegmenter

# Class name and color mappings used for segmentation evaluation
CLASS_NAME_MAPPING = {
    0: "Black Background",
    1: "Abdominal Wall",
    2: "Liver",
    3: "Gastrointestinal Tract",
    4: "Fat",
    5: "Grasper",
    6: "Connective Tissue",
    7: "Blood",
    8: "Cystic Duct",
    9: "L-hook Electrocautery",
    10: "Gallbladder",
    11: "Hepatic Vein",
    12: "Liver Ligament",
}

CLASS_COLOR_MAPPING = {
    0: (127, 127, 127),
    1: (210, 140, 140),
    2: (255, 114, 114),
    3: (231, 70, 156),
    4: (186, 183, 75),
    5: (170, 255, 0),
    6: (255, 85, 0),
    7: (255, 0, 0),
    8: (255, 255, 0),
    9: (169, 255, 184),
    10: (255, 160, 165),
    11: (0, 50, 128),
    12: (111, 74, 0),
}

from pathlib import Path
from torch import nn
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import shutil
import math
from monai.metrics import DiceMetric

class Evaluator:
    def __init__(self, conditioning_db_path: str, output_db_path: str, metrics_db_path: str):
        self.conditioning_conn = sqlite3.connect(conditioning_db_path)
        self.output_conn = sqlite3.connect(output_db_path)
        self.evaluation_conn = sqlite3.connect(metrics_db_path)
        self._create_evaluation_table()

        # Load YOLO class mappings for efficient class ID lookup
        self.yolo_class_to_id = self._load_yolo_classes()

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

    def _load_yolo_classes(self) -> Dict[str, int]:
        """
        Load YOLO class mappings from JSON file for efficient class ID lookup.
        Returns a dictionary mapping class names to class IDs.
        """
        try:
            # Try to find the assets directory relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            assets_path = os.path.join(os.path.dirname(current_dir), 'assets', 'yolo_classes.json')
            
            with open(assets_path, 'r') as f:
                yolo_classes = json.load(f)
            
            # Convert from {class_id: class_name} to {class_name: class_id}
            class_to_id = {}
            for class_id, class_name in yolo_classes['class'].items():
                class_to_id[class_name] = int(class_id)
            
            print(f"Loaded {len(class_to_id)} YOLO class mappings from {assets_path}")
            return class_to_id
            
        except FileNotFoundError:
            print("Warning: Could not load YOLO classes JSON file. Class filtering will be disabled.")
            return {}
        except Exception as e:
            print(f"Error loading YOLO classes: {e}. Class filtering will be disabled.")
            return {}

    def _create_evaluation_table(self):
        cursor = self.evaluation_conn.cursor()
        
        # Single table for comprehensive per-image evaluation results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                condition_id INTEGER,
                condition_type TEXT,
                variant TEXT,
                image_path TEXT,
                model_name TEXT,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                
                -- Count evaluation metrics
                expected_count INTEGER,
                detected_count INTEGER,
                count_accuracy REAL,
                target_object TEXT,
                
                -- Segmentation metrics (updated pixel_ratio calculation)
                target_pixels INTEGER,  -- Total pixels covered by objects (for coverage)
                total_pixels INTEGER,
                pixel_ratio REAL,  -- Now: segmented pixels / bounding box area
                coverage_percentage REAL,  -- Unchanged: percentage of image covered
                segmented_pixels_in_bbox INTEGER,  -- New: segmented pixels within bounding boxes
                total_bbox_pixels INTEGER,  -- New: total bounding box area
                
                -- Per-bbox granular metrics (JSON storage)
                bbox_pixel_ratios TEXT,  -- JSON array of per-bbox metrics
                min_bbox_pixel_ratio REAL,
                max_bbox_pixel_ratio REAL,
                std_bbox_pixel_ratio REAL,
                
                -- Color evaluation metrics  
                expected_color TEXT,
                detected_color TEXT,
                color_accuracy REAL,
                color_confidence REAL,
                
                -- Composition metrics
                expected_relation TEXT,
                composition_accuracy REAL,
                obj1_detected INTEGER,
                obj2_detected INTEGER,
                object1 TEXT,
                object2 TEXT,
                
                timestamp TEXT
            )
        """)
        
        # Keep experiment aggregates table but simpler
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                model_name TEXT,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                condition_type TEXT,
                
                total_images INTEGER,
                mean_count_accuracy REAL,
                mean_pixel_ratio REAL,
                mean_coverage_percentage REAL,
                mean_color_accuracy REAL,
                mean_composition_accuracy REAL,
                
                timestamp TEXT
            )
        """)

        # Table for per-image dice scores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS segmentation_dice (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                condition_id INTEGER,
                image_path TEXT,
                model_name TEXT,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                class_name TEXT,
                dice_score REAL,
                timestamp TEXT
            )
        """)

        # Table for average dice scores per class
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS segmentation_dice_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                class_name TEXT,
                mean_dice REAL,
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
        if "Segmentation Dice" in metrics:
            self.evaluate_segmentation_dice(experiment_id)
        
        # Compute experiment summary once at the end
        self._compute_experiment_summary(experiment_id)

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
            
            # Store FID score in experiment summary directly since it's a configuration-level metric
            cursor = self.evaluation_conn.cursor()
            cursor.execute("""
                INSERT INTO experiment_summary (
                    experiment_id, model_name, guidance_scale, num_inference_steps, condition_type,
                    total_images, mean_count_accuracy, mean_pixel_ratio, mean_coverage_percentage,
                    mean_color_accuracy, mean_composition_accuracy, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, model_name, guidance_scale, num_steps, 'FID',
                  len(image_paths), fid_score, None, None, None, None, datetime.now().isoformat()))
            self.evaluation_conn.commit()

    def evaluate_object_count(self, experiment_id: str):
        """
        Evaluates if the number of generated objects matches the number in the prompt.
        Also performs segmentation pixel analysis.
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
            cond_cursor.execute("SELECT type, number, object FROM conditions WHERE id = ?", (cond_id,))
            cond_result = cond_cursor.fetchone()
            if not cond_result or not cond_result[0].startswith('count_prompt'):
                continue
            
            condition_type, expected_number, target_object = cond_result
            variant = condition_type.split('_')[-1] if '_' in condition_type else 'base'

            if not os.path.exists(image_path):
                print(f"Skipping non-existent image: {image_path}")
                continue

            # Get target class ID for improved filtering
            target_class_id = self.yolo_class_to_id.get(target_object) if self.yolo_class_to_id else None
            
            # Use higher confidence threshold and class filtering for better detection quality
            conf_threshold = 0.6  # Increased from default 0.25 to reduce false positives
            
            # Get segmentation data with improved parameters
            seg_data = obj_counter.count_and_segment_objects(
                image_path, 
                target_object=target_object,  # Keep for backward compatibility
                target_class_id=target_class_id,  # More efficient class filtering
                conf_threshold=conf_threshold  # Higher confidence threshold
            )
            detected_count = seg_data['count']
            detections = seg_data['detections']
            image_shape = seg_data['image_shape']
            
            # Calculate all metrics
            count_accuracy = 1.0 if detected_count == expected_number else 0.0
            
            # Pixel analysis
            pixel_metrics = None
            if detections and image_shape:
                pixel_metrics = self._analyze_segmentation_pixels(detections, image_shape, target_object)
            
            # Log comprehensive evaluation for this image
            self._log_image_evaluation(
                experiment_id=experiment_id,
                condition_id=cond_id,
                condition_type=condition_type,
                variant=variant,
                image_path=image_path,
                model_name=model,
                guidance_scale=gs,
                num_inference_steps=steps,
                # Count metrics
                expected_count=expected_number,
                detected_count=detected_count,
                count_accuracy=count_accuracy,
                target_object=target_object,
                # Segmentation metrics
                pixel_metrics=pixel_metrics
            )
            
            print(f"Image: {os.path.basename(image_path)}, Expected: {expected_number} {target_object}, Found: {detected_count}, Accuracy: {count_accuracy}, Variant: {variant} (conf≥{conf_threshold})")
            if pixel_metrics:
                print(f"  Pixel Analysis - Seg/BBox Ratio: {pixel_metrics['pixel_ratio']:.4f}, Coverage: {pixel_metrics['coverage_percentage']:.2f}%")
    
    def _analyze_segmentation_pixels(self, detections: List[Dict], image_shape: Tuple[int, int], target_object: str) -> Dict[str, float]:
        """
        Analyzes pixel-level segmentation results with per-bbox granularity.
        
        Returns:
            Dict containing pixel analysis metrics where pixel_ratio is based on 
            segmented pixels to bounding box area ratio (averaged across all objects)
        """
        height, width = image_shape
        total_image_pixels = height * width
        
        # Create combined mask for all target objects (for coverage calculation)
        combined_mask = np.zeros((height, width), dtype=bool)
        
        # For aggregated metrics
        total_segmented_pixels = 0
        total_bbox_pixels = 0
        target_object_count = 0
        
        # For per-bbox metrics
        bbox_metrics = []
        
        for bbox_idx, detection in enumerate(detections):
            if detection['class_name'] == target_object:
                mask = detection['mask']
                box = detection['box']  # [x1, y1, x2, y2]
                confidence = detection.get('confidence', 0.0)
                
                # Resize mask to image dimensions if needed
                if mask.shape != (height, width):
                    mask_resized = cv2.resize(mask.astype(np.uint8), (width, height))
                    mask_bool = mask_resized > 0.5
                else:
                    mask_bool = mask > 0.5
                
                # Update combined mask for coverage calculation
                combined_mask = np.logical_or(combined_mask, mask_bool)
                
                # Calculate bounding box area
                x1, y1, x2, y2 = box
                bbox_width = max(0, x2 - x1)
                bbox_height = max(0, y2 - y1)
                bbox_area = bbox_width * bbox_height
                
                # Count segmented pixels within this bounding box
                bbox_mask = np.zeros((height, width), dtype=bool)
                x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
                x2_int, y2_int = min(width, int(x2)), min(height, int(y2))
                bbox_mask[y1_int:y2_int, x1_int:x2_int] = True
                
                # Count segmented pixels within the bounding box
                segmented_in_bbox = np.sum(mask_bool & bbox_mask)
                
                # Calculate per-bbox pixel ratio
                bbox_pixel_ratio = segmented_in_bbox / bbox_area if bbox_area > 0 else 0.0
                
                # Store per-bbox metrics
                bbox_metrics.append({
                    'bbox_id': bbox_idx,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(confidence),
                    'pixel_ratio': float(bbox_pixel_ratio),
                    'segmented_pixels': int(segmented_in_bbox),
                    'bbox_pixels': int(bbox_area),
                    'bbox_area_percentage': float(bbox_area / total_image_pixels * 100) if total_image_pixels > 0 else 0.0
                })
                
                # Update aggregated totals
                total_segmented_pixels += segmented_in_bbox
                total_bbox_pixels += bbox_area
                target_object_count += 1
        
        # Count total pixels covered by target objects (for coverage calculation)
        total_coverage_pixels = np.sum(combined_mask)
        
        # Calculate aggregated metrics
        pixel_ratio = total_segmented_pixels / total_bbox_pixels if total_bbox_pixels > 0 else 0.0
        coverage_percentage = (total_coverage_pixels / total_image_pixels * 100) if total_image_pixels > 0 else 0.0
        
        return {
            # Existing aggregated metrics
            'target_pixels': int(total_coverage_pixels),
            'total_pixels': total_image_pixels,
            'pixel_ratio': pixel_ratio,
            'coverage_percentage': coverage_percentage,
            'num_objects': target_object_count,
            'segmented_pixels_in_bbox': int(total_segmented_pixels),
            'total_bbox_pixels': int(total_bbox_pixels),
            
            # New per-bbox granular data
            'bbox_pixel_ratios': json.dumps(bbox_metrics) if bbox_metrics else None,
            
            # Additional summary statistics
            'min_bbox_pixel_ratio': min([b['pixel_ratio'] for b in bbox_metrics]) if bbox_metrics else None,
            'max_bbox_pixel_ratio': max([b['pixel_ratio'] for b in bbox_metrics]) if bbox_metrics else None,
            'std_bbox_pixel_ratio': float(np.std([b['pixel_ratio'] for b in bbox_metrics])) if len(bbox_metrics) > 1 else 0.0
        }

    def evaluate_color_classification(self, experiment_id: str):
        """
        Evaluates if the color of objects in generated images matches the expected color.
        """
        print(f"Evaluating Object Color Classification Accuracy for experiment: {experiment_id}")
        color_classifier = ObjectColorClassifier()
        object_segmenter = ObjectSegmenter()
        output_cursor = self.output_conn.cursor()
        cond_cursor = self.conditioning_conn.cursor()
            
        output_cursor.execute("""
            SELECT condition_id, image_path, model_name, guidance_scale, num_inference_steps
            FROM dm_outputs
            WHERE experiment_id = ?
        """, (experiment_id,))
        
        for cond_id, image_path, model, gs, steps in output_cursor.fetchall():
            cond_cursor.execute("SELECT type, color1, object FROM conditions WHERE id = ?", (cond_id,))
            cond_result = cond_cursor.fetchone()
            if not cond_result or cond_result[0] != 'color_prompt':
                continue

            condition_type, expected_color, target_object = cond_result
            if not os.path.exists(image_path):
                print(f"Skipping non-existent image: {image_path}")
                continue

            seg_data = object_segmenter.segment_objects_in_image(image_path, target_object=target_object)
            detections = seg_data['detections']
            image_shape = seg_data['image_shape']
            if detections:
                best_detection = max(detections, key=lambda x: x['confidence'])

            detected_color, confidence = color_classifier.classify_color(image_path, object_name=target_object, detections=detections)

            # Accuracy is 1 if colors match, 0 otherwise.
            color_accuracy = 1.0 if detected_color == expected_color else 0.0
            
            print(f"Image: {os.path.basename(image_path)}, Expected: {expected_color} {target_object}, Found: {detected_color}, Accuracy: {color_accuracy}")


            # Pixel analysis
            pixel_metrics = None
            if detections and image_shape:
                pixel_metrics = self._analyze_segmentation_pixels([best_detection], image_shape, target_object)

            # Log comprehensive evaluation for this image
            self._log_image_evaluation(
                experiment_id=experiment_id,
                condition_id=cond_id,
                condition_type=condition_type,
                image_path=image_path,
                model_name=model,
                guidance_scale=gs,
                num_inference_steps=steps,
                # Color metrics
                expected_color=expected_color,
                detected_color=detected_color,
                color_accuracy=color_accuracy,
                color_confidence=confidence,
                target_object=target_object,
                pixel_metrics=pixel_metrics
            )
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

            condition_type, obj1_name, num1, relation, obj2_name, num2 = cond_result
            obj1_singular = obj1_name.rstrip('s')
            obj2_singular = obj2_name.rstrip('s')

            if not os.path.exists(image_path):
                print(f"Skipping non-existent image: {image_path}")
                continue
            
            # Use higher confidence for more reliable object detection in composition analysis
            detected_objects = obj_detector.detect_objects_with_boxes(
                image_path, 
                conf_threshold=0.6  # Higher confidence to reduce false positives
            )
            
            obj1_boxes = [d['box'] for d in detected_objects if d['class_name'] == obj1_singular]
            obj2_boxes = [d['box'] for d in detected_objects if d['class_name'] == obj2_singular]

            # Basic check: did we detect at least one of each required object?
            obj1_detected = len(obj1_boxes) > 0
            obj2_detected = len(obj2_boxes) > 0
            
            if not obj1_detected or not obj2_detected:
                composition_accuracy = 0.0
            else:
                # If objects are detected, check if any pair satisfies the spatial relationship.
                composition_accuracy = 0.0 # Assume failure until a correct pair is found
                for b1 in obj1_boxes:
                    for b2 in obj2_boxes:
                        if self._check_spatial_relation(b1, b2, relation):
                            composition_accuracy = 1.0
                            break # Found a valid pair, no need to check others
                    if composition_accuracy == 1.0:
                        break

            print(f"Image: {os.path.basename(image_path)}, Relation: '{relation}', Accuracy: {composition_accuracy}")

            # Log comprehensive evaluation for this image
            self._log_image_evaluation(
                experiment_id=experiment_id,
                condition_id=cond_id,
                condition_type=condition_type,
                image_path=image_path,
                model_name=model,
                guidance_scale=gs,
                num_inference_steps=steps,
                # Composition metrics
                expected_relation=relation,
                composition_accuracy=composition_accuracy,
                obj1_detected=1 if obj1_detected else 0,
                obj2_detected=1 if obj2_detected else 0,
                object1=obj1_singular,
                object2=obj2_singular
            )

    def evaluate_segmentation_dice(self, experiment_id: str):
        """Computes per-class Dice score between conditioning masks and generated images."""
        print(f"Evaluating Segmentation Dice for experiment: {experiment_id}")
        # Segment generated images using the finetuned YOLO model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        yolo_model = YOLO(
            "/mnt/projects/mlmi/dmcaf_laparoscopic/cholecseg8k-yolov8/runs/segment/train/weights/best.pt"
        )
        output_cursor = self.output_conn.cursor()
        cond_cursor = self.conditioning_conn.cursor()

        output_cursor.execute(
            """
            SELECT condition_id, image_path, model_name, guidance_scale, num_inference_steps
            FROM dm_outputs
            WHERE experiment_id = ?
            AND model_name LIKE ?
            """,
            (experiment_id, "laparoscopic"),
        )

        dice_sums: Dict[str, float] = {name: 0.0 for name in CLASS_NAME_MAPPING.values()}
        dice_counts: Dict[str, int] = {name: 0 for name in CLASS_NAME_MAPPING.values()}

        for cond_id, image_path, model, gs, steps in output_cursor.fetchall():
            cond_cursor.execute(
                "SELECT segmentation_path FROM conditions WHERE id = ?", (cond_id,)
            )
            seg_row = cond_cursor.fetchone()
            if not seg_row or not seg_row[0] or not os.path.exists(seg_row[0]):
                continue

            gt_masks = self._load_condition_masks(seg_row[0])
            if not gt_masks:
                continue

            # Only evaluate classes that are present in the ground truth segmentation
            gt_masks = {name: mask for name, mask in gt_masks.items() if mask.any()}
            if not gt_masks:
                continue

            h, w = next(iter(gt_masks.values())).shape

            # Run YOLO segmentation on the generated image
            results = yolo_model(
                image_path,
                save=False,
                device=device,
                show_labels=False,
                verbose=False,
            )       
            result = results[0]
            detections = []
            if result.masks:
                masks = result.masks.data
                classes = result.boxes.cls.to(torch.int64).cpu().tolist()
                h, w = masks.shape[-2:]
                seg_mask = np.zeros((h, w, 3), dtype=np.uint8)
                seg_mask[:] = CLASS_COLOR_MAPPING[0]
                for mask, cls in zip(masks, classes):
                    class_name = CLASS_NAME_MAPPING.get(cls, CLASS_NAME_MAPPING[0])
                    detections.append({
                        "class_name": class_name,
                        "mask": mask.cpu().numpy(),
                    })
                    color = CLASS_COLOR_MAPPING.get(cls, CLASS_COLOR_MAPPING[0])
                    seg_mask[mask.bool().cpu().numpy()] = color
                    
                seg_mask_bgr = cv2.cvtColor(seg_mask, cv2.COLOR_RGB2BGR)
                base, ext = os.path.splitext(image_path)
                segmented_path = f"{base}_segmented{ext}"
                cv2.imwrite(segmented_path, seg_mask_bgr)

            pred_masks = self._combine_masks_by_class(detections, h, w)

            cursor = self.evaluation_conn.cursor()
            for class_name, gt in gt_masks.items():
                pred = pred_masks.get(class_name, np.zeros((h, w), dtype=bool))
                dice = self._dice_score(pred, gt)

                cursor.execute(
                    """
                    INSERT INTO segmentation_dice (
                        experiment_id, condition_id, image_path, model_name,
                        guidance_scale, num_inference_steps, class_name, dice_score, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        cond_id,
                        image_path,
                        model,
                        gs,
                        steps,
                        class_name,
                        dice,
                        datetime.now().isoformat(),
                    ),
                )

                dice_sums[class_name] += dice
                dice_counts[class_name] += 1

        # Store per-class averages
        cursor = self.evaluation_conn.cursor()
        for class_name, total in dice_sums.items():
            count = dice_counts[class_name]
            if count == 0:
                continue
            cursor.execute(
                """
                INSERT INTO segmentation_dice_summary (
                    experiment_id, class_name, mean_dice, timestamp
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    class_name,
                    total / count,
                    datetime.now().isoformat(),
                ),
            )

        self.evaluation_conn.commit()

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

    def _load_condition_masks(self, seg_path: str) -> Dict[str, np.ndarray]:
        """Load segmentation masks for each class from conditioning image."""
        image = cv2.imread(seg_path)
        if image is None:
            return {}
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = {}
        for class_id, color in CLASS_COLOR_MAPPING.items():
            class_name = CLASS_NAME_MAPPING[class_id]
            masks[class_name] = np.all(image == color, axis=-1)
        return masks

    def _combine_masks_by_class(self, detections: List[Dict[str, Any]], h: int, w: int) -> Dict[str, np.ndarray]:
        """Combine multiple instance masks into a single mask per class."""
        masks: Dict[str, np.ndarray] = {}
        for det in detections:
            class_name = det['class_name']
            mask = det['mask']
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h)) > 0.5
            else:
                mask = mask > 0.5
            if class_name in masks:
                masks[class_name] = np.logical_or(masks[class_name], mask)
            else:
                masks[class_name] = mask

        # Background mask is everything not covered by other masks
        bg_mask = np.ones((h, w), dtype=bool)
        for m in masks.values():
            bg_mask &= ~m
        masks.setdefault(CLASS_NAME_MAPPING[0], bg_mask)
        return masks

    def _dice_score(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Dice coefficient between two binary masks."""
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        if pred_sum + gt_sum == 0:
            return 1.0
        intersection = np.logical_and(pred, gt).sum()
        return 2.0 * intersection / (pred_sum + gt_sum)

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

    def _log_image_evaluation(self, experiment_id: str, condition_id: int, condition_type: str,
                             image_path: str, model_name: str, guidance_scale: float, 
                             num_inference_steps: int, **kwargs):
        """
        Log comprehensive evaluation results for a single image.
        """
        cursor = self.evaluation_conn.cursor()
        
        # Extract metrics from kwargs and handle None case
        pixel_metrics = kwargs.get('pixel_metrics', {})
        if pixel_metrics is None:
            pixel_metrics = {}
        
        cursor.execute("""
            INSERT INTO image_evaluations (
                experiment_id, condition_id, condition_type, variant, image_path, model_name, 
                guidance_scale, num_inference_steps,
                expected_count, detected_count, count_accuracy, target_object,
                target_pixels, total_pixels, pixel_ratio, coverage_percentage,
                segmented_pixels_in_bbox, total_bbox_pixels,
                bbox_pixel_ratios, min_bbox_pixel_ratio, max_bbox_pixel_ratio, std_bbox_pixel_ratio,
                expected_color, detected_color, color_accuracy, color_confidence,
                expected_relation, composition_accuracy, obj1_detected, obj2_detected,
                object1, object2, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, condition_id, condition_type, kwargs.get('variant', 'base'), image_path, model_name,
            guidance_scale, num_inference_steps,
            # Count metrics
            kwargs.get('expected_count'), kwargs.get('detected_count'), 
            kwargs.get('count_accuracy'), kwargs.get('target_object'),
            # Segmentation metrics
            pixel_metrics.get('target_pixels'), pixel_metrics.get('total_pixels'),
            pixel_metrics.get('pixel_ratio'), pixel_metrics.get('coverage_percentage'),
            pixel_metrics.get('segmented_pixels_in_bbox'), pixel_metrics.get('total_bbox_pixels'),
            # Per-bbox granular metrics
            pixel_metrics.get('bbox_pixel_ratios'), pixel_metrics.get('min_bbox_pixel_ratio'),
            pixel_metrics.get('max_bbox_pixel_ratio'), pixel_metrics.get('std_bbox_pixel_ratio'),
            # Color metrics  
            kwargs.get('expected_color'), kwargs.get('detected_color'),
            kwargs.get('color_accuracy'), kwargs.get('color_confidence'),
            # Composition metrics
            kwargs.get('expected_relation'), kwargs.get('composition_accuracy'),
            kwargs.get('obj1_detected'), kwargs.get('obj2_detected'),
            kwargs.get('object1'), kwargs.get('object2'),
            datetime.now().isoformat()
        ))
        self.evaluation_conn.commit()

    def _compute_experiment_summary(self, experiment_id: str):
        """
        Compute experiment-wide summary statistics from image evaluations.
        """
        cursor = self.evaluation_conn.cursor()
        
        # Group by model configuration, condition type, and variant
        cursor.execute("""
            SELECT model_name, guidance_scale, num_inference_steps, condition_type, variant,
                   COUNT(*) as total_images,
                   AVG(count_accuracy) as mean_count_accuracy,
                   AVG(pixel_ratio) as mean_pixel_ratio,
                   AVG(coverage_percentage) as mean_coverage_percentage,
                   AVG(color_accuracy) as mean_color_accuracy,
                   AVG(composition_accuracy) as mean_composition_accuracy
            FROM image_evaluations
            WHERE experiment_id = ?
            GROUP BY model_name, guidance_scale, num_inference_steps, condition_type, variant
        """, (experiment_id,))
        
        for row in cursor.fetchall():
            model, gs, steps, cond_type, variant, total, count_acc, pixel_ratio, coverage, color_acc, comp_acc = row
            
            # Create condition type with variant for summary
            condition_summary_type = f"{cond_type}_{variant}" if variant != 'base' else cond_type
            
            cursor.execute("""
                INSERT INTO experiment_summary (
                    experiment_id, model_name, guidance_scale, num_inference_steps, condition_type,
                    total_images, mean_count_accuracy, mean_pixel_ratio, mean_coverage_percentage,
                    mean_color_accuracy, mean_composition_accuracy, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, model, gs, steps, condition_summary_type, total, count_acc, 
                  pixel_ratio, coverage, color_acc, comp_acc, datetime.now().isoformat()))
        
        self.evaluation_conn.commit()


class Evaluator_Fundus:
    def __init__(self, monai_home: str, output_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monai_home = monai_home
        self.output_dir = output_dir

    def evaluate_outputs(self, experiment_id: str, metrics: List[str]):
        # if "FID" in metrics:
        #     self._evaluate_fid(experiment_id)
        if "DICE" in metrics:
            self._generate_segmentations(experiment_id)
            self._evaluate_dice(experiment_id)

        # Compute experiment summary once at the end
        # self._compute_experiment_summary(experiment_id)

    def _evaluate_dice(self, experiment_id: str):

        root = Path(self.output_dir)
        GT_DIR = root / "Images"  # ground truths (as stated)
        PRED_DIR = root / "SegmentationPreds"  # predicted masks
        RESULTS_FILE = root / f"{experiment_id}_dice_scores.txt"

        # --- Helpers ---
        def load_png(path: Path) -> torch.Tensor:
            """Load image, drop alpha if present, return grayscale float32 [H,W] in [0,1]."""
            img = plt.imread(path)
            if img.ndim == 3:  # RGB or RGBA
                if img.shape[-1] == 4:
                    img = img[..., :3]
                img = img.mean(-1)
            elif img.ndim == 1:  # rare: flattened
                side = int(math.isqrt(img.size))
                if side * side != img.size:
                    raise ValueError(f"{path.name}: cannot reshape {img.shape}")
                img = img.reshape(side, side)
            return torch.tensor(img, dtype=torch.float32)

        def to_one_hot(bin_mask: torch.Tensor) -> torch.Tensor:
            """bin_mask [H,W] -> one-hot [1,2,H,W] (bg, fg), float32."""
            bg = (1 - bin_mask).unsqueeze(0)
            fg = bin_mask.unsqueeze(0)
            return torch.stack([bg, fg], dim=0).unsqueeze(0).float()

        # collect files (File name must match)
        gt_files = sorted([p for p in GT_DIR.glob("*.png")])
        pred_files = sorted([p for p in PRED_DIR.glob("*.png")])

        if not gt_files:
            raise FileNotFoundError(f"No ground-truth PNGs in {GT_DIR}")
        if not pred_files:
            raise FileNotFoundError(f"No prediction PNGs in {PRED_DIR}")

        # build a filename->path map for predictions and align order by GT names
        pred_map = {p.name: p for p in pred_files}
        pairs = [(gt_fp, pred_map[gt_fp.name]) for gt_fp in gt_files if gt_fp.name in pred_map]
        if not pairs:
            raise ValueError("No filename overlap between ground truths and predictions.")
        if len(pairs) < len(gt_files):
            missing = {p.name for p in gt_files} - set(pred_map.keys())
            print(f"Warning: {len(missing)} GT files missing predictions (e.g., {next(iter(missing))})")


        # --- DICE ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dice_metric = DiceMetric(include_background=False, reduction="none")

        scores, names = [], []
        print(f"Total evaluated pairs: {len(pairs)}  |  Device: {device}")
        for idx, (gt_fp, pred_fp) in enumerate(tqdm(pairs, desc="Evaluating DICE")):
            gt_gray = load_png(gt_fp)
            pr_gray = load_png(pred_fp)

            # binarize: non-zero -> 1
            gt_bin = (gt_gray > 0).to(torch.uint8)
            pr_bin = (pr_gray > 0).to(torch.uint8)

            # skip empty GT masks
            if gt_bin.sum() == 0:
                continue

            y = to_one_hot(gt_bin).to(device)  # [1,2,H,W]
            y_pred = to_one_hot(pr_bin).to(device)

            dice = dice_metric(y_pred, y).item()
            scores.append(float(dice))
            names.append(gt_fp.name)

            # save summary
            RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

        with open(RESULTS_FILE, "w") as f:
            for n, s in zip(names, scores):
                f.write(f"{n}: Dice = {s:.4f}\n")

            if scores:
                arr = np.array(scores, dtype=float)
                f.write("\n==== Summary Statistics ====\n")
                f.write(f"Samples Evaluated       : {len(arr)}\n")
                f.write(f"Mean Dice               : {arr.mean():.4f}\n")
                f.write(f"Median Dice             : {np.median(arr):.4f}\n")
                f.write(f"Std Dev Dice            : {arr.std():.4f}\n")
                f.write(f"Min Dice                : {arr.min():.4f}\n")
                f.write(f"Max Dice                : {arr.max():.4f}\n")
                f.write(f"Dice > 0.90             : {(arr > 0.9).mean() * 100:.2f}%\n")
                f.write(f"Dice > 0.80             : {(arr > 0.8).mean() * 100:.2f}%\n")

        print(f"Saved DICE summary to: {RESULTS_FILE}")


    def _generate_segmentations(self, experiment_id: str) -> None:
        images_dir = Path(self.output_dir) / "Images"
        seg_dir = Path(self.output_dir) / "SegmentationPreds"

        # iterate all images in output_dir/Images
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        img_files = [p for p in images_dir.iterdir() if p.suffix.lower() in exts]
        if not img_files:
            raise FileNotFoundError(f"No images found in {images_dir}")

        # if SegmentationPreds already exists with files, move to Old/<timestamp>
        if seg_dir.exists() and any(seg_dir.iterdir()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = Path(self.output_dir) / "Old" / f"SegmentationPreds_{timestamp}"
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(seg_dir), str(archive_dir))
            print(f"Moved old predictions to {archive_dir}")


        seg_dir.mkdir(parents=True, exist_ok=True)

        # load model + processor once
        model_id = "pamixsun/segformer_for_optic_disc_cup_segmentation"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device).eval()



        for img_path in tqdm(img_files, desc="Segmenting generated images"):
            # force 3-channel (model expects RGB)
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            inputs = processor(rgb, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                up = nn.functional.interpolate(
                    logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
                )
                pred = up.argmax(dim=1)[0]  # (H,W), class ids

            # save with same filename in SegmentationPreds
            out_path = seg_dir / img_path.name
            plt.imsave(out_path, pred.detach().cpu().numpy(), cmap="gray", vmin=0, vmax=2)

import os
import random
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict

import pandas as pd


class ConditionGenerator:
    def __init__(self, conditioning_db_path):
        self.db_path = conditioning_db_path
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()
        self.numbers = list(range(1, 5))
        # Construct path to yolo_classes.json relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        classes_json_path = os.path.join(current_dir, '..', 'assets', 'yolo_classes.json')
        classes_json_path_plural = os.path.join(current_dir, '..', 'assets', 'yolo_classes_plural.json')
        colors_json_path = os.path.join(current_dir, '..', 'assets', 'colors.json')
        
        with open(classes_json_path, 'r') as f:
            data = json.load(f)
            self.objects = list(data['class'].values())
        
        with open(classes_json_path_plural, 'r') as f:
            data_plural = json.load(f)
            self.objects_plural = list(data_plural['class'].values())

        with open(colors_json_path, 'r') as f:
            data_colors = json.load(f)
            self.colors = data_colors['colors']

        self.singular_to_plural = dict(zip(self.objects, self.objects_plural))
        self.relationships = ['on top of', 'above', 'below', 'to the left of', 'to the right of', 'next to']        
        self.backgrounds = ['table', 'mountains', 'river']

    def _create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                type TEXT,
                prompt TEXT,
                number INTEGER,
                object TEXT,
                color1 TEXT,
                background TEXT,
                image_path TEXT,
                segmentation_path TEXT,
                timestamp TEXT,
                relationship TEXT,
                object2 TEXT,
                color2 TEXT,
                number2 INTEGER
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                experiment_id TEXT PRIMARY KEY,
                timestamp TEXT,
                n_text_prompts INTEGER,
                n_segmentation_maps INTEGER,
                notes TEXT
            )
        """)
        self.conn.commit()

    def generate_experiment(
        self,
        experiment_id: str,
        n_text: int,
        n_compositional: int,
        n_seg: int,
        n_color: int,
        n_count: int,
        count_params: Dict = None,
        dataset_path: Optional[str] = None,
    ) -> List[Dict]:
        """Generate an experiment configuration.

        If ``dataset_path`` is provided, segmentation map entries are read from a
        ``train.jsonl`` file located at that path. Each line in the file is
        expected to be a JSON object containing ``text`` and
        ``conditioning_image`` fields. The ``text`` field is stored as the
        prompt and the ``conditioning_image`` as the segmentation path.
        """
        conditions = []
        conditions += self._generate_text_prompts(experiment_id, n_text)
        conditions += self._generate_compositional_prompts(experiment_id, n_compositional)
        conditions += self._generate_segmentation_maps(
            experiment_id, n_seg, dataset_path=dataset_path
        )
        conditions += self._generate_color_prompts(experiment_id, n_color)
        conditions += self._generate_count_prompts(
            experiment_id, n_count, **(count_params or {})
        )
        self._log_metadata(
            experiment_id,
            n_text + n_compositional + n_color + n_count,
            max(n_seg, len([c for c in conditions if c["type"] == "segmentation_map"])),
        )
        return conditions

    def _generate_text_prompts(self, experiment_id: str, count: int) -> List[Dict]:
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        conditions = []
        for _ in range(count):
            number = random.choice(self.numbers)
            obj_singular = random.choice(self.objects)
            bg = random.choice(self.backgrounds)
            obj_display = self.singular_to_plural[obj_singular] if number > 1 else obj_singular
            prompt = f"{number} {obj_display} in front of the {bg}"
            cursor.execute("""
                INSERT INTO conditions (experiment_id, type, prompt, number, object, background, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, "text_prompt", prompt, number, obj_singular, bg, timestamp))
            conditions.append({
                "experiment_id": experiment_id,
                "type": "text_prompt",
                "prompt": prompt,
                "number": number,
                "object": obj_singular,
                "background": bg,
                "timestamp": timestamp
            })
        self.conn.commit()
        return conditions

    def _generate_count_prompts(self, experiment_id: str, count: int, **kwargs) -> List[Dict]:
        """
        Generates count prompts with configurable variations.
        
        Args:
            count: Base number of prompts to generate
            **kwargs: Configuration from YAML:
                - include_numeral_variant: bool (default False)
                - include_background_variant: bool (default False) 
                - object_indices: List[int] (default [39-45])
                - backgrounds: List[str] (default ["unicolor background"])
                - number_range: List[int] (default [1-4])
        """
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        conditions = []
        
        # Extract configuration with robust defaults
        include_numeral = kwargs.get('include_numeral_variant', False)
        include_background = kwargs.get('include_background_variant', False)
        object_indices = kwargs.get('object_indices', list(range(39, 46)))
        backgrounds = kwargs.get('backgrounds', ["unicolor background"])
        number_range = kwargs.get('number_range', [1, 2, 3, 4])
        
        # Validate object indices
        valid_indices = [i for i in object_indices if 0 <= i < len(self.objects)]
        if not valid_indices:
            print("Warning: No valid object indices provided, using default range")
            valid_indices = list(range(39, 46))
        
        selected_objects = [self.objects[i] for i in valid_indices]
        
        # Helper function to convert numbers to words
        number_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        
        for _ in range(count):
            number = random.choice(number_range)
            obj_singular = random.choice(selected_objects)
            obj_display = self.singular_to_plural[obj_singular] if number > 1 else obj_singular
            
            # Get number word, fallback to numeral if not in dictionary
            number_word = number_words.get(number, str(number))
            
            # Base prompt: "a photo of <number in text> <object>"
            base_prompt = f"a photo of {number_word} {obj_display}"
            conditions.extend(self._create_count_condition(
                experiment_id, base_prompt, number, obj_singular, "base", timestamp, cursor
            ))
            
            # Variant 1: Numeral version
            if include_numeral:
                numeral_prompt = f"a photo of {number} {obj_display}"
                conditions.extend(self._create_count_condition(
                    experiment_id, numeral_prompt, number, obj_singular, "numeral", timestamp, cursor
                ))
            
            # Variant 2: Background version  
            if include_background:
                bg = random.choice(backgrounds)
                bg_prompt = f"a photo of {number_word} {obj_display} in front of {bg}"
                conditions.extend(self._create_count_condition(
                    experiment_id, bg_prompt, number, obj_singular, "background", timestamp, cursor
                ))
        
        self.conn.commit()
        return conditions

    def _create_count_condition(self, experiment_id: str, prompt: str, number: int, 
                               obj_singular: str, variant: str, timestamp: str, cursor) -> List[Dict]:
        """Helper to create count condition with proper type annotation."""
        condition_type = f"count_prompt_{variant}"
        
        cursor.execute("""
            INSERT INTO conditions (experiment_id, type, prompt, number, object, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (experiment_id, condition_type, prompt, number, obj_singular, timestamp))
        
        return [{
            "experiment_id": experiment_id,
            "type": condition_type,
            "prompt": prompt,
            "number": number,
            "object": obj_singular,
            "variant": variant,
            "timestamp": timestamp
        }]
    

    def _generate_color_prompts(self, experiment_id: str, count: int) -> List[Dict]:
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        conditions = []
        for _ in range(count):
            number = 1
            color = random.choice(self.colors)['name']
            obj_singular = random.choice(self.objects)
            bg = random.choice(self.backgrounds)
            prompt = f"A {color} {obj_singular} in front of the {bg}"
            cursor.execute("""
                INSERT INTO conditions (experiment_id, type, prompt, number, object, color1, background, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, "color_prompt", prompt, number, obj_singular, color, bg, timestamp))
            conditions.append({
                "experiment_id": experiment_id,
                "type": "text_prompt",
                "prompt": prompt,
                "number": number,
                "object": obj_singular,
                "color1": color,
                "background": bg,
                "timestamp": timestamp
            })
        self.conn.commit()
        return conditions
    
    def _generate_compositional_prompts(self, experiment_id: str, count: int) -> List[Dict]:
        """
        Generates compositional prompts and stores their components in the database.
        """
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        conditions = []
        for _ in range(count):
            num1 = random.choice(self.numbers)
            obj1_singular = random.choice(self.objects)
            num2 = random.choice(self.numbers)
            obj2_singular = random.choice(self.objects)
            # Ensure objects are not the same
            while obj1_singular == obj2_singular:
                obj2_singular = random.choice(self.objects)

            obj1_display = self.singular_to_plural[obj1_singular] if num1 > 1 else obj1_singular
            obj2_display = self.singular_to_plural[obj2_singular] if num2 > 1 else obj2_singular

            relation = random.choice(self.relationships)
            bg = random.choice(self.backgrounds)

            prompt = f"{num1} {obj1_display} {relation} {num2} {obj2_display} on {bg}"

            cursor.execute("""
                INSERT INTO conditions (
                    experiment_id, type, prompt,
                    number, object, relationship, number2, object2, background,
                    timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, "compositional_prompt", prompt,
                num1, obj1_singular, relation, num2, obj2_singular, bg,
                timestamp
            ))
            conditions.append({
                "experiment_id": experiment_id,
                "type": "compositional_prompt",
                "prompt": prompt,
                "number": num1,
                "object": obj1_singular,
                "relationship": relation,
                "number2": num2,
                "object2": obj2_singular,
                "background": bg,
                "timestamp": timestamp
            })
        self.conn.commit()
        return conditions

    def _generate_segmentation_maps(
        self, experiment_id: str, count: int, dataset_path: Optional[str] = None
    ) -> List[Dict]:
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        conditions: List[Dict] = []

        if dataset_path:
            train_file = os.path.join(dataset_path, "train.jsonl")
            if not os.path.exists(train_file):
                raise FileNotFoundError(
                    f"train.jsonl not found in dataset path: {dataset_path}"
                )
            with open(train_file, "r") as f:
                lines = f.readlines()

            # Randomly sample lines from the dataset. If ``count`` is zero or
            # greater than the available number of entries, fall back to using
            # the entire dataset. Sampling without replacement ensures unique
            # elements when a limit is specified.
            if count and len(lines) > count:
                sampled_lines = random.sample(lines, count)
            else:
                sampled_lines = lines

            for line in sampled_lines:
                record = json.loads(line)
                prompt = record.get("text", "")
                img_rel = record.get("image")
                seg_rel = record.get("conditioning_image")
                image_path = (
                    os.path.join(dataset_path, img_rel) if img_rel else None
                )
                seg_path = (
                    os.path.join(dataset_path, seg_rel) if seg_rel else None
                )
                cursor.execute(
                    """
                INSERT INTO conditions (experiment_id, type, prompt, image_path, segmentation_path, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        experiment_id,
                        "segmentation_map",
                        prompt,
                        image_path,
                        seg_path,
                        timestamp,
                    ),
                )
                conditions.append(
                    {
                        "experiment_id": experiment_id,
                        "type": "segmentation_map",
                        "prompt": prompt,
                        "image_path": image_path,
                        "segmentation_path": seg_path,
                        "timestamp": timestamp,
                    }
                )
        else:
            for _ in range(count):
                image_path = "/mnt/data/example_image.png"
                seg_path = "/mnt/data/example_segmentation.png"
                cursor.execute(
                    """
                INSERT INTO conditions (experiment_id, type, image_path, segmentation_path, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                    (experiment_id, "segmentation_map", image_path, seg_path, timestamp),
                )
                conditions.append(
                    {
                        "experiment_id": experiment_id,
                        "type": "segmentation_map",
                        "image_path": image_path,
                        "segmentation_path": seg_path,
                        "timestamp": timestamp,
                    }
                )

        self.conn.commit()
        return conditions

    def _log_metadata(self, experiment_id: str, n_text: int, n_seg: int):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO metadata (experiment_id, timestamp, n_text_prompts, n_segmentation_maps, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (experiment_id, datetime.now().isoformat(), n_text, n_seg, "Auto-generated experiment"))
        self.conn.commit()

    def load_conditions(self, experiment_id: str, condition_type: Optional[str] = None) -> List[Dict]:
        cursor = self.conn.cursor()
        if condition_type:
            cursor.execute("""
                SELECT * FROM conditions WHERE experiment_id = ? AND type = ?
            """, (experiment_id, condition_type))
        else:
            cursor.execute("""
                SELECT * FROM conditions WHERE experiment_id = ?
            """, (experiment_id,))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

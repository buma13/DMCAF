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

    def generate_experiment(self, experiment_id: str, n_text: int, n_compositional: int, n_seg: int, n_color: int) -> List[Dict]:
        conditions = []
        conditions += self._generate_text_prompts(experiment_id, n_text)
        conditions += self._generate_compositional_prompts(experiment_id, n_compositional)
        conditions += self._generate_segmentation_maps(experiment_id, n_seg)
        conditions += self._generate_color_prompts(experiment_id, n_color)
        self._log_metadata(experiment_id, n_text + n_compositional + n_color, n_seg)
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

    def _generate_segmentation_maps(self, experiment_id: str, count: int) -> List[Dict]:
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        conditions = []
        for _ in range(count):
            image_path = "/mnt/data/example_image.png"
            seg_path = "/mnt/data/example_segmentation.png"
            cursor.execute("""
                INSERT INTO conditions (experiment_id, type, image_path, segmentation_path, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (experiment_id, "segmentation_map", image_path, seg_path, timestamp))
            conditions.append({
                "experiment_id": experiment_id,
                "type": "segmentation_map",
                "image_path": image_path,
                "segmentation_path": seg_path,
                "timestamp": timestamp
            })
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

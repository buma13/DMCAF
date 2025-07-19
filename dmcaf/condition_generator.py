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
        self.numbers = list(range(1, 10))
        # Construct path to yolo_classes.json relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, '..', 'assets', 'yolo_classes.json')
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            self.objects = list(data['class'].values())
        
        self.backgrounds = ['school', 'mountains', 'river']

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
                background TEXT,
                image_path TEXT,
                segmentation_path TEXT,
                timestamp TEXT
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

    def generate_experiment(self, experiment_id: str, n_text: int, n_seg: int) -> List[Dict]:
        conditions = []
        conditions += self._generate_text_prompts(experiment_id, n_text)
        conditions += self._generate_segmentation_maps(experiment_id, n_seg)
        self._log_metadata(experiment_id, n_text, n_seg)
        return conditions

    def _generate_text_prompts(self, experiment_id: str, count: int) -> List[Dict]:
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        conditions = []
        for _ in range(count):
            number = random.choice(self.numbers)
            obj = random.choice(self.objects)
            bg = random.choice(self.backgrounds)
            prompt = f"{number} {obj} in front of the {bg}"
            cursor.execute("""
                INSERT INTO conditions (experiment_id, type, prompt, number, object, background, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (experiment_id, "text_prompt", prompt, number, obj, bg, timestamp))
            conditions.append({
                "experiment_id": experiment_id,
                "type": "text_prompt",
                "prompt": prompt,
                "number": number,
                "object": obj,
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

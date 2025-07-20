import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

from PIL import Image
import torch
from diffusers import StableDiffusionPipeline


class DMRunner:
    def __init__(self, conditioning_db_path: str, output_db_path: str, output_dir: str):
        self.conditioning_conn = sqlite3.connect(conditioning_db_path)
        self.output_conn = sqlite3.connect(output_db_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._create_output_table()

    def _create_output_table(self):
        cursor = self.output_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dm_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                model_name TEXT,
                guidance_scale REAL,
                num_inference_steps INTEGER,
                condition_id INTEGER,
                prompt TEXT,
                image_path TEXT,
                timestamp TEXT
            )
        """)
        self.output_conn.commit()

    def run_experiment(self, experiment_id: str, model_configs: List[Dict[str, Any]], condition_sets: List[Dict[str, Any]] | None = None):
        cursor = self.conditioning_conn.cursor()
        conditions = []
        if condition_sets:
            for cs in condition_sets:
                cs_id = cs["condition_set_id"]
                limit = cs.get("limit_number_of_conditions")
                # The default is to run on all types of conditions.
                types = cs.get("types")

                query = "SELECT id, prompt FROM conditions WHERE experiment_id = ?"
                params: List[Any] = [cs_id]

                if types:
                    placeholders = ','.join('?' for _ in types)
                    query += f" AND type IN ({placeholders})"
                    params.extend(types)

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor.execute(query, tuple(params))
                conditions.extend(cursor.fetchall())
        else:
            cursor.execute(
                "SELECT id, prompt FROM conditions WHERE experiment_id = ?",
                (experiment_id,),
            )
            conditions = cursor.fetchall()

        for config in model_configs:
            model_name = config['model_name']
            guidance_scale = config['guidance_scale']
            num_inference_steps = config['num_inference_steps']

            print(f"Loading model: {model_name}")
            pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            pipe.scheduler.set_timesteps(num_inference_steps)

            for condition_id, prompt in conditions:
                print(f"[{model_name}] Generating: {prompt} (guidance={guidance_scale}, steps={num_inference_steps})")
                image = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
                image_path = self._save_image(experiment_id, model_name, condition_id, image, guidance_scale, num_inference_steps)
                self._log_output(experiment_id, model_name, guidance_scale, num_inference_steps, condition_id, prompt, image_path)

    def _save_image(self, experiment_id: str, model_name: str, condition_id: int, image: Image.Image,
                    guidance_scale: float, num_inference_steps: int) -> str:
        model_tag = model_name.replace("/", "_")
        filename = f"{experiment_id}_{model_tag}_gs{guidance_scale}_steps{num_inference_steps}_cond{condition_id}.png"
        path = os.path.join(self.output_dir, filename)
        image.save(path)
        return path

    def _log_output(self, experiment_id: str, model_name: str, guidance_scale: float,
                    num_inference_steps: int, condition_id: int, prompt: str, image_path: str):
        cursor = self.output_conn.cursor()
        cursor.execute("""
            INSERT INTO dm_outputs (
                experiment_id, model_name, guidance_scale, num_inference_steps,
                condition_id, prompt, image_path, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id, model_name, guidance_scale, num_inference_steps,
            condition_id, prompt, image_path, datetime.now().isoformat()
        ))
        self.output_conn.commit()

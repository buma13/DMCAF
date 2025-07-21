import os
import sqlite3
from typing import List, Dict, Any
from datetime import datetime

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from prompt_to_prompt.sd_attention_google import AttentionStore, show_cross_attention, run_and_display


class CrossAttentionVisualizer:
    def __init__(self, conditioning_db_path: str, output_dir: str, token: str, save_images: bool = False, low_resource: bool = False):
        self.conditioning_conn = sqlite3.connect(conditioning_db_path)
        self.output_dir = output_dir
        self.token = token
        self.save_images = save_images
        self.low_resource = low_resource
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, experiment_id: str, model_configs: List[Dict[str, Any]], condition_sets: List[Dict[str, Any]] | None = None):
        cursor = self.conditioning_conn.cursor()
        conditions = []

        if condition_sets:
            for cs in condition_sets:
                cs_id = cs["condition_set_id"]
                limit = cs.get("limit_number_of_conditions")
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator().manual_seed(8888)

        for config in model_configs:
            model_name = config['model_name']
            guidance_scale = config['guidance_scale']
            num_inference_steps = config['num_inference_steps']

            print(f"Loading model: {model_name}")
            pipe = DiffusionPipeline.from_pretrained(
                model_name,
                use_auth_token=self.token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                variant="fp16"
            )
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(device)

            tokenizer = pipe.tokenizer

            for condition_id, prompt in conditions:
                print(f"[{model_name}] Visualizing attention: {prompt}")
                controller = AttentionStore()
                images, _ = run_and_display(
                    ldm_stable=pipe,
                    prompts=[prompt],
                    controller=controller,
                    generator=generator,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    low_resource=self.low_resource,
                    run_baseline=False,
                    save_image=self.save_images,
                    output_dir=self.output_dir,
                    prompt_slug=f"{experiment_id}_cond{condition_id}"
                )
                show_cross_attention(controller, res=16, from_where=["up", "down"], select=0)

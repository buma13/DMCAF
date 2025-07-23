import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, PNDMScheduler, AutoencoderKL
from . prompt_to_prompt.sd_attention_google import AttentionStore, show_cross_attention, run_and_display
from . prompt_to_prompt.ptp_utils import view_images

class DMRunner:
    def __init__(self, conditioning_db_path: str, output_db_path: str, output_dir: str,visualize_cfg: Dict[str, Any] = None):
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

    def run_experiment(self, experiment_id: str, model_configs: List[Dict[str, Any]], visualizer_configs: List[Dict[str, Any]],condition_sets: List[Dict[str, Any]] | None = None):
        cursor = self.conditioning_conn.cursor()
        conditions = []
        #setup visualizer
        visualizer_configs = visualizer_configs or {}
        visualize = visualizer_configs.get('visualize', False)
        cross_attention_cfg = visualizer_configs.get('cross_attention', {})
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
            vae_name = config.get('vae')
            scheduler_name = config.get('scheduler')
            low_resource = config.get('low_resource', False)
            guidance_scale = config.get('guidance_scale', 7.5)
            num_inference_steps = config.get('num_inference_steps', 50)
            generator = torch.Generator(device="cpu").manual_seed(config.get('seed', 42))

            print(f"Loading model: {model_name}")

            pipeline_kwargs = {}
            if vae_name:
                print(f"Loading VAE: {vae_name}")
                pipeline_kwargs['vae'] = AutoencoderKL.from_pretrained(vae_name)

            pipe = StableDiffusionPipeline.from_pretrained(model_name, **pipeline_kwargs)
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

            if scheduler_name:
                scheduler_class = {
                    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
                    "DDIMScheduler": DDIMScheduler,
                    "PNDMScheduler": PNDMScheduler,
                }.get(scheduler_name)

                if scheduler_class:
                    print(f"Using scheduler: {scheduler_name}")
                    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
                else:
                    print(f"Warning: Scheduler '{scheduler_name}' not found. Using pipeline default.")

            pipe.scheduler.set_timesteps(num_inference_steps)

            for condition_id, prompt in conditions:
                print(f"[{model_name}] Generating: {prompt} (guidance={guidance_scale}, steps={num_inference_steps})")
                controller = AttentionStore()
                # inference with prompt-to-prompt
                images, _ = run_and_display(
                    ldm_stable=pipe,
                    prompts=[prompt],
                    controller=controller,
                    latent=None,
                    run_baseline=False,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    low_resource=low_resource,
                    output_dir=self.output_dir
                )

                image = images[0]
                image_path = self._save_image(
                    experiment_id=experiment_id,
                    model_name=model_name,
                    condition_id=condition_id,
                    image=image,
                    visualize=False,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps)
                self._log_output(experiment_id, model_name, guidance_scale, num_inference_steps, condition_id, prompt, image_path,visualize=False)
                # If visualization is enabled, show cross-attention maps
                if visualize:
                    print(f"[{model_name}] Visualizing cross-attention for: {prompt}")
                    cross_attention_images=show_cross_attention(
                        tokenizer=pipe.tokenizer,
                        prompts=[prompt],
                        attention_store=controller,
                        res=cross_attention_cfg.get("res", 16),
                        from_where=tuple(cross_attention_cfg.get("from_where", ["up", "down"])),
                        select=cross_attention_cfg.get("select", 0),
                    )
                    # Convert cross-attention images to a format suitable for saving
                    attention_format_images=view_images((cross_attention_images), num_rows=1, offset_ratio=0.02)
                    image_path = self._save_image(
                        experiment_id=experiment_id,
                        model_name=model_name,
                        condition_id=condition_id,
                        image=attention_format_images,
                        visualize=True,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps)
                    self._log_output(experiment_id, model_name, guidance_scale, num_inference_steps, condition_id, prompt, image_path, visualize=True)




    def _save_image(self, experiment_id: str, model_name: str, condition_id: int, image: Image.Image,visualize: bool,
                    guidance_scale: float, num_inference_steps: int) -> str:
        model_tag = model_name.replace("/", "_")
        if visualize:
            filename = f"{experiment_id}_{model_tag}_gs{guidance_scale}_steps{num_inference_steps}_cond{condition_id}_visualization.png"
        else:
            filename = f"{experiment_id}_{model_tag}_gs{guidance_scale}_steps{num_inference_steps}_cond{condition_id}.png"
        path = os.path.join(self.output_dir, filename)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image.save(path)
        return path

    def _log_output(self, experiment_id: str, model_name: str, guidance_scale: float,
                    num_inference_steps: int, condition_id: int, prompt: str, image_path: str, visualize: bool):
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

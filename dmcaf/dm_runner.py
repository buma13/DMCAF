import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

from PIL import Image
import numpy as np
import torch
import copy
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, PNDMScheduler, AutoencoderKL, PixArtAlphaPipeline,StableDiffusion3Pipeline
import dmcaf.prompt_to_prompt.ptp_utils as PtpUtilsUnet
from . prompt_to_prompt.transformer_2d import Transformer2DModel
import dmcaf.prompt_to_prompt.ptp_utils_vit as PtpUtilsTransformer
import gc

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
        gc.collect()  # Clear any previous garbage collection to free up memory
        torch.cuda.empty_cache()
        #setup visualizer
        visualizer_configs = visualizer_configs or {}
        visualize = visualizer_configs.get('visualize', False)
        cross_attention_cfg = visualizer_configs.get('cross_attention', {})
        if condition_sets:
            for cs in condition_sets:
                cs_id = cs["condition_set_id"]
                limit = cs.get("limit_number_of_conditions")
                
                # Build query based on configuration
                query_parts = ["SELECT id, prompt, type FROM conditions WHERE experiment_id = ?"]
                params: List[Any] = [cs_id]
                
                # Handle type selection
                type_conditions = self._build_type_conditions(cs, params)
                if type_conditions:
                    query_parts.append(f"AND ({type_conditions})")
                
                # Handle filters
                filter_conditions = self._build_filter_conditions(cs, params)
                if filter_conditions:
                    query_parts.append(f"AND ({filter_conditions})")
                
                # Add limit
                if limit:
                    query_parts.append("LIMIT ?")
                    params.append(limit)
                
                query = " ".join(query_parts)
                
                # DEBUG: Print the actual query and parameters
                print(f"DEBUG: Executing query: {query}")
                print(f"DEBUG: Parameters: {params}")
                
                cursor.execute(query, tuple(params))
                batch_conditions = cursor.fetchall()
                
                print(f"Found {len(batch_conditions)} conditions for condition set '{cs_id}'")
                
                # DEBUG: Show what types were found
                if batch_conditions:
                    types_found = set(cond[2] for cond in batch_conditions)
                    print(f"DEBUG: Types found: {types_found}")
                else:
                    # DEBUG: Let's see what's actually available
                    cursor.execute("SELECT DISTINCT type FROM conditions WHERE experiment_id = ?", (cs_id,))
                    available_types = [row[0] for row in cursor.fetchall()]
                    print(f"DEBUG: Available types in {cs_id}: {available_types}")
                
                conditions.extend(batch_conditions)
        else:
            cursor.execute(
                "SELECT id, prompt, type FROM conditions WHERE experiment_id = ?",
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

            """pipe = StableDiffusionPipeline.from_pretrained(model_name, **pipeline_kwargs)
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

            pipe.scheduler.set_timesteps(num_inference_steps)"""
            
            pixart_transformer = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", subfolder="transformer",torch_dtype=torch.float16,)
            pipe = PixArtAlphaPipeline.from_pretrained(
                "PixArt-alpha/PixArt-XL-2-512x512", 
                transformer = pixart_transformer,
                torch_dtype=torch.float16)
            """hf_token = "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"
            pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", token=hf_token)
            pipe = pipe.to("cuda")"""

            for condition_id, prompt in conditions:
                print(f"[{model_name}] Generating: {prompt} (guidance={guidance_scale}, steps={num_inference_steps})")
                controller = PtpUtilsTransformer.AttentionStore()
                PtpUtilsTransformer.register_attention_control(pipe,controller)
                images = pipe(prompt=prompt,negative_prompt="",height=512,width=512).images[0]

                for i in range(28):
                    attn_map = PtpUtilsTransformer.get_self_attention_map(controller,256,i,False)
                    transform_attn_maps = copy.deepcopy(attn_map)
                    PtpUtilsTransformer.visualize_and_save_features_pca(
                            torch.cat([attn_map], dim=0),
                            torch.cat([transform_attn_maps], dim=0),
                            ['debug'],
                            i,
                            './self_attn_maps'
                        )
                images.save('generated_img_cond{condition_id}.png')
                # inference with prompt-to-prompt
                """images, _ = run_and_display(
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
                    self._log_output(experiment_id, model_name, guidance_scale, num_inference_steps, condition_id, prompt, image_path, visualize=True)"""




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

    def _build_type_conditions(self, cs_config: Dict, params: List[Any]) -> str:
        """Build SQL conditions for type selection."""
        conditions = []
        
        # Handle simple types list (backwards compatible)
        if "types" in cs_config:
            for requested_type in cs_config["types"]:
                # Fix: Use proper LIKE pattern with explicit parentheses
                conditions.append("(type = ? OR type LIKE ?)")
                params.extend([requested_type, f"{requested_type}_%"])
        
        # Handle type_variants (fine-grained control)
        if "type_variants" in cs_config:
            for base_type, config in cs_config["type_variants"].items():
                variants = config.get("variants", [])
                
                if variants == "all":
                    # Include all variants of this base type
                    conditions.append("(type = ? OR type LIKE ?)")
                    params.extend([base_type, f"{base_type}_%"])
                else:
                    # Include specific variants
                    variant_conditions = []
                    for variant in variants:
                        variant_conditions.append("type = ?")
                        params.append(f"{base_type}_{variant}")
                    
                    if variant_conditions:
                        conditions.append(f"({' OR '.join(variant_conditions)})")
    
        return " OR ".join(conditions) if conditions else ""

    def _build_filter_conditions(self, cs_config: Dict, params: List[Any]) -> str:
        """Build SQL conditions for advanced filtering."""
        conditions = []
        filters = cs_config.get("filters", {})
        
        # Object filtering
        if "objects" in filters:
            placeholders = ','.join('?' for _ in filters["objects"])
            conditions.append(f"object IN ({placeholders})")
            params.extend(filters["objects"])
        
        # Number range filtering
        if "number_range" in filters:
            placeholders = ','.join('?' for _ in filters["number_range"])
            conditions.append(f"number IN ({placeholders})")
            params.extend(filters["number_range"])
        
        # Background filtering
        if "backgrounds" in filters:
            bg_conditions = []
            for bg in filters["backgrounds"]:
                bg_conditions.append("prompt LIKE ?")
                params.append(f"%{bg}%")
            if bg_conditions:
                conditions.append(f"({' OR '.join(bg_conditions)})")
        
        # Custom SQL conditions
        if "custom_where" in filters:
            conditions.append(filters["custom_where"])
        
        return " AND ".join(conditions) if conditions else ""

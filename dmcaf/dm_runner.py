import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import copy
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, PNDMScheduler, AutoencoderKL, PixArtAlphaPipeline,StableDiffusion3Pipeline
import dmcaf.prompt_to_prompt.ptp_utils as PtpUtilsUnet
from . prompt_to_prompt.transformer_2d import Transformer2DModel
import dmcaf.prompt_to_prompt.ptp_utils_dit as PtpUtilsTransformer
import gc

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai import transforms as T
from dmcaf.models_fundus import get_model, get_controlnet, get_scheduler, get_inferers, SEED
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        dit = visualizer_configs.get('dit', False)
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
            pipeline_kwargs = {} # Additional kwargs for pipeline initialization
            if vae_name:
                print(f"Loading VAE: {vae_name}")
                pipeline_kwargs['vae'] = AutoencoderKL.from_pretrained(vae_name)

            # run standard dm_runner for unet-based models
            if not dit and not ("stable-diffusion-3" in model_name): 
                print(f"Loading model: {model_name}")

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
                # inference with prompt-to-prompt
                for condition_id, prompt, _ in conditions:
                    controller = PtpUtilsUnet.AttentionStore()
                    images, _ = PtpUtilsUnet.run_and_display(
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
                        cross_attention_images=PtpUtilsUnet.show_cross_attention(
                            tokenizer=pipe.tokenizer,
                            prompts=[prompt],
                            attention_store=controller,
                            res=cross_attention_cfg.get("res", 16),
                            from_where=tuple(cross_attention_cfg.get("from_where", ["up", "down"])),
                            select=cross_attention_cfg.get("select", 0),
                        )
                        # Convert cross-attention images to a format suitable for saving
                        attention_format_images=PtpUtilsUnet.view_images((cross_attention_images), num_rows=1, offset_ratio=0.02)
                        image_path = self._save_image(
                            experiment_id=experiment_id,
                            model_name=model_name,
                            condition_id=condition_id,
                            image=attention_format_images,
                            visualize=True,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps)
                        self._log_output(experiment_id, model_name, guidance_scale, num_inference_steps, condition_id, prompt, image_path, visualize=True)

            # run dit model with transformer backbone, we currently support sd-3 med, sd-3.5 med and PixArt-XL.
            else:
                if "stable-diffusion-3" in model_name:
                    pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16)
                    
                else: # assume PixArt-XL because we only support this additionally to 3 and 3.5. Also it takes more steps to load the model.
                    model_name="PixArt-alpha/PixArt-XL-2-512x512"
                    pixart_transformer = Transformer2DModel.from_pretrained(model_name, subfolder="transformer",torch_dtype=torch.float16,)
                    pipe = PixArtAlphaPipeline.from_pretrained(
                    model_name, 
                    transformer = pixart_transformer,
                    torch_dtype=torch.float16)
                print(f"Loading model: {model_name}")
                pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
                #pipe.enable_attention_slicing()
             

                for condition_id, prompt, _ in conditions:
                    print(f"[{model_name}] Generating: {prompt} (guidance={guidance_scale}, steps={num_inference_steps})")
                    #since visualization needs the controller, and only PixArt-XL supports it for now, we only create the controller for this model.
                    if model_name=="PixArt-alpha/PixArt-XL-2-512x512":
                        controller = PtpUtilsTransformer.AttentionStore()
                        PtpUtilsTransformer.register_attention_control(pipe,controller)
                    
                    images = pipe(prompt=prompt,height=512,width=512).images[0]
                    model_tag = model_name.replace("/", "_")
                    if visualize and model_name=="PixArt-alpha/PixArt-XL-2-512x512":
                        for i in range(28):
                            attn_map = PtpUtilsTransformer.get_self_attention_map(controller,256,i,False)
                            transform_attn_maps = copy.deepcopy(attn_map)
                            path = os.path.join(self.output_dir, f'self_attn_maps_{model_tag}_cond{condition_id}')
                            PtpUtilsTransformer.visualize_and_save_features_pca(
                                    torch.cat([attn_map], dim=0),
                                    torch.cat([transform_attn_maps], dim=0),
                                    ['debug'],
                                    i,
                                    path
                                )
                    filename = f"{experiment_id}_{model_tag}_cond{condition_id}.png"
                    path = os.path.join(self.output_dir, filename)
                    images.save(path)
                



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

class DMRunner_Fundus:
    def __init__(self, monai_home: str, output_dir: str):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.monai_home = monai_home
        self.output_dir = output_dir
        self.BATCH_SIZE = 32
        self.IMG_SIZE = 128
        self.NUM_WORKERS = 4
        self.SEED = 42

        # fundus transforms
        self.fundus_tf = T.Compose([
            T.LoadImaged(keys=["image", "mask"]),
            T.EnsureChannelFirstd(keys=["image", "mask"]),
            T.EnsureTyped(keys=["image", "mask"]),
            T.Lambdad(keys="image", func=lambda x: x.mean(dim=0, keepdim=True)),
            T.ResizeD(keys=["image", "mask"],
                      spatial_size=(self.IMG_SIZE, self.IMG_SIZE),
                      mode=("bilinear", "nearest")),
            T.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
            T.CastToTyped(keys=["image", "mask"], dtype=np.float32),
        ])



    def run_experiment(self, experiment_id: str, model_configs: List[Dict[str, Any]], condition_sets: List[Dict[str, Any]]):

        # --- create experiment directories ---
        exp_dir = Path(self.output_dir)
        masks_dir = exp_dir / "Masks"
        images_dir = exp_dir / "Images"
        masks_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # --- collect data from condition_sets ---
        val_data = {}
        for cs in condition_sets:
            cs_id = cs["condition_set_id"]
            limit = cs.get("limit_number_of_conditions")
            types = cs.get("types", [])

            if "segmentation_mask" not in types:
                raise ValueError(f"Error in {cs_id}: unknown type(s) {types}")

            if cs_id == "fundus":
                _, val_ds, _, val_loader = self._get_datasets_and_loaders(
                    batch_size=self.BATCH_SIZE,
                    num_workers=self.NUM_WORKERS,
                    verbose=True,
                    dataset="fundus",
                    data_dir=None,  # uses self.monai_home/datasets by default
                )
                val_data[cs_id] = (val_ds, val_loader)
            else:
                raise ValueError(f"Error in {cs_id}: unknown condition_set_id {cs_id}")

        # --- loop over models ---
        for config in model_configs:
            model_name = config['model_name'].lower()
            num_inference_steps = config.get('num_inference_steps', 1000)
            seed = config.get('seed', 42)

            if model_name != "unet":
                raise ValueError(f"Unsupported model_name: {model_name}")

            for cs_id, (val_ds, val_loader) in val_data.items():
                # --- build checkpoint paths ---
                model_ckpt_path = Path(self.monai_home) / "hub" / "DiffusionModelUNet" / f"best_model_{cs_id}.pth"
                cn_ckpt_path = Path(
                    self.monai_home) / "hub" / "ControlNet_DiffusionModelUNet" / f"best_model_{cs_id}.pth"

                # --- build & load models in fp16 + eval ---
                model = get_model(self.device, in_channels=1, out_channels=1,
                                  dtype=torch.float16, ckpt_path=str(model_ckpt_path),
                                  eval_mode=True, freeze=True)
                controlnet = get_controlnet(self.device, model, in_channels=1,
                                            dtype=torch.float16, ckpt_path=str(cn_ckpt_path),
                                            init_from_unet=True, eval_mode=True)

                # --- scheduler ---
                scheduler = get_scheduler(num_train_steps=num_inference_steps)
                # if available, set explicit timesteps once
                if hasattr(scheduler, "set_timesteps"):
                    scheduler.set_timesteps(num_inference_steps)

                # --- optional determinism for sampling ---
                g = torch.Generator(device=self.device).manual_seed(seed)

                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                    for batch_idx, val_batch in enumerate(val_loader):

                        val_masks = val_batch["mask"].to(self.device, dtype=torch.float16)

                        n = val_masks.shape[0]
                        sample = torch.randn((n, 1, self.IMG_SIZE, self.IMG_SIZE),
                                             device=self.device, dtype=torch.float16, generator=g)
                        t_tensor = torch.empty(n, device=self.device, dtype=torch.long)

                        for t in tqdm(scheduler.timesteps, desc=f"Sampling batch {batch_idx}", leave=False):
                            t_tensor.fill_(int(t))
                            down_res, mid_res = controlnet(
                                x=sample, timesteps=t_tensor, controlnet_cond=val_masks
                            )
                            noise_pred = model(
                                sample,
                                timesteps=t_tensor,
                                down_block_additional_residuals=down_res,
                                mid_block_additional_residual=mid_res,
                            )
                            sample, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=sample)

                        # --- save after sampling ---
                        for i in range(n):
                            mask_img = val_masks[i, 0].float().cpu().numpy()
                            result_img = sample[i, 0].float().cpu().numpy()
                            plt.imsave(masks_dir / f"mask_{batch_idx}_{i}.png",
                                       mask_img, cmap="gray", vmin=0, vmax=1)
                            plt.imsave(images_dir / f"generated_{batch_idx}_{i}.png",
                                       result_img, cmap="gray", vmin=0, vmax=1)

                        # free
                        del val_masks, sample, noise_pred, down_res, mid_res, t_tensor
                        torch.cuda.empty_cache()
                del model, controlnet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        pass

    def _get_datasets_and_loaders(self,
                                  batch_size: Optional[int] = None,
                                  num_workers: Optional[int] = None,
                                  verbose: bool = True,
                                  dataset: str = "fundus",
                                  data_dir: Optional[Path] = None):
        """
        dataset: supports "fundus"
        data_dir:
          - If None: uses Path(self.monai_home)/'datasets' when available,
                     else falls back to Path.cwd()/'datasets'
          - If provided: used as the datasets root
        """
        batch_size = batch_size or self.BATCH_SIZE
        num_workers = num_workers or self.NUM_WORKERS

        # Determine datasets root
        if data_dir is not None:
            root = Path(data_dir)
        else:
            root = Path(self.monai_home).joinpath("datasets") if self.monai_home else Path.cwd().joinpath("datasets")
        root.mkdir(parents=True, exist_ok=True)

        if dataset.lower() == "fundus":
            train_list, val_list = self._build_fundus_lists(data_dir=root, verbose=verbose, seed=self.SEED)
            train_ds = Dataset(train_list, transform=self.fundus_tf)
            val_ds = Dataset(val_list, transform=self.fundus_tf)
        else:
            raise ValueError(f"dataset must be 'fundus', got '{dataset}'")

        persistent = num_workers > 0
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, persistent_workers=persistent)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, drop_last=True, persistent_workers=persistent)

        return train_ds, val_ds, train_loader, val_loader

    @staticmethod
    def _build_fundus_lists( data_dir: Path,
                            verbose: bool = True,
                            test_size: float = 0.2,
                            seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

        fundus_dir = data_dir / "Fundus"
        csv_path = fundus_dir / "metadata - standardized.csv"
        df = pd.read_csv(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing CSV for fundus images: {csv_path}")

        # keep only rows with mask
        df = df[df["fundus_od_seg"].notna()].reset_index(drop=True)

        def resolve(rel_path: str) -> str:
            rel_path = rel_path.strip().lstrip("/")  # normalize
            return str(fundus_dir / rel_path)

        img_paths = df["fundus"].apply(resolve)
        mask_paths = df["fundus_od_seg"].apply(resolve)

        items = [
            {"image": i, "mask": m}
            for i, m in zip(img_paths, mask_paths)
            if Path(i).is_file() and Path(m).is_file()
        ]

        if verbose:
            print(f"Total with masks in CSV: {len(df)}")
            print(f"Valid image-mask pairs: {len(items)}")

        train_items, val_items = train_test_split(
            items, test_size=test_size, random_state=seed, shuffle=True
        )

        return train_items, val_items
import os
import torch
from typing import List
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from prompt_to_prompt.sd_attention_google import AttentionStore, show_cross_attention, run_and_display


class CrossAttentionVisualizer:
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        prompts: List[str],
        token: str,
        guidance_scale: float = 7.5,
        num_diffusion_steps: int = 50,
        seed: int = 8888,
        low_resource: bool = False
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.prompts = prompts
        self.token = token
        self.guidance_scale = guidance_scale
        self.num_diffusion_steps = num_diffusion_steps
        self.seed = seed
        self.low_resource = low_resource
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_pipeline()

    def _init_pipeline(self):
        pipe = DiffusionPipeline.from_pretrained(
            self.model_name,variant="fp16", use_auth_token=self.token
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.ldm = pipe.to(self.device)
        self.tokenizer = self.ldm.tokenizer
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)

    def visualize(self):
        controller = AttentionStore()
        image, _ = run_and_display(
            ldm_stable=self.ldm,
            prompts=self.prompts,
            controller=controller,
            latent=None,
            run_baseline=False,
            num_inference_steps=self.num_diffusion_steps,
            guidance_scale=self.guidance_scale,
            generator=self.generator,
            low_resource=self.low_resource
        )
        show_cross_attention(
            attention_store=controller,
            tokenizer=self.tokenizer,
            prompts=self.prompts,
            res=16,
            from_where=("up", "down"),
            select=0
        )
        return image


visualizer = CrossAttentionVisualizer(
model_name="stabilityai/stable-diffusion-2-base",
output_dir="./attention_output",
prompts=["5 cats in a futuristic city"],
token="hf_yourtokenhere")

visualizer.visualize()

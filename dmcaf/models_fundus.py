from monai.networks.nets import DiffusionModelUNet, ControlNet
from monai.inferers import DiffusionInferer, ControlNetDiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism
import torch

SEED = 42
set_determinism(SEED)

def get_model(device, in_channels=1, out_channels=1, *,
              dtype=torch.float32, ckpt_path: str | None = None,
              eval_mode: bool = True, freeze: bool = False):
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    ).to(device=device, dtype=dtype)

    if ckpt_path:
        sd = torch.load(ckpt_path, map_location=device)
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    if eval_mode:
        model.eval()
    return model

def get_controlnet(device, model, in_channels=1, *,
                   dtype=torch.float32, ckpt_path: str | None = None,
                   init_from_unet: bool = True, eval_mode: bool = True):
    controlnet = ControlNet(
        spatial_dims=2,
        in_channels=in_channels,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
        conditioning_embedding_num_channels=(16,),
    ).to(device=device, dtype=dtype)

    if init_from_unet:
        controlnet.load_state_dict(model.state_dict(), strict=False)

    if ckpt_path:
        sd = torch.load(ckpt_path, map_location=device)
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        controlnet.load_state_dict(sd, strict=False)

    if eval_mode:
        controlnet.eval()
    return controlnet


def get_scheduler(num_train_steps:int =1000):
    return DDPMScheduler(num_train_timesteps=num_train_steps)

def get_inferers(scheduler):
    return DiffusionInferer(scheduler), ControlNetDiffusionInferer(scheduler)
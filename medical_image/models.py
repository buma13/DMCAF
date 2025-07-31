import torch
from monai.networks.nets import DiffusionModelUNet, ControlNet
from monai.inferers import DiffusionInferer, ControlNetDiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism

SEED = 42
set_determinism(SEED)

def get_model(device, in_channels=1, out_channels=1):
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    ).to(device)
    return model

def get_controlnet(device, model, in_channels=1):
    controlnet = ControlNet(
        spatial_dims=2,
        in_channels=in_channels,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
        conditioning_embedding_num_channels=(16,),
    ).to(device)

    controlnet.load_state_dict(model.state_dict(), strict=False)

    for p in model.parameters():
        p.requires_grad = False

    return controlnet


def get_scheduler():
    return DDPMScheduler(num_train_timesteps=1000)

def get_inferers(scheduler):
    return DiffusionInferer(scheduler), ControlNetDiffusionInferer(scheduler)

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from diffusers import DDIMScheduler
from pathlib import Path
from IPython import embed
import os

try:
    from local_sd_pipeline import LocalStableDiffusionPipeline
    from optim_utils import *
except ModuleNotFoundError:
    import os; os.chdir("..")
    from local_sd_pipeline import LocalStableDiffusionPipeline
    from optim_utils import *


def gen_picture_magnitude(person):   
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    #model_id = "/dtu/blackhole/14/207860/memo/training/chexpert_checkpoints/v1-20000/"
    model_id = "/dtu/blackhole/14/207860/memo/training/chexpert_shorter_checkpoints/v1-40000"
    pipe = LocalStableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    num_inference_steps = 50
    guidance_scale = 7.5
    num_images_per_prompt = 4
    image_size = 512


    with Path(f'/dtu/blackhole/14/207860/memo/data/memodata/{person}.txt').open('rt') as file:
        prompt = file.read()

    outputs, track_stats = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        track_noise_norm=True,
    )
    text_noise_norm = track_stats["text_noise_norm"]
    outputs = outputs.images
    directory = f"/zhome/ca/2/153088/memorization/images_magnitudes/{person}"
    if not os.path.exists(directory):
    # If it doesn't exist, create it
        os.makedirs(directory)
    media.write_image(os.path.join(directory, f"{person}_0_v1-40000.png"), outputs[0])
    media.write_image(os.path.join(directory, f"{person}_1_v1-40000.png"), outputs[1])
    media.write_image(os.path.join(directory, f"{person}_2_v1-40000.png"), outputs[2])
    media.write_image(os.path.join(directory, f"{person}_3_v1-40000.png"), outputs[3])
    fontsize=26

    x = list(range(0, 1000, 20))
    x.remove(0)
    x.append(1000)
    x.reverse()

    viridis = plt.colormaps['Paired']
    cmap = viridis.colors

    fig, ax1 = plt.subplots(figsize=(8, 5))

    lns1 = ax1.plot(x, text_noise_norm[0], color=cmap[0], label='0', linewidth=4.0)
    lns2 = ax1.plot(x, text_noise_norm[1], color=cmap[1], label='1', linewidth=4.0)
    lns3 = ax1.plot(x, text_noise_norm[2], color=cmap[2], label='2', linewidth=4.0)
    lns4 = ax1.plot(x, text_noise_norm[3], color=cmap[3], label='3', linewidth=4.0)

    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]

    ax1.set_xlabel('Time-step', fontsize=fontsize)
    ax1.set_ylabel('Metric', fontsize=fontsize)
    ax1.invert_xaxis()
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    directory = f"/zhome/ca/2/153088/memorization/images_magnitudes/{person}/"
    os.makedirs(directory, exist_ok=True)
    fig.savefig(f"/zhome/ca/2/153088/memorization/images_magnitudes/{person}/{person}_magnitude_v1-40000.png")

gen_picture_magnitude("00005-1-1-f")
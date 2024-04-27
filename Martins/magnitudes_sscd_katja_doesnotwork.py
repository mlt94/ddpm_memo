import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from diffusers import DDIMScheduler
from pathlib import Path
from local_sd_pipeline import LocalStableDiffusionPipeline
from optim_utils import *
import PIL
import os 
import csv

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sim_model = torch.jit.load("/zhome/ca/2/153088/memorization/sscd_disc_large.torchscript.pt").to(device)

gt_mem_pics = []
base_path = Path('/dtu/blackhole/14/207860/memo/data/memodata/')
for num in range(1,1000):
    mem_image_path = base_path/f'{num:05d}-1-1-f.jpg'
    if not mem_image_path.is_file():
        continue
    mem_jpg = PIL.Image.open(mem_image_path)
    gt_mem_pics.append(mem_jpg)    


model_id = "/dtu/blackhole/14/207860/memo/training/chexpert_shorter_checkpoints/v1-40000"
select_prompts = np.random.randint(64541, 64708, size=100)
memorized_prompt_dict = {}

SSCD_max_total = []
text_noise_mean_total =  []
for i in select_prompts:
    non_mem_jpg = PIL.Image.open(f'/dtu/blackhole/14/207860/memo/data/memodata-shorter-prompts/valid/{i}-1-1-f.jpg')
    with Path(f'/dtu/blackhole/14/207860/memo/data/memodata-shorter-prompts/valid/{i}-1-1-f.txt').open('rt') as file:
        prompt_non_mem = file.read()
    
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
    num_images_per_prompt = 1
    image_size = 512

    outputs, track_stats = pipe(
        prompt_non_mem,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        track_noise_norm=True,
    )
    text_noise_norm = track_stats["text_noise_norm"]
    output_non_mem = outputs.images #1 generet billede pba. et non-mem prompt
    SSCD_sim = measure_SSCD_similarity(output_non_mem, gt_mem_pics, sim_model, device).cpu().detach().numpy()
    SSCD_index = np.argmax(SSCD_sim) 
    SSCD_max = SSCD_sim[SSCD_index]
    text_noise_mean = np.mean(text_noise_norm) #gennemsnit af de 50 text-conditioned noise
    
    SSCD_max_total.append(SSCD_max)
    text_noise_mean_total.append(text_noise_mean)

csv_file = "/zhome/ca/2/153088/memorization/diffusion_memorization/SSCD_and_magnitudes_memorized_katja.csv"
keys_to_save = ['name', 'SSCD', 'noise',]
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=keys_to_save)
    writer.writeheader()
    for prompt_id, SSCD, text_mean in zip(select_prompts, SSCD_max_total, text_noise_mean_total):
        writer.writerow({"name": prompt_id, "SSCD": SSCD, "noise": text_mean})




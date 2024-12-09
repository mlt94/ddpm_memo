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

'''This code generates two csv files - for training and validation images - comprising
SSCD scores and text_noise_norms (between a ground truth image and the 4 generated ones)
Note, that I am only running it a 100 times for the training, and a 100 times for the validation
It can then be plotted in the SSCD.ipynb. Remember that you need the "sscd_disc_large.torchscript.pt 
from Yuxins github. I guess we need to investigate if we really can use this SSCD model from their github
or if we need to finetune or something'''

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sim_model = torch.jit.load("/zhome/ca/2/153088/memorization/sscd_disc_large.torchscript.pt").to(device)

#for memorized prompts
folder_path = "/dtu/blackhole/14/207860/memo/data/memodata-shorter-prompts/train" 
num_files_to_select = 100 
random_numbers = random.sample(range(64000), num_files_to_select)
formatted_numbers = ["{:05d}".format(num) for num in random_numbers]
memorized_prompt_dict = {}

for i in formatted_numbers:
    gt = PIL.Image.open(f'/dtu/blackhole/14/207860/memo/data/memodata/{i}-1-1-f.jpg')
    with Path(f'/dtu/blackhole/14/207860/memo/data/memodata/{i}-1-1-f.txt').open('rt') as file:
        prompt_gt = file.read()

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

    outputs, track_stats = pipe(
        prompt_gt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        track_noise_norm=True,
    )
    text_noise_norm = track_stats["text_noise_norm"]
    outputs = outputs.images
    SSCD_sim = measure_SSCD_similarity([gt], outputs, sim_model, device).cpu().detach().numpy()
    memorized_prompt_dict[i] = {
        "SSCD": SSCD_sim,
        "noise": text_noise_norm
    }
    
    keys_to_save = ['name', 'SSCD', 'noise',]

csv_file = "/zhome/ca/2/153088/memorization/diffusion_memorization/SSCD_and_magnitudes_memorized.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=keys_to_save)
    writer.writeheader()
    for prompt_id, data in memorized_prompt_dict.items():
        writer.writerow({"name": prompt_id, "SSCD": data["SSCD"], "noise": data["noise"]})



# for non-memorized
folder_path = "/dtu/blackhole/14/207860/memo/data/memodata-shorter-prompts/valid" 
memorized_prompt_dict = {}

for i in range(64541,64642): #100 instances in the validation range
    gt = PIL.Image.open(f'/dtu/blackhole/14/207860/memo/data/memodata-shorter-prompts/valid/{i}-1-1-f.jpg')
    with Path(f'/dtu/blackhole/14/207860/memo/data/memodata-shorter-prompts/valid/{i}-1-1-f.txt').open('rt') as file:
        prompt_gt = file.read()

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

    outputs, track_stats = pipe(
        prompt_gt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        track_noise_norm=True,
    )
    text_noise_norm = track_stats["text_noise_norm"]
    outputs = outputs.images
    SSCD_sim = measure_SSCD_similarity([gt], outputs, sim_model, device).cpu().detach().numpy()
    memorized_prompt_dict[i] = {
        "SSCD": SSCD_sim,
        "noise": text_noise_norm
    }
    
    keys_to_save = ['name', 'SSCD', 'noise',]

csv_file = "/zhome/ca/2/153088/memorization/diffusion_memorization/SSCD_and_magnitudes_nonmemorized.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=keys_to_save)
    writer.writeheader()
    for prompt_id, data in memorized_prompt_dict.items():
        writer.writerow({"name": prompt_id, "SSCD": data["SSCD"], "noise": data["noise"]})
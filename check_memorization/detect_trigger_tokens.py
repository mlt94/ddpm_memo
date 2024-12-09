import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from diffusers import DDIMScheduler
from pathlib import Path

try:
    from local_sd_pipeline import LocalStableDiffusionPipeline
    from optim_utils import *
except ModuleNotFoundError:
    import os; os.chdir("..")
    from local_sd_pipeline import LocalStableDiffusionPipeline
    from optim_utils import *

# load model
def get_trigger_tokens(person):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    seed = 1

    with Path(f'/dtu/blackhole/14/207860/memo/training/martin/fake_validation_prompts/patient_sentences/{person}.txt').open('rt') as file:
        prompt = file.read()
    set_random_seed(seed)

    outputs, track_stats = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        track_noise_norm=True,
    )
    outputs = outputs.images

    set_random_seed(seed)
    token_grads = pipe.get_text_cond_grad(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=4,
        target_steps=list(range(50)),
    )
    torch.cuda.empty_cache()

    prompt_tokens = pipe.tokenizer.encode(prompt)
    prompt_tokens = prompt_tokens[1:-1]
    prompt_tokens = prompt_tokens[:75]
    token_grads = token_grads[1:(1+len(prompt_tokens))]
    token_grads = token_grads.cpu().tolist()

    all_tokes = []

    for curr_token in prompt_tokens:
        all_tokes.append(pipe.tokenizer.decode(curr_token))

    print(f"prompt:{prompt}")
    plt.figure(figsize=(20, 16))
    plt.bar(all_tokes, token_grads)
    plt.xticks(fontsize=26, rotation=45)
    plt.yticks(fontsize=18)
    plt.savefig("/zhome/ca/2/153088/memorization/diffusion_memorization/detect_trigger_tokens.png")

get_trigger_tokens("patient_35")
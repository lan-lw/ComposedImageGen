import os
import sys
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets.dataset_utils import ComposedEmbedsDataset


"""
Parameters
"""
import argparse

parser = argparse.ArgumentParser(description="Generate CIRR images with SDXL using embeddings.")
parser.add_argument("--text_embeddings_dir", type=str, default="/path/to/embedded_text_lincir_cirr_test/", help="Directory containing {pairid}.pt embedding files")
parser.add_argument("--dataset_dir", type=str, default="/path/to/cirr/dataset", help="CIRR dataset root directory")
parser.add_argument("--model_path", type=str, default="/path/to/checkpoint/", help="Path to SDXL UNet checkpoint directory (expects a 'unet' subfolder)")
parser.add_argument("--save_path", type=str, default="/path/to/generated_images/", help="Directory to save generated images")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader and batched generation")
parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
parser.add_argument("--height", type=int, default=512, help="Output image height")
parser.add_argument("--width", type=int, default=512, help="Output image width")
parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
parser.add_argument("--seed", type=int, default=1600, help="Base random seed per sample")
parser.add_argument("--brightness_thresh", type=float, default=10.0, help="Brightness threshold to accept an image")
parser.add_argument("--max_retries", type=int, default=5, help="Max retries per image if brightness is too low")
parser.add_argument("--cache_dir", type=str, default="./", help="HuggingFace cache directory")
parser.add_argument("--vae_repo", type=str, default="madebyollin/sdxl-vae-fp16-fix", help="VAE repo id for SDXL")
parser.add_argument("--sdxl_repo", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="SDXL base repo id")
args = parser.parse_args(sys.argv[1:])

# Resolve and create directories
os.makedirs(args.save_path, exist_ok=True)
os.makedirs(args.text_embeddings_dir, exist_ok=True)

# Data
dataset = ComposedEmbedsDataset(args.dataset_dir, args.text_embeddings_dir, split='test1')
loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

dtype = torch.float16
device = "cuda"

unet = UNet2DConditionModel.from_pretrained(
    f"{args.model_path}/unet", use_safetensors=True, torch_dtype=dtype
).to(device)

vae = AutoencoderKL.from_pretrained(
    args.vae_repo,
    torch_dtype=dtype,
    cache_dir=args.cache_dir
).to(device)

pipe = StableDiffusionXLPipeline.from_pretrained(
    args.sdxl_repo,
    unet=unet,
    vae=vae,
    torch_dtype=dtype,
    cache_dir=args.cache_dir
)

pipe.to(device)

print("Number of files to process", len(dataset))
for batch in tqdm(loader, desc="Generating images", unit="batch"):
    pairids = batch['pairid']  # list of str
    prompt_embeds_batch = batch['prompt_embeds']  # Tensor [B, S, D]
    pooled2_batch = batch['pooled2']  # Tensor [B, D]

    # Filter out items already generated
    indices = [i for i, pid in enumerate(pairids) if not os.path.exists(os.path.join(args.save_path, f"{pid}.png"))]
    if len(indices) == 0:
        continue

    # Prepare sub-batch tensors and seeds
    pe_sub = prompt_embeds_batch[indices].to(device, dtype=dtype)
    pooled_sub = pooled2_batch[indices].to(device, dtype=dtype)
    pids_sub = [pairids[i] for i in indices]

    # Initial generators per item
    seeds = [args.seed for _ in range(len(indices))]
    generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

    # Generate batched
    images = pipe(
        prompt_embeds=pe_sub,
        pooled_prompt_embeds=pooled_sub,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generators
    ).images

    # Check brightness and collect failures
    valid_mask = []
    for img in images:
        brightness = np.asarray(img).mean()
        valid_mask.append(brightness > args.brightness_thresh)

    # Save valid images
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            out_path = os.path.join(args.save_path, f"{pids_sub[i]}.png")
            images[i].save(out_path)

    # Retry failing ones up to a few times (e.g., 5 attempts)
    max_retries = args.max_retries
    attempt = 1
    while attempt <= max_retries and (not all(valid_mask)):
        fail_indices = [i for i, ok in enumerate(valid_mask) if not ok]
        if len(fail_indices) == 0:
            break

        # Prepare subset for retry
        pe_retry = pe_sub[fail_indices]
        pooled_retry = pooled_sub[fail_indices]
        pids_retry = [pids_sub[i] for i in fail_indices]

        # Bump seeds for retries
        for j in range(len(fail_indices)):
            seeds[fail_indices[j]] += 1
        generators_retry = [torch.Generator(device=device).manual_seed(seeds[i]) for i in fail_indices]

        images_retry = pipe(
            prompt_embeds=pe_retry,
            pooled_prompt_embeds=pooled_retry,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            generator=generators_retry
        ).images

        # Evaluate and save passing images
        retry_valid = []
        for img in images_retry:
            brightness = np.asarray(img).mean()
            retry_valid.append(brightness > args.brightness_thresh)

        k = 0
        for idx_ok, ok in zip(fail_indices, retry_valid):
            if ok:
                out_path = os.path.join(args.save_path, f"{pids_sub[idx_ok]}.png")
                images_retry[k].save(out_path)
                valid_mask[idx_ok] = True
            k += 1

        attempt += 1


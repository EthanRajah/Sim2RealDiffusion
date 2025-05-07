import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector
import numpy as np
from PIL import Image
import argparse
from time import time
import os

def resize_for_condition_image(input_image: Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def controlnet_infer(input_path, model_id, output_dir, prompt):
    """
    Function to run inference on sim2real dreambooth model with tile and softedge control net.
    Supports passing a single image or a directory of images.
    """

    # Load tile and softedge control net models
    tile_control = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
    softedge_control = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    controlnet = [tile_control, softedge_control]
    
    # Apply control net to sim2real model to generate pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16).to('cuda')
    generator = torch.Generator(device='cpu').manual_seed(0)
    
    # Reduce inference times by using a multistep scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    # Check if input_path is a file or directory
    if os.path.isdir(input_path):
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    else:
        image_paths = [input_path]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    start = time()
    for img_path in image_paths:
        img = load_image(img_path)
        tile_condition_img = resize_for_condition_image(img, 64)

        # Prepare edge mask for softedge control net
        processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        edge_condition_image = processor(img, safe=True)

        # Run inference using pipeline
        images = [tile_condition_img, edge_condition_image]
        output_latent = pipe(prompt, images, num_inference_steps=20, generator=generator, controlnet_conditioning_scale=[0.35, 1.35], guidance_scale=5.5, output_type="latent").images[0]
        print(output_latent)
        
    print(f"Total time taken: {time() - start:.2f} seconds")
    print(f"Average time per image: {(time() - start) / len(image_paths):.2f} seconds")
    print(f"Inference completed. Results saved to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str, default='./')
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--resolution', type=int, default=512)

    args = parser.parse_args()
    img_path = args.img_path
    model_path = args.model_path
    output_path = args.output_path
    prompt = args.prompt
    resolution = args.resolution

    controlnet_infer(img_path, model_path, output_path, prompt)
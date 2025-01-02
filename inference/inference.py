import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from controlnet_aux import PidiNetDetector
import numpy as np
from PIL import Image
import argparse

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

def controlnet_infer(img_path, model_id, output_path, prompt):
    '''Function to run inference on sim2real dreambooth model with tile and softedge control net'''
    # Load the image
    img = load_image(img_path)
    tile_condition_img = resize_for_condition_image(img, 512)

    # Prepare edge mask for softedge control net
    processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    edge_condition_image = processor(img, safe=True)
    edge_condition_image.save("./images/control.png")

    # Load tile and softedge control net models
    tile_control = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
    softedge_control = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    controlnet = [tile_control, softedge_control]

    # Apply control net to sim2real model to generate pipeline
    pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, controlnet=controlnet, torch_dtype=torch.float16, device='cuda')
    # Reduce inference times by using a multistep scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    # Run inference using pipeline
    generator = torch.Generator(device='cuda').manual_seed(0)
    images = [tile_condition_img, edge_condition_image]
    image = pipe(prompt, images, num_inference_steps=30, generator=generator).images[0]
    image.save(output_path)

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--prompt', type=str)

    args = parser.parse_args()
    img_path = args.img_path
    model_path = args.model_path
    output_path = args.output_path
    prompt = args.prompt
import torch
from diffusers import ControlNetModel, StableDiffusionPipeline, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from pipeline_rf_ctrl import RectifiedFlowCtrlPipeline
from controlnet_aux import PidiNetDetector
from PIL import Image
import argparse
from time import time
import os
from peft import PeftModel, LoraConfig
import safetensors.torch

MODEL_NAME = "runwayml/stable-diffusion-v1-5"

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

def merge_dW_to_unet(pipe, dW_dict, alpha=1.0):
    # Add delta weights to the unet model
    _tmp_sd = pipe.unet.state_dict()
    device = next(pipe.unet.parameters()).device
    for key in dW_dict.keys():
        if key in _tmp_sd:
            _tmp_sd[key] += dW_dict[key].to(device) * alpha
    pipe.unet.load_state_dict(_tmp_sd, strict=False)
    print(f"Unet model updated with alpha={alpha}")
    return pipe

def merge_dW_to_text_encoder(pipe, dW_dict, alpha=1.0):
    # Add delta weights to the text encoder model
    _tmp_sd = pipe.text_encoder.state_dict()
    device = next(pipe.text_encoder.parameters()).device
    for key in dW_dict.keys():
        if key in _tmp_sd:
            _tmp_sd[key] += dW_dict[key].to(device) * alpha
    pipe.text_encoder.load_state_dict(_tmp_sd, strict=False)
    print(f"Text encoder model updated with alpha={alpha}")
    return pipe

def load_hf_hub_lora(pipe_rf, lora_path='Lykon/dreamshaper-7', save_dW = False, base_sd='runwayml/stable-diffusion-v1-5', alpha=1.0):    
    # get weights of base sd models
    _pipe = StableDiffusionPipeline.from_pretrained(
        base_sd, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    sd_state_dict = _pipe.unet.state_dict()
    sd_text_encoder_state_dict = _pipe.text_encoder.state_dict()
    
    # get weights of the customized sd models
    _pipe = StableDiffusionPipeline.from_pretrained(
        lora_path, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    lora_unet_checkpoint = _pipe.unet.state_dict()
    lora_text_encoder_checkpoint = _pipe.text_encoder.state_dict()
    
    # get the dW
    dW_dict = {}
    for key in lora_unet_checkpoint.keys():
        dW_dict[key] = lora_unet_checkpoint[key] - sd_state_dict[key]

    dW_dict_te = {}
    for key in lora_text_encoder_checkpoint.keys():
        dW_dict_te[key] = lora_text_encoder_checkpoint[key] - sd_text_encoder_state_dict[key]
    
    # return and save dW dict
    if save_dW:
        save_name = lora_path.split('/')[-1] + '_dW.pt'
        torch.save(dW_dict, save_name)
        
    pipe_rf = merge_dW_to_unet(pipe_rf, dW_dict=dW_dict, alpha=alpha)
    pipe_rf = merge_dW_to_text_encoder(pipe_rf, dW_dict=dW_dict_te, alpha=alpha)
    pipe_rf.vae = _pipe.vae
    # pipe_rf.text_encoder = _pipe.text_encoder
    
    return dW_dict

def load_custom_adapter(adapter_path, alpha=1.0):
    """
    Load a custom lora adapter into SDv1.5 base model prior to merging delta weights with the rectified flow model.
    """
    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter path {adapter_path} does not exist.")
    
    # Load base model
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        safety_checker = None,
    )
    # Load the adapter
    pipe.load_lora_weights(os.path.join(adapter_path, 'unet'))
    pipe.load_lora_weights(os.path.join(adapter_path, 'text_encoder'))
    pipe.fuse_lora(alpha=alpha)

    # Save model for use in load_hf_hub_lora
    save_name = os.path.join(adapter_path, 'base_sd_with_adapter')
    pipe.save_pretrained(save_name)

# Function to apply LoRA weights manually
def apply_lora_to_state_dict(base_state_dict, lora_weights, alpha=1.0, device="cuda"):
    """Apply LoRA weights to base model state dict without using PEFT."""
    updated_state_dict = base_state_dict.copy()
    
    # Group LoRA A and B weights
    lora_pairs = {}
    print("Size of LoRA weights:", len(lora_weights))
    for key, weight in lora_weights.items():
        # Remove the base_model.model. prefix and extract the base key
        clean_key = key.replace("_orig_mod.base_model.model.", "")
        
        if ".lora_A." in clean_key:
            base_key = clean_key.replace(".lora_A.default", "").replace(".lora_A.weight", ".weight")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['A'] = weight
        elif ".lora_B." in clean_key:
            base_key = clean_key.replace(".lora_B.default", "").replace(".lora_B.weight", ".weight")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = {}
            lora_pairs[base_key]['B'] = weight
    
    # Apply LoRA: W = W + alpha * (B @ A)
    applied_count = 0
    for base_key, lora_weights_pair in lora_pairs.items():
        if 'A' in lora_weights_pair and 'B' in lora_weights_pair:
            if base_key in updated_state_dict:
                lora_A = lora_weights_pair['A'].to(device)
                lora_B = lora_weights_pair['B'].to(device)
                
                # Compute delta weight: delta = alpha * (B @ A)
                if lora_A.dim() == 2 and lora_B.dim() == 2:
                    delta = alpha * (lora_B @ lora_A)
                    updated_state_dict[base_key] = updated_state_dict[base_key].to(device) + delta
                    applied_count += 1
                    # print(f"Applied LoRA to: {base_key}")
                else:
                    print(f"Skipped {base_key}: dimensions A={lora_A.shape}, B={lora_B.shape}")
            else:
                print(f"Key not found in base model: {base_key}")
    
    # Should print half of the number of LoRA weights detected from the LoRA state dict
    print(f"Applied {applied_count} LoRA weight pairs")
    return updated_state_dict

def get_lora_sd_pipeline(ckpt_dir, device="cuda", adapter_name="default", dtype=torch.float16):
    """Use this function for LoRA trained on Instaflow model, to combine with InstaFlow model."""
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir):
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path
    
    tile_control = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
    softedge_control = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    controlnet = [tile_control, softedge_control]

    pipe = RectifiedFlowCtrlPipeline.from_pretrained(base_model_name_or_path, torch_dtype=torch.float16, controlnet=controlnet, safety_checker=None)
    # PEFT will combine the weights of the base model with the fine-tuned LoRA weights using LoraConfig
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)
    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name)

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe

def get_lora_sd_pipeline_peft(ckpt_dir, device="cuda", dtype=torch.float16):
    """Use this function for LoRA trained on base SDv1.5 model, to combine with InstaFlow model."""
    """Load LoRA adapter for Stable Diffusion pipeline using PEFT. This combines the weights of the base model with the fine-tuned LoRA weights using LoraConfig"""
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir):
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model = config.base_model_name_or_path

    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, safety_checker=None).to(device)

    # Store the state dicts of the base model
    sd_state_dict = pipe.unet.state_dict()
    sd_text_encoder_state_dict = pipe.text_encoder.state_dict()

    # Load adapter weights into the unet and text encoder using PEFT
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir)
    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_sub_dir)

    # Merge LoRA weights into the unet and text encoder: this function applies lora deltas and removes the LoRA layers
    # W = W + alpha * (A @ B) where A and B are the LoRA trained decomposition matrices
    pipe.unet.merge_and_unload()
    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder.merge_and_unload()

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()
    # Store the LoRA fine-tuned weights
    lora_unet_checkpoint_init = pipe.unet.state_dict()
    lora_text_encoder_checkpoint_init = pipe.text_encoder.state_dict()

    # LoRA weight keys have an additional prefix of "base_model.model." Need to remove this prefix to compute delta weights wrt the base model
    # Attention layers are made to include "base_layer", which needs to be removed as well 
    lora_unet_checkpoint = {}
    lora_text_encoder_checkpoint = {}
    for key in lora_unet_checkpoint_init.keys():
        new_key = key.replace("base_model.model.", "")
        new_key = new_key.replace("base_layer.", "")
        lora_unet_checkpoint[new_key] = lora_unet_checkpoint_init[key]
    for key in lora_text_encoder_checkpoint_init.keys():
        new_key = key.replace("base_model.model.", "")
        new_key = new_key.replace("base_layer.", "")
        lora_text_encoder_checkpoint[new_key] = lora_text_encoder_checkpoint_init[key]

    # Compute delta weights between the base model and the LoRA fine-tuned model to update the InstaFlow model with
    dW_dict = {}
    for key in lora_unet_checkpoint.keys():
        if key in sd_state_dict:
            dW_dict[key] = lora_unet_checkpoint[key] - sd_state_dict[key]
        else:
            print(f"Warning: Key {key} not found in base model state dict")
    
    dW_dict_te = {}
    for key in lora_text_encoder_checkpoint.keys():
        if key in sd_text_encoder_state_dict:
            dW_dict_te[key] = lora_text_encoder_checkpoint[key] - sd_text_encoder_state_dict[key]
        else:
            print(f"Warning: Key {key} not found in base text encoder state dict")
    
    # Debug: Check magnitude of delta weights
    if dW_dict:
        unet_delta_norms = [torch.norm(delta).item() for delta in dW_dict.values()]
        print(f"UNet delta weight norms - min: {min(unet_delta_norms):.6f}, max: {max(unet_delta_norms):.6f}, mean: {sum(unet_delta_norms)/len(unet_delta_norms):.6f}")
    
    if dW_dict_te:
        te_delta_norms = [torch.norm(delta).item() for delta in dW_dict_te.values()]
        print(f"Text encoder delta weight norms - min: {min(te_delta_norms):.6f}, max: {max(te_delta_norms):.6f}, mean: {sum(te_delta_norms)/len(te_delta_norms):.6f}")

    # Initialize InstaFlow model
    # This process will load the LoRA fine tuned model and compute delta weights (dW) from the base model to use for updating the InstaFlow model for inference
    tile_control = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
    softedge_control = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    controlnet = [tile_control, softedge_control]
    pipe = RectifiedFlowCtrlPipeline.from_pretrained("XCLiu/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float16, controlnet=controlnet, safety_checker=None).to(device)
    
    # Merge delta weights into the InstaFlow model
    pipe = merge_dW_to_unet(pipe, dW_dict=dW_dict, alpha=1.0)
    pipe = merge_dW_to_text_encoder(pipe, dW_dict=dW_dict_te, alpha=1.0)
    pipe.to(device)
    return pipe

def test_lora(ckpt_dir, device="cuda", dtype=torch.float16):
    """Function to load LoRA with base model and perform inference. Useful for making sure LoRA hyperparams are resulting in effective learning"""
    print("Testing LoRA weights with base model...")
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    
    # Set default base model
    base_model = "runwayml/stable-diffusion-v1-5"
    if os.path.exists(text_encoder_sub_dir):
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        if config.base_model_name_or_path:
            base_model = config.base_model_name_or_path

    tile_control = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
    softedge_control = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    controlnet = [tile_control, softedge_control]

    print(f"Loading LoRA weights on top of: {base_model}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model, torch_dtype=dtype, controlnet=controlnet, safety_checker=None).to(device)
    sd_state_dict = pipe.unet.state_dict()
    sd_text_encoder_state_dict = pipe.text_encoder.state_dict()

    # Load UNet LoRA weights
    unet_lora_path = os.path.join(unet_sub_dir, "adapter_model.safetensors")
    if os.path.exists(unet_lora_path):
        unet_lora_weights = safetensors.torch.load_file(unet_lora_path)
        print(f"Loaded {len(unet_lora_weights)} UNet LoRA weights")
    else:
        print("No UNet LoRA weights found")
        unet_lora_weights = {}
    
    # Load Text Encoder LoRA weights
    te_lora_path = os.path.join(text_encoder_sub_dir, "adapter_model.safetensors")
    te_lora_weights = {}
    if os.path.exists(text_encoder_sub_dir) and os.path.exists(te_lora_path):
        te_lora_weights = safetensors.torch.load_file(te_lora_path)
        print(f"Loaded {len(te_lora_weights)} Text Encoder LoRA weights")
    
    # Apply LoRA weights to create fine-tuned state dicts
    lora_unet_state_dict = apply_lora_to_state_dict(sd_state_dict, unet_lora_weights, alpha=1.0)
    lora_text_encoder_state_dict = apply_lora_to_state_dict(sd_text_encoder_state_dict, te_lora_weights, alpha=1.0)

    pipe.unet.load_state_dict(lora_unet_state_dict, strict=False)
    pipe.text_encoder.load_state_dict(lora_text_encoder_state_dict, strict=False)

    return pipe

def get_lora_sd_pipeline_final(ckpt_dir, device="cuda", dtype=torch.float16, alpha=1.0):
    """Use this function for LoRA trained on base SDv1.5 model, to combine with InstaFlow model."""
    print("Applying LoRA weights to InstaFlow for inference...")
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    
    # Set default base model
    base_model = "runwayml/stable-diffusion-v1-5"
    if os.path.exists(text_encoder_sub_dir):
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        if config.base_model_name_or_path:
            base_model = config.base_model_name_or_path

    print(f"Loading LoRA weights on top of: {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, safety_checker=None).to(device)

    # Store the state dicts of the base model
    sd_state_dict = pipe.unet.state_dict()
    sd_text_encoder_state_dict = pipe.text_encoder.state_dict()
    
    # Load UNet LoRA weights
    unet_lora_path = os.path.join(unet_sub_dir, "adapter_model.safetensors")
    if os.path.exists(unet_lora_path):
        unet_lora_weights = safetensors.torch.load_file(unet_lora_path)
        print(f"Loaded {len(unet_lora_weights)} UNet LoRA weights")
    else:
        print("No UNet LoRA weights found")
        unet_lora_weights = {}
    
    # Load Text Encoder LoRA weights
    te_lora_path = os.path.join(text_encoder_sub_dir, "adapter_model.safetensors")
    te_lora_weights = {}
    if os.path.exists(text_encoder_sub_dir) and os.path.exists(te_lora_path):
        te_lora_weights = safetensors.torch.load_file(te_lora_path)
        print(f"Loaded {len(te_lora_weights)} Text Encoder LoRA weights")
    
    # Apply LoRA weights to create fine-tuned state dicts
    lora_unet_state_dict = apply_lora_to_state_dict(sd_state_dict, unet_lora_weights, alpha=alpha)
    lora_text_encoder_state_dict = apply_lora_to_state_dict(sd_text_encoder_state_dict, te_lora_weights, alpha=alpha)
    
    # Compute delta weights between the base model and the LoRA fine-tuned model
    dW_dict = {}
    for key in lora_unet_state_dict.keys():
        if key in sd_state_dict:
            dW_dict[key] = lora_unet_state_dict[key] - sd_state_dict[key]
    
    dW_dict_te = {}
    for key in lora_text_encoder_state_dict.keys():
        if key in sd_text_encoder_state_dict:
            dW_dict_te[key] = lora_text_encoder_state_dict[key] - sd_text_encoder_state_dict[key]
    
    # Debug: Check magnitude of delta weights
    if dW_dict:
        unet_delta_norms = [torch.norm(delta).item() for delta in dW_dict.values()]
        print(f"UNet delta weight norms - min: {min(unet_delta_norms):.6f}, max: {max(unet_delta_norms):.6f}, mean: {sum(unet_delta_norms)/len(unet_delta_norms):.6f}")
    
    if dW_dict_te:
        te_delta_norms = [torch.norm(delta).item() for delta in dW_dict_te.values()]
        print(f"Text encoder delta weight norms - min: {min(te_delta_norms):.6f}, max: {max(te_delta_norms):.6f}, mean: {sum(te_delta_norms)/len(te_delta_norms):.6f}")

    # Initialize InstaFlow model
    tile_control = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
    softedge_control = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
    controlnet = [tile_control, softedge_control]
    pipe = RectifiedFlowCtrlPipeline.from_pretrained("XCLIU/instaflow_0_9B_from_sd_1_5", torch_dtype=torch.float16, controlnet=controlnet, safety_checker=None).to(device)
    
    # Merge delta weights into the InstaFlow model
    pipe = merge_dW_to_unet(pipe, dW_dict=dW_dict, alpha=alpha)
    pipe = merge_dW_to_text_encoder(pipe, dW_dict=dW_dict_te, alpha=alpha)
    pipe.to(device)
    return pipe

def main(args):
    # Perform inference using the LoRA fine-tuned model
    if args.test_lora:
        pipe = test_lora(args.lora_ckpt, device=args.device, dtype=getattr(torch, args.dtype))
    else:
        pipe = get_lora_sd_pipeline_final(args.lora_ckpt, alpha=args.alpha, device=args.device, dtype=getattr(torch, args.dtype))
    # Reduce inference times by using a multistep scheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    # Load image
    img = Image.open(args.img_path)
    resolution = 512
    tile_condition_img = resize_for_condition_image(img, resolution)
    # Prepare edge mask for softedge control net
    processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
    edge_condition_image = processor(img, safe=True, image_resolution=resolution, detect_resolution=resolution)

    # Run inference using pipeline
    generator = torch.manual_seed(8)
    images = [tile_condition_img, edge_condition_image]
    prompt = "pushblock"
    time_start = time()
    if not args.test_lora:
        # Instaflow model uses 1 inference step for faster inference
        time_start = time()
        image = pipe(
                    prompt, 
                    num_inference_steps=1, 
                    guidance_scale=1.5,
                    controlnet_conditioning_scale=[0.6, 0.8],
                    generator = generator,
                    image=images,
                ).images[0]
    else:
        # Regular inference for LoRA fine-tuned model
        image = pipe(
                    prompt, 
                    num_inference_steps=10, 
                    guidance_scale=4.5,
                    controlnet_conditioning_scale=[0.6, 0.8],
                    generator = generator,
                    image=images,
                ).images[0]
    time_end = time()
    print(f"Inference time: {time_end - time_start:.2f} seconds")
    
    # Save image into inference testing directory
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    lora_ckpt_name = args.lora_ckpt.split('/')[-1]
    if not args.test_lora:
        image.save(os.path.join(args.outdir, "lora_inference_result_{}.png".format(lora_ckpt_name)))
    else:
        image.save(os.path.join(args.outdir, "lora_inference_result_test_{}.png".format(lora_ckpt_name)))
    tile_condition_img.save(os.path.join(args.outdir, "lora_inference_tile_condition.png"))
    edge_condition_image.save(os.path.join(args.outdir, "lora_inference_edge_condition.png"))
    print(f"Image saved to {os.path.join(args.outdir, 'lora_inference_result_{}.png'.format(lora_ckpt_name))}")

if __name__ == "__main__":
    # pipe = get_lora_sd_pipeline_final("/home/ethan/DiffusionResearch/Sim2RealDiffusion/training/checkpoints/lora_3000")
    # pipe = get_lora_sd_pipeline_final("/home/ethan/DiffusionResearch/Sim2RealDiffusion/training/checkpoints/instaflow_lora/checkpoint-500")
    # lora_ckpt = "/home/ethan/DiffusionResearch/Sim2RealDiffusion/training/checkpoints/base_lora/checkpoint-1000"
    # img = "/home/ethan/DiffusionResearch/Sim2RealDiffusion/inference/test_images/simsolid_interim_512.png"
    # outdir = "/home/ethan/DiffusionResearch/InstaFlow/code/lora_inference_results"

    parser = argparse.ArgumentParser(description="Load LoRA fine-tuned Stable Diffusion model")
    parser.add_argument("--lora_ckpt", type=str, required=True, help="Path to the LoRA checkpoint directory")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the input image for inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda)")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for the model (default: float16)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for merging LoRA weights (default: 1.0)")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory to save the model (default: current directory)")
    parser.add_argument("--test_lora", action='store_true', help="Test LoRA weights merged to base model instead of InstaFlow model")
    args = parser.parse_args()

    main(args)
    print("Inference completed successfully.")

    
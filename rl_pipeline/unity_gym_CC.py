# Modified unity_gym codebase for Compute Canada support - specifically for no network access
# Need to have ControlNet models installed locally and monkey patch the UnityEnvRegistry to not require internet access

# Monkey-patch mlagents to prevent network calls
import sys
# Patch the _load_all_manifests method to do nothing instead of loading remote manifests
from mlagents_envs.registry.unity_env_registry import UnityEnvRegistry
def patched_load_all_manifests(self) -> None:
    """Skip loading remote manifests - we only use local Unity environments"""
    self._manifests = []
    return
UnityEnvRegistry._load_all_manifests = patched_load_all_manifests

# Main Process
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents.trainers.cli_utils import load_config
import gym
import shimmy
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from gymnasium.core import ActType
from typing import Any
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.logger import configure
import numpy as np
import argparse
import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from pipeline_rf_ctrl import RectifiedFlowCtrlPipeline
from controlnet_aux import PidiNetDetector
import logging
import wandb

class UnityGymPipeline:
    def __init__(self, env_path, yaml_path, timesteps, timescale, diffusion_prompt, diffusion_model, base_port, out_type='img', control_condition=[0.5, 0.5], guidance_scale=4.5, denoise=10, rl_resolution=64, log_dir='logs', instaflow=False):
        self.env_path = env_path
        self.yaml_config = yaml_path
        self.timesteps = timesteps
        self.timescale = timescale
        self.diffusion_prompt = diffusion_prompt
        self.diffusion_model = diffusion_model
        self.out_type = out_type # Can either be 'img' or 'latent' for diffusion output
        self.control_condition = control_condition
        self.guidance_scale = guidance_scale
        self.denoise = denoise
        self.rl_res = rl_resolution
        self.log_dir = log_dir
        self.base_port = base_port
        self.instaflow = instaflow
        self.env = None # Unity-Gym environment loaded in create_env()
        self.seed = 499 # Seed for domain randomization

        # Validate input parameters
        if self.out_type not in ['img', 'latent']:
            raise ValueError("Invalid output type. Must be 'img' or 'latent'")
        if not os.path.exists(self.diffusion_model):
            raise FileNotFoundError(f"Diffusion model not found at {self.diffusion_model}")
        if not os.path.exists(self.env_path):
            raise FileNotFoundError(f"Unity environment not found at {self.env_path}")
        if not os.path.exists(self.yaml_config):
            raise FileNotFoundError(f"YAML configuration not found at {self.yaml_config}")
        if not os.path.exists(self.log_dir):
            if self.log_dir is not None:
                os.makedirs(self.log_dir, exist_ok=True)
            else:
                raise FileNotFoundError(f"Log directory not found at {self.log_dir}")
        
        # Override OpenAI Gym compatibility for step function changes
        shimmy.GymV21CompatibilityV0 = GymV21Compatibility
    
    def create_env(self):
        """Create a Unity environment based on the class path and wrap it in a gym environment for training"""
        # Load YAML configuration for Unity environment to be used for domain randomization
        config = load_config(self.yaml_config)
        # Configure timescale for Unity environment
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale = self.timescale)
        # Configure domain randomization parameters for Unity environment
        param_channel = EnvironmentParametersChannel()
        for k, v in config.items():
            if k == 'environment_parameters':
                for k2, v2 in v.items():
                    # Ensure that sampler_type and sampler_parameters are present in the YAML configuration
                    if 'sampler_type' not in v2 or 'sampler_parameters' not in v2:
                        raise ValueError("Invalid YAML configuration. Must have 'sampler_type' and 'sampler_parameters' keys.")
                    logger = logging.getLogger(__name__)
                    logger.info(f"Setting parameter {k2} to {v2}...")
                    if v2['sampler_type'] == 'uniform':
                        # Check if min and max values are present. If so use set_uniform_sampler_parameters from mlagents
                        if 'min_value' in v2['sampler_parameters'] and 'max_value' in v2['sampler_parameters']:
                            param_channel.set_uniform_sampler_parameters(k2, v2['sampler_parameters']['min_value'], v2['sampler_parameters']['max_value'], self.seed)
                        else:
                            param_channel.set_float_parameter(k2, v2['sampler_parameters']['value'])
                    else:
                        raise Warning("Parameter not being set for domain randomization. Only uniform sampler type is supported.")
        # Create Unity environment and wrap it in a Gym environment
        unity_env = UnityEnvironment(self.env_path, side_channels=[channel, param_channel], base_port=self.base_port)
        gym_env = UnityToGymWrapper(unity_env)
        # Wrap the environment in a custom observation wrapper for diffusion inference and load pipeline
        self.env = DiffusionPipeline(gym_env, self.diffusion_model, self.diffusion_prompt, self.out_type, self.control_condition, self.guidance_scale, self.denoise, self.rl_res, self.log_dir, self.instaflow)

    def train_ppo(self, resume=False):
        """Train a PPO policy using the Unity-Gym environment"""
        monitor_dump_dir = os.path.join(self.log_dir, f'ppo_{self.diffusion_prompt}_tensorboard')
        os.makedirs(monitor_dump_dir, exist_ok=True)
        # Set n_steps to 5 for smaller step training - useful for initial testing
        if not resume:
            model = PPO('CnnPolicy', self.env, verbose=1, tensorboard_log=monitor_dump_dir, stats_window_size=50, batch_size=256, n_steps=10240, n_epochs=3)
        else:
            # Resume training from latest checkpoint
            ckpt_files = [f for f in os.listdir(self.log_dir) if 'unity_rl_ckpt' in f]
            ckpt_files = [os.path.join(self.log_dir, f) for f in ckpt_files]
            if len(ckpt_files) == 0:
                raise FileNotFoundError("No checkpoint files found in log directory to resume from.")
            latest_ckpt = max(ckpt_files, key=os.path.getctime)
            model = PPO.load(latest_ckpt, env=self.env)
            model.verbose = 1
            model._stats_window_size = 50
            model.tensorboard_log = monitor_dump_dir
            # Load logger object for tensorboard logging
            logger = configure(monitor_dump_dir, ['tensorboard'])
            model.set_logger(logger)
        # Configure training for the PPO model
        checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=self.log_dir, name_prefix="unity_rl_ckpt", save_replay_buffer=True, save_vecnormalize=True, verbose=1)
        wandb_callback = WandbCallback(model_save_path=self.log_dir, model_save_freq=50000, gradient_save_freq=5000, verbose=2)
        callback_list = CallbackList([checkpoint_callback, wandb_callback])
        # Train model
        model.learn(total_timesteps=self.timesteps, progress_bar=True, callback=callback_list, reset_num_timesteps=False)
        # Save model
        model_save = os.path.join(self.log_dir, 'unity_model')
        model.save(model_save)
        return model
    
    def train_sac(self, resume=False):
        """Train a SAC policy using the Unity-Gym environment"""
        monitor_dump_dir = os.path.join(self.log_dir, f'sac_{self.diffusion_prompt}_tensorboard')
        os.makedirs(monitor_dump_dir, exist_ok=True)
        # Set n_steps to 5 for smaller step training - useful for initial testing
        if not resume:
            model = SAC('CnnPolicy', self.env, verbose=1, tensorboard_log=monitor_dump_dir, stats_window_size=50)
        else:
            # Resume training from latest checkpoint
            ckpt_files = [f for f in os.listdir(self.log_dir) if 'unity_rl_ckpt' in f]
            ckpt_files = [os.path.join(self.log_dir, f) for f in ckpt_files]
            if len(ckpt_files) == 0:
                raise FileNotFoundError("No checkpoint files found in log directory to resume from.")
            latest_ckpt = max(ckpt_files, key=os.path.getctime)
            model = SAC.load(latest_ckpt, env=self.env)
            model.verbose = 1
            model._stats_window_size = 50
            model.tensorboard_log = monitor_dump_dir
            # Load logger object for tensorboard logging
            logger = configure(monitor_dump_dir, ['tensorboard'])
            model.set_logger(logger)
        # Configure training for the SAC model
        checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=self.log_dir, name_prefix="unity_rl_ckpt", save_replay_buffer=True, save_vecnormalize=True, verbose=1)
        wandb_callback = WandbCallback(model_save_path=self.log_dir, model_save_freq=50000, gradient_save_freq=5000, verbose=2)
        callback_list = CallbackList([checkpoint_callback, wandb_callback])
        # Train model
        model.learn(total_timesteps=self.timesteps, progress_bar=True, callback=callback_list, reset_num_timesteps=False)
        # Save model
        model_save = os.path.join(self.log_dir, 'unity_model')
        model.save(model_save)
        return model
    
    def inference(self):
        """Use trained model to get Agent to perform task in Unity environment"""
        model = PPO.load(os.path.join(self.log_dir, 'unity_model'))
        obs = self._reset()
        while True:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = self._step(action)
            if terminated:
                break
    
    def inference_no_diffusion(self):
        """Use trained model to get Agent to perform task in Unity environment without diffusion processing for observations"""
        self.env.no_diffusion = True
        model = PPO.load(os.path.join(self.log_dir, 'unity_model'))
        obs = self._reset()
        while True:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = self._step(action)
            if terminated:
                break

    def _reset(self):
        """Reset the environment and return initial observation"""
        obs= self.env.reset() # Calls difusion pipeline wrapper for reset
        return obs
    
    def _step(self, action):
        """Step through the environment with the given action and return observation, reward, terminated, truncated, info"""
        result = self.env.step(action) # Calls diffusion pipeline wrapper for step
        if len(result) == 4:
            # Old Gym API
            obs, reward, done, info = result
            return obs, reward, done, False, info
        else:
            # New Gym API
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated, truncated, info
    
    def _close(self):
        """Close the environment and release resources"""
        self.env.close()
    
class DiffusionPipeline(gym.ObservationWrapper):
    def __init__(self, env, model_id, prompt, out_type, control_condition, guidance_scale, denoise, rl_resolution, log_dir, instaflow=False):
        super().__init__(env)
        self.model = model_id
        self.prompt = prompt
        self.out_type = out_type
        self.control_condition = control_condition
        self.guidance_scale = guidance_scale
        self.denoise = denoise
        self.rl_res = rl_resolution
        self.log_dir = log_dir
        self.instaflow = instaflow
        self.no_diffusion = False # Useful if wanting to do training or inference without diffusion processing
        # Diffusion parameters to be set on initialization
        self.pipe = None
        self.generator = None
        self.mask_processor = None
        
        # Initialize diffusion pipeline
        self.initialiize_diffusion_pipeline()
        # Set observation space to RL resolution and image format
        if self.out_type == 'img':
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, self.rl_res, self.rl_res), dtype=np.uint8)
        else:
            # Shape based on UNet latent output shape for SD model
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4, 64, 96), dtype=torch.float16)

    def reset(self, **kwargs):
        """Override reset to handle different Gym API versions"""
        obs = self.env.reset(**kwargs)
        # If obs is a tuple (from older Gym versions), extract just the observation
        if isinstance(obs, tuple):
            obs = obs[0]
        return self.observation(obs)
    
    def step(self, action):
        """Override step to ensure observation processing and handle different Gym API versions"""
        result = self.env.step(action)
        if len(result) == 4:
            # Old Gym API: obs, reward, done, info
            obs, reward, done, info = result
            return self.observation(obs), reward, done, False, info
        else:
            # New Gym API: obs, reward, terminated, truncated, info
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            return self.observation(obs), reward, done, False, info

    def initialiize_diffusion_pipeline(self):
        """
        Load fine-tuned diffusion model and control nets for inference. 
        Runs prior to environment reset to prevent inference overhead and uses optimized scheduler and xformers for faster inference.
        """
        # Initialize diffusion pipeline and parameters
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load tile and softedge control net models from local paths
        tile_control = ControlNetModel.from_pretrained('/src/Tile_ControlNet', torch_dtype=torch.float16)
        softedge_control = ControlNetModel.from_pretrained('/src/SoftEdge_ControlNet', torch_dtype=torch.float16)
        self.mask_processor = PidiNetDetector.from_pretrained('/src/Annotators')
        controlnet = [tile_control, softedge_control]
        # Apply control net to sim2real model to generate pipeline
        self.generator = torch.Generator(device='cpu').manual_seed(0)
        # Load either Instaflow or Stable Diffusion ControlNet pipeline based on user input
        logger = logging.getLogger(__name__)
        if self.instaflow:
            logger.info("Loading Instaflow Rectified Flow ControlNet Pipeline for inference...")
            self.pipe = RectifiedFlowCtrlPipeline.from_pretrained(self.model, controlnet=controlnet, torch_dtype=torch.float16).to(device)
        else:
            logger.info("Loading Stable Diffusion ControlNet Pipeline for inference...")
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self.model, controlnet=controlnet, torch_dtype=torch.float16).to(device)
        # Reduce inference times by using a multistep scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    def resize_for_condition_image(self, input_image: Image, resolution: int):
        """Resize input image to 64 multiple resolution for diffusion processing with ControlNet"""
        input_image = input_image.convert("RGB")
        W, H = input_image.size
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
        H = int(round(H / 64.0)) * 64
        W = int(round(W / 64.0)) * 64
        img = input_image.resize((W, H), resample=Image.LANCZOS)
        return img
    
    def post_process_image_output(self, output_image):
        """Post process output image by resizing to lower resolution and converting to CxHxW image format to match observation_space"""
        if self.out_type == 'img':
            output_image = output_image.resize((self.rl_res, self.rl_res), resample=Image.LANCZOS)
            post_output = np.array(output_image).transpose(2, 0, 1).astype(np.uint8)
            return post_output
        else:
            return output_image

    def observation(self, obs):
        """
        Automatic processing function of incoming observations.
        Convert observation to (H, W, C) form, with 0-255 pixel values from normalized 0-1 values and transform to Image object for diffusion processing.
        """
        # Preprocess Box observation to Image object
        obs = np.transpose(obs, (1, 2, 0))
        obs_img = (obs * 255).astype(np.uint8)
        obs_img = Image.fromarray(obs_img)
        if not self.no_diffusion:
            # Resample and resize image for tile control
            resolution = obs_img.size[0]
            tile_condition_img = self.resize_for_condition_image(obs_img, resolution)
            # Generate PIDI edge mask for softedge control
            edge_condition_image = self.mask_processor(obs_img, safe=True, image_resolution=resolution, detect_resolution=resolution)
            # Run inference using pipeline
            control_images = [tile_condition_img, edge_condition_image]
            if self.out_type == 'latent':
                # Return latent output for RL training. This is pre-decoded from the diffusion model.
                output_image = self.pipe(self.prompt, control_images, num_inference_steps=self.denoise, 
                                        generator=self.generator, controlnet_conditioning_scale=self.control_condition, 
                                        guidance_scale=self.guidance_scale, output_type="latent").images[0]
            elif self.instaflow:
                # Return decoded image output for RL training using Instaflow pipeline
                output_image = self.pipe(self.prompt, image=control_images, num_inference_steps=self.denoise, 
                                        generator=self.generator, controlnet_conditioning_scale=self.control_condition, 
                                        guidance_scale=self.guidance_scale).images[0]
            else:
                # Return decoded image output for RL training
                output_image = self.pipe(self.prompt, control_images, num_inference_steps=self.denoise, 
                                        generator=self.generator, controlnet_conditioning_scale=self.control_condition, 
                                        guidance_scale=self.guidance_scale).images[0]
        else:
            # No diffusion processing was set to True, return original observation in RGB format
            output_image = obs_img
        # Post process output based on out_type and return augmented observation
        aug_obs = self.post_process_image_output(output_image)
        # Save observation image for validation
        # self.save_obs_img(aug_obs, self.log_dir)
        return aug_obs
    
    def save_obs_img(self, obs, dir):
        """Save observation image to log directory for validation"""
        obs_img = Image.fromarray(obs.transpose(1, 2, 0))
        # Images saved as num.png for easy sorting
        obs_img.save(os.path.join(dir, f"{len(os.listdir(dir))}.png"))

class GymV21Compatibility(shimmy.GymV21CompatibilityV0):
    def step(self, action: ActType) -> tuple[Any, float, bool, bool, dict]:
        """Modified step function from Shimmy openai_gym_compatibility.py script to handle terminated and truncated flags"""
        result = self.gym_env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = False
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        if self.render_mode is not None:
            self.render()
        return convert_to_terminated_truncated_step_api((obs, reward, done, info))
    
def main():
    """Main function to run the UnityGymPipeline"""
    parser = argparse.ArgumentParser(description='Run Unity Gym Pipeline for RL training with diffusion model')
    parser.add_argument('--env_path', type=str, help='Path to Unity environment binary')
    parser.add_argument('--dr_yaml_path', type=str, help='Path to YAML configuration file for domain randomization')
    parser.add_argument('--diffusion_model', type=str, help='Path to fine-tuned diffusion model')
    parser.add_argument('--diffusion_prompt', type=str, help='Prompt used for diffusion model training')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs and models (default: logs)')
    parser.add_argument('--out_type', type=str, default='img', help="Output type from diffusion model, either 'img' or 'latent' (default: img)")
    parser.add_argument('--timesteps', type=int, default=1000000, help='Number of RL training timesteps (default: 1000000)')
    parser.add_argument('--timescale', type=int, default=4, help='Time scale for Unity environment (default: 4)')
    parser.add_argument('--control_condition', nargs=2, type=float, default=[1.2, 1.5], help='ControlNet conditioning scales for tile and softedge control nets (default: [1.2, 1.5])')
    parser.add_argument('--guidance_scale', type=float, default=4.5, help='Guidance scale for diffusion model (default: 4.5)')
    parser.add_argument('--denoise', type=int, default=10, help='Number of denoising steps for diffusion model (default: 10)')
    parser.add_argument('--rl_resolution', type=int, default=64, help='Resolution for RL training observations (default: 64)')
    parser.add_argument('--base_port', nargs='?', type=int, default=5004, help='Base port for Unity environment (default: 5004)')
    parser.add_argument('--wandb_project', type=str, default='unity_rl_pipeline', help='Weights & Biases project name for logging (default: unity_rl_pipeline)')
    parser.add_argument('--instaflow', action='store_true', help='Using Instaflow diffusion model pipeline')
    
    args = parser.parse_args()

    # Initialize Weights & Biases for logging. Store configuration parameters
    wandb.init(
        project=args.wandb_project,
        entity="medcvr",
        name="exp1",
        config={
            'env_path': args.env_path,
            'dr_yaml_path': args.dr_yaml_path,
            'diffusion_model': args.diffusion_model,
            'diffusion_prompt': args.diffusion_prompt,
            'out_type': args.out_type,
            'control_condition': args.control_condition,
            'guidance_scale': args.guidance_scale,
            'denoise': args.denoise,
            'rl_resolution': args.rl_resolution,
            'timesteps': args.timesteps,
            'timescale': args.timescale,
            'log_dir': args.log_dir
        },
        sync_tensorboard=True,
        monitor_gym=True
    )

    logger = logging.getLogger(__name__)
    logger.info("=================== Unity Gym Pipeline ==================")
    logger.info(f"Environment Path: {args.env_path}")
    logger.info(f"Domain Randomization YAML: {args.dr_yaml_path}")
    logger.info(f"Diffusion Model Path: {args.diffusion_model}")
    logger.info(f"Diffusion Prompt: {args.diffusion_prompt}")
    logger.info(f"Output Type: {args.out_type}")
    logger.info(f"ControlNet Conditioning Scales: {args.control_condition}")
    logger.info(f"Guidance Scale: {args.guidance_scale}")
    logger.info(f"Denoising Steps: {args.denoise}")
    logger.info(f"RL Resolution: {args.rl_resolution}")
    logger.info(f"Training Timesteps: {args.timesteps}")
    logger.info(f"Time Scale: {args.timescale}")
    logger.info(f"Log Directory: {args.log_dir}")
    logger.info(f"Using an Instaflow Diffusion Pipeline: {args.instaflow}")
    logger.info("========================================================")
    
    unity_pipeline = UnityGymPipeline(args.env_path,
                                      args.dr_yaml_path,
                                      args.timesteps,
                                      args.timescale,
                                      args.diffusion_prompt,
                                      args.diffusion_model,
                                      args.base_port,
                                      args.out_type,
                                      args.control_condition,
                                      args.guidance_scale,
                                      args.denoise,
                                      args.rl_resolution,
                                      args.log_dir,
                                      args.instaflow)
    
    logger.info("Creating Unity environment...")
    unity_pipeline.create_env()
    logger.info("Starting PPO training...")
    model = unity_pipeline.train_ppo()
    logger.info("Training complete. Closing environment...")
    # Uncomment to run inference without diffusion processing
    # for i in range(5):
    #     logger.info(f"Running inference {i+1}...")
    #     unity_pipeline.inference_no_diffusion()
    unity_pipeline._close()
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
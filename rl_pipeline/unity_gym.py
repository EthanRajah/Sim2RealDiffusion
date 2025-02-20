from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents.trainers.cli_utils import load_config
import gym
from shimmy import GymV21CompatibilityV0
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from gymnasium.core import ActType
from typing import Any
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
import numpy as np
import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import PidiNetDetector

class UnityGymPipeline:
    def __init__(self, env_path, yaml_path, timesteps, timescale, diffusion_prompt, diffusion_model, out_type='img', control_condition=[0.5, 0.5], guidance_scale=4.5, denoise=10, rl_resolution=64, log_dir='logs'):
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
                    print(f"Setting parameter {k2} to {v2}...")
                    if v2['sampler_type'] == 'uniform':
                        # Check if min and max values are present. If so use set_uniform_sampler_parameters from mlagents
                        if 'min_value' in v2['sampler_parameters'] and 'max_value' in v2['sampler_parameters']:
                            param_channel.set_uniform_sampler_parameters(k2, v2['sampler_parameters']['min_value'], v2['sampler_parameters']['max_value'], self.seed)
                        else:
                            param_channel.set_float_parameter(k2, v2['sampler_parameters']['value'])
                    else:
                        raise Warning("Parameter not being set for domain randomization. Only uniform sampler type is supported.")
            else:
                raise Warning("No environment parameters found in YAML configuration. Domain randomization will not be applied.")
        # Create Unity environment and wrap it in a Gym environment
        unity_env = UnityEnvironment(self.env_path, side_channels=[channel, param_channel])
        gym_env = UnityToGymWrapper(unity_env)
        gym_env.close()
        # Wrap the environment in a custom observation wrapper for diffusion inference and load pipeline
        self.env = DiffusionPipeline(gym_env, self.diffusion_model, self.diffusion_prompt, self.out_type, self.control_condition, self.guidance_scale, self.denoise, self.rl_res, self.log_dir)

    def train_ppo(self, resume=False):
        """Train a PPO model using the Unity-Gym environment"""
        # Create a monitoring wrapper for the environment
        monitor_dump_dir = os.path.join(self.log_dir, f'ppo_{self.diffusion_prompt}_tensorboard')
        os.makedirs(monitor_dump_dir, exist_ok=True)
        # Configure training for the PPO model
        checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=self.log_dir, name_prefix="unity_rl_ckpt", save_replay_buffer=True, save_vecnormalize=True, verbose=1)
        # Set n_steps to 1 for single step training - useful for initial testing
        if not resume:
            model = PPO('CnnPolicy', self.env, verbose=1, tensorboard_log=monitor_dump_dir)
        else:
            # Resume training from latest checkpoint
            ckpt_files = [f for f in os.listdir(self.log_dir) if 'unity_rl_ckpt' in f]
            if len(ckpt_files) == 0:
                raise FileNotFoundError("No checkpoint files found in log directory to resume from.")
            latest_ckpt = max(ckpt_files, key=os.path.getctime)
            model = PPO.load(os.path.join(self.log_dir, latest_ckpt), env=self.env)
            # Load logger object for tensorboard logging
            logger = configure(monitor_dump_dir, ['tensorboard'])
            model.set_logger(logger)
        # Train model
        model.learn(total_timesteps=self.timesteps, progress_bar=True, callback=checkpoint_callback)
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
        self._close()

    def _reset(self):
        """Reset the environment and return initial observation"""
        obs= self.env.reset()
        return obs
    
    def _step(self, action):
        """Step through the environment with the given action and return observation, reward, terminated, truncated, info"""
        result = self.env.step(action)
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
    def __init__(self, env, model_id, prompt, out_type, control_condition, guidance_scale, denoise, rl_resolution, log_dir):
        super().__init__(env)
        self.model = model_id
        self.prompt = prompt
        self.out_type = out_type
        self.control_condition = control_condition
        self.guidance_scale = guidance_scale
        self.denoise = denoise
        self.rl_res = rl_resolution
        self.log_dir = log_dir
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
        # Load tile and softedge control net models
        tile_control = ControlNetModel.from_pretrained('lllyasviel/control_v11f1e_sd15_tile', torch_dtype=torch.float16)
        softedge_control = ControlNetModel.from_pretrained('lllyasviel/control_v11p_sd15_softedge', torch_dtype=torch.float16)
        self.mask_processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        controlnet = [tile_control, softedge_control]
        # Apply control net to sim2real model to generate pipeline
        self.generator = torch.Generator(device='cpu').manual_seed(0)
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
        # Resample and resize image for tile control
        resolution = obs_img.size[0]
        tile_condition_img = self.resize_for_condition_image(obs_img, resolution)
        # Generate PIDI edge mask for softedge control
        edge_condition_image = self.mask_processor(obs_img, safe=True, image_resolution=resolution, detect_resolution=resolution)
        # Run inference using pipeline
        control_images = [tile_condition_img, edge_condition_image]
        if self.out_type == 'latent':
            # Return latent output for RL training. This is pre-decoded from the diffusion model.
            output_image = self.pipe(self.prompt, control_images, num_inference_steps=10, 
                                     generator=self.generator, controlnet_conditioning_scale=self.control_condition, 
                                     guidance_scale=self.guidance_scale, output_type="latent").images[0]
        else:
            # Return decoded image output for RL training
            output_image = self.pipe(self.prompt, control_images, num_inference_steps=10, 
                                    generator=self.generator, controlnet_conditioning_scale=self.control_condition, 
                                    guidance_scale=self.guidance_scale).images[0]
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

class GymV21Compatibility(GymV21CompatibilityV0):
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
    
if __name__ == '__main__':
    env_path = '/home/ethan/DiffusionResearch/Sim2RealDiffusion/rl_pipeline/PushBlockBuild_512DR/pushblock_solid_dr.x86_64' # Linux path
    # env_path = '/Users/ethan/Documents/Robotics/Thesis/DiffusionResearch/Sim2RealDiffusion/rl_pipeline/pushblock_solid.app' # Mac path
    yaml_path = '/Users/ethan/Documents/Robotics/Thesis/MEDCVR_Unity/medcvr_localsims/dvrk_mlagents/unity_project/Assets/Tasks/PushBlock/Scripts/DiffusionPushBlock.yaml'
    diffusion_prompt = 'pushblock'
    diffusion_model = '/home/ethan/DiffusionResearch/Sim2RealDiffusion/inference/solid_pushblock/model_v8/2000'
    log_dir = '/home/ethan/DiffusionResearch/Sim2RealDiffusion/rl_pipeline/test1'
    out_type = 'img'
    timesteps = 1000000
    timescale = 4
    control_condition = [1.2, 1.5]
    guidance_scale = 4.5
    denoise = 10
    rl_resolution = 64

    unity_pipeline = UnityGymPipeline(env_path, yaml_path, timesteps, timescale, diffusion_prompt, diffusion_model, out_type, control_condition, guidance_scale, denoise, rl_resolution, log_dir)
    unity_pipeline.create_env()
    model = unity_pipeline.train_ppo()
    # unity_pipeline._reset()
    # unity_pipeline._step(unity_pipeline.env.action_space.sample().reshape(1, 2))
    unity_pipeline._close()


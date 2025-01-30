from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import gym
import stable_baselines3.ppo.ppo as ppo
import numpy as np
import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import PidiNetDetector

class UnityGymPipeline:
    def __init__(self, env_path, timesteps, diffusion_prompt, diffusion_model):
        self.env_path = env_path
        self.timesteps = timesteps
        self.diffusion_prompt = diffusion_prompt
        self.diffusion_model = diffusion_model
        self.env = None # Unity-Gym environment loaded in create_env()
    
    def create_env(self):
        """Create a Unity environment based on the class path and wrap it in a gym environment for training"""
        unity_env = UnityEnvironment(self.env_path)
        gym_env = UnityToGymWrapper(unity_env)
        # Wrap the environment in a custom observation wrapper for diffusion inference and load pipeline
        self.env = DiffusionPipeline(gym_env, self.diffusion_model, self.diffusion_prompt)

    def _reset(self):
        """Reset the environment and return initial observation"""
        obs= self.env.reset()
        return obs
    
    def _step(self, action):
        """Step through the environment with the given action and return observation, reward, terminated, truncated, info"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Close the environment and release resources"""
        self.env.close()
    
class DiffusionPipeline(gym.ObservationWrapper):
    def __init__(self, env, model_id, prompt):
        super().__init__(env)
        self.model = model_id
        self.prompt = prompt
        self.pipe = None
        self.generator = None
        
        # Modify image observation space to reflect (H, W, C) format, maintaining normalized pixel values. Used for RL training.
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_shape[1], obs_shape[2], obs_shape[0]), dtype=np.float16)
        # Initialize diffusion pipeline
        #self.initialiize_diffusion_pipeline()

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
            return self.observation(obs), reward, terminated, truncated, info

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
        controlnet = [tile_control, softedge_control]
        # Apply control net to sim2real model to generate pipeline
        self.generator = torch.Generator(device='cpu').manual_seed(0)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self.model, controlnet=controlnet, torch_dtype=torch.float16).to(device)
        # Reduce inference times by using a multistep scheduler
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

    def resize_for_condition_image(input_image: Image, resolution: int):
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

    def observation(self, obs):
        """
        Automatic processing function of incoming observations.
        Convert observation to (H, W, C) form, with 0-255 pixel values from normalized 0-1 values and transform to Image object for diffusion processing.
        """
        obs = np.transpose(obs, (1, 2, 0))
        obs_img = (obs * 255).astype(np.uint8)
        obs_img = Image.fromarray(obs_img)
        obs_img.save('observation.png')
        return obs
        
    
if __name__ == '__main__':
    env_path = '/home/ethan/DiffusionResearch/Sim2RealDiffusion/rl_pipeline/PushBlockBuild/pushblock_solid.x86_64'
    timesteps = 1000
    diffusion_prompt = 'pushblock'
    diffusion_model = '/home/ethan/DiffusionResearch/Sim2RealDiffusion/inference/solid_pushblock/model_v8/2000'

    unity_pipeline = UnityGymPipeline(env_path, timesteps, diffusion_prompt, diffusion_model)
    unity_pipeline.create_env()
    unity_pipeline._reset()
    unity_pipeline._step(unity_pipeline.env.action_space.sample().reshape(1, 2))
    unity_pipeline.close()


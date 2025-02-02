from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gym
from stable_baselines3 import PPO
from gym.wrappers import monitoring
import numpy as np
import os
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import PidiNetDetector

class UnityGymPipeline:
    def __init__(self, env_path, timesteps, timescale, diffusion_prompt, diffusion_model, control_condition=[0.5, 0.5], guidance_scale=4.5, denoise=10, rl_resolution=64):
        self.env_path = env_path
        self.timesteps = timesteps
        self.timescale = timescale
        self.diffusion_prompt = diffusion_prompt
        self.diffusion_model = diffusion_model
        self.control_condition = control_condition
        self.guidance_scale = guidance_scale
        self.denoise = denoise
        self.rl_res = rl_resolution
        self.env = None # Unity-Gym environment loaded in create_env()
    
    def create_env(self):
        """Create a Unity environment based on the class path and wrap it in a gym environment for training"""
        unity_env = UnityEnvironment(self.env_path)
        gym_env = UnityToGymWrapper(unity_env)
        # Wrap the environment in a custom observation wrapper for diffusion inference and load pipeline
        self.env = DiffusionPipeline(gym_env, self.diffusion_model, self.diffusion_prompt, self.control_condition, self.guidance_scale, self.denoise, self.rl_res)

    def train_ppo(self):
        """Train a PPO model using the Unity-Gym environment"""
        # Configure timescale for Unity environment
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale = self.timescale)
        # Create a monitoring wrapper for the environment
        monitor_dump_dir = os.path.join(os.path.dirname(__file__), 'gym_monitor')
        os.makedirs(monitor_dump_dir, exist_ok=True)
        self.env = monitoring.Monitor(self.env, monitor_dump_dir, allow_early_resets=True)
        # Train PPO model
        model = PPO('CnnPolicy', self.env, verbose=1)
        model.learn(total_timesteps=self.timesteps)
        model.save("unity_model")
        model = PPO.load("unity_model")
        return model
    
    def inference(self, model):
        """Use trained model to get Agent to perform task in Unity environment"""
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
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def _close(self):
        """Close the environment and release resources"""
        self.env.close()
    
class DiffusionPipeline(gym.ObservationWrapper):
    def __init__(self, env, model_id, prompt, control_condition, guidance_scale, denoise, rl_resolution):
        super().__init__(env)
        self.model = model_id
        self.prompt = prompt
        self.control_condition = control_condition
        self.guidance_scale = guidance_scale
        self.denoise = denoise
        self.rl_res = rl_resolution
        self.pipe = None
        self.generator = None
        self.mask_processor = None
        
        # Initialize diffusion pipeline
        self.initialiize_diffusion_pipeline()

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
        """Post process output image by resizing to lower resolution and converting to normalized numpy array"""
        output_image = output_image.resize((self.rl_res, self.rl_res), resample=Image.LANCZOS)
        output = np.array(output_image, dtype=np.float16) / 255
        output = np.transpose(output, (2, 0, 1))
        return output

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
        output_image = self.pipe(self.prompt, control_images, num_inference_steps=10, 
                                 generator=self.generator, controlnet_conditioning_scale=self.control_condition, 
                                 guidance_scale=self.guidance_scale).images[0]
        # Post process output based on diffusion_type and return augmented observation
        aug_obs = self.post_process_image_output(output_image)
        return aug_obs
    
if __name__ == '__main__':
    env_path = '/home/ethan/DiffusionResearch/Sim2RealDiffusion/rl_pipeline/PushBlockBuild_512/pushblock_solid.x86_64' # Linux path
    # env_path = '/Users/ethan/Documents/Robotics/Thesis/DiffusionResearch/Sim2RealDiffusion/rl_pipeline/pushblock_solid.app' # Mac path
    diffusion_prompt = 'pushblock'
    diffusion_model = '/home/ethan/DiffusionResearch/Sim2RealDiffusion/inference/solid_pushblock/model_v8/2000'
    timesteps = 1000
    timescale = 4
    control_condition = [1.2, 1.5]
    guidance_scale = 4.5
    denoise = 10
    rl_resolution = 64

    unity_pipeline = UnityGymPipeline(env_path, timesteps, timescale, diffusion_prompt, diffusion_model, control_condition, guidance_scale, denoise, rl_resolution)
    unity_pipeline.create_env()
    model = unity_pipeline.train_ppo()
    # unity_pipeline._reset()
    # unity_pipeline._step(unity_pipeline.env.action_space.sample().reshape(1, 2))
    # unity_pipeline._close()


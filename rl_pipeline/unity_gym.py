from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import gymnasium as gym
import stable_baselines3.ppo.ppo as ppo
import numpy as np
import os
from PIL import Image

class UnityGymPipeline:
    def __init__(self, env_path, timesteps, diffusion_prompt):
        self.env_path = env_path
        self.timesteps = timesteps
        self.diffusion_prompt = diffusion_prompt
        self.env = None # Unity environment loaded in create_env()
    
    def create_env(self):
        '''Create a Unity environment based on the class path and wrap it in a gym environment for training'''
        unity_env = UnityEnvironment(self.env_path)
        gym_env = UnityToGymWrapper(unity_env)
        # Wrap the environment in a custom observation wrapper for diffusion inference
        self.env = DiffusionPipeline(gym_env)

    def reset(self):
        '''Reset the environment and return the initial observation'''
        return self.env.reset()
    
    def step(self, action):
        '''Step through the environment with the given action and return the next observation'''
        return self.env.step(action)
    
class DiffusionPipeline(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # Modify image observation space to reflect (H, W, C) format, maintaining normalized pixel values
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_shape[1], obs_shape[2], obs_shape[0]), dtype=np.float16)

    def observation(self, obs):
        '''Automatic processing function of incoming observations.
        Convert observation to 0-255 pixel values from normalized 0-1 values and transform to Image object for diffusion processing.
        '''
        obs_img = (obs * 255).astype(np.uint8)
        obs_img = Image.fromarray(obs)
        
    
if __name__ == '__main__':
    env_path = '/home/ethan/DiffusionResearch/Sim2RealDiffusion/rl_pipeline/PushBlockBuild/pushblock_solid.x86_64'
    timesteps = 1000
    diffusion_prompt = 'pushblock'
    unity_pipeline = UnityGymPipeline(env_path, timesteps, diffusion_prompt)
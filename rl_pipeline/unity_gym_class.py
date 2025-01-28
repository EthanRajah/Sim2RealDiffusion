from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
import gym
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import stable_baselines3.ppo.ppo as ppo
import numpy as np
import os

class UnityGymPipeline:
    def __init__(self, env_path, timesteps, diffusion_prompt):
        self.env_path = env_path
        self.timesteps = timesteps
        self.diffusion_prompt = diffusion_prompt
    
    def create_env(self):
        '''Create a Unity environment based on the class path and wrap it in a gym environment for training'''
        unity_env = UnityEnvironment(self.env_path)
        env = UnityToGymWrapper(unity_env)
        return env
    
    def get_obs_space(self, env):
        '''Get the observation space of the environment'''
        return env.observation_space
    
if __name__ == '__main__':
    env_path = 'path/to/unity/environment'
    timesteps = 1000
    diffusion_prompt = 'pushblock'
    unity_pipeline = UnityGymPipeline(env_path, timesteps, diffusion_prompt)
    env = unity_pipeline.create_env()
    obs_space = unity_pipeline.get_obs_space(env)
    print(obs_space)
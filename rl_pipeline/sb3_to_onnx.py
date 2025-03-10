import torch
from torch import nn
from torch.nn import Parameter
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
import onnx
import onnxruntime as ort
import numpy as np

# Wrap SB3 policy so that it can be exported to ONNX
class OnnxableSB3Policy(nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor):
        # The policy returns (continuous_actions, value_estimate, log_prob)
        return self.policy(observation, deterministic=True)

# Create a wrapper that adds extra constants as new parameters to the policy module, required by Unity ML-Agents
class WrapperNet(nn.Module):
    def __init__(self, policy_module: nn.Module, continuous_output_size: int):
        super(WrapperNet, self).__init__()
        self.policy_module = policy_module
        
        # ML-Agents2_0 expects a version_number of 3
        version_number = torch.Tensor([3])
        self.version_number = Parameter(version_number, requires_grad=False)
        
        # No recurrent memory is used, so memory_size is 0
        memory_size = torch.Tensor([0])
        self.memory_size = Parameter(memory_size, requires_grad=False)
        
        # continuous_action_output_shape: the number of continuous actions output by policy.
        continuous_action_output_shape = torch.Tensor([continuous_output_size])
        self.continuous_shape = Parameter(continuous_action_output_shape, requires_grad=False)
        
    def forward(self, obs: torch.Tensor):
        # Get outputs from the original policy (continuous_actions, value_estimate, log_prob)
        continuous_actions, value_estimate, log_prob = self.policy_module(obs)
        # Return only the continuous actions along with the extra constants.
        # Unity expects outputs in this order:
        # 1. continuous_actions
        # 2. continuous_action_output_shape
        # 3. version_number
        # 4. memory_size
        return continuous_actions, self.continuous_shape, self.version_number, self.memory_size

# Load trained PPO model
model = PPO.load("unity_rl_ckpt_220000_steps", device="cpu")
onnx_policy = OnnxableSB3Policy(model.policy)

# Wrap the policy module. Change continuous_output_size if model outputs a different number.
wrapper_net = WrapperNet(onnx_policy, continuous_output_size=2)

# Get the observation space shape from model
observation_size = model.observation_space.shape
dummy_input = torch.randn(1, *observation_size)

# Export the wrapped network to ONNX
torch.onnx.export(
    wrapper_net,
    dummy_input,
    "ppo_model.onnx",
    opset_version=11,  # Barracuda: opset 11
    input_names=["obs_0"],
    output_names=["continuous_actions", "continuous_action_output_shape", "version_number", "memory_size"],
    dynamic_axes={
        'obs_0': {0: 'batch'},
        'continuous_actions': {0: 'batch'},
    }
)

# Test the exported model using ONNX Runtime.
onnx_model = onnx.load("ppo_model.onnx")
onnx.checker.check_model(onnx_model)

observation = np.zeros((1, *observation_size), dtype=np.float32)
ort_sess = ort.InferenceSession("ppo_model.onnx")
outputs = ort_sess.run(None, {"obs_0": observation})
print("ONNX Runtime outputs:", outputs)

# For comparison, check predictions from PyTorch model.
with torch.no_grad():
    pytorch_out = model.policy(torch.as_tensor(observation), deterministic=True)
    print("PyTorch model outputs:", pytorch_out)
import torch
from torch import nn
from torch.nn import Parameter
from stable_baselines3 import PPO
from stable_baselines3.common.policies import BasePolicy
import onnx
import onnxruntime as ort
import numpy as np
import os

# Wrap SB3 policy so that it can be exported to ONNX
class OnnxableSB3Policy(nn.Module):
    def __init__(self, policy: BasePolicy):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor):
        # Get raw outputs from the policy; note that PPO's policy returns (actions, values, log_prob)
        actions, value_estimate, log_prob = self.policy(observation, deterministic=False)
        # Clip actions to be within the valid action space
        low = torch.tensor(self.policy.action_space.low, dtype=actions.dtype, device=actions.device)
        high = torch.tensor(self.policy.action_space.high, dtype=actions.dtype, device=actions.device)
        actions = torch.clamp(actions, low, high)
        # Clone the tensor to avoid in-place modification issues
        swappedActions = actions.clone()
        # Swap first two columns to match Unity's coordinate system
        # swappedActions[:, 0] = actions[:, 1]
        # swappedActions[:, 1] = actions[:, 0]
        return swappedActions, value_estimate, log_prob


# Create a wrapper that adds extra constants required by Unity ML-Agents
class WrapperNet(nn.Module):
    def __init__(self, policy_module: nn.Module, continuous_output_size: int):
        super(WrapperNet, self).__init__()
        self.policy_module = policy_module
        
        # ML-Agents v2.0 expects a version_number of 3
        version_number = torch.tensor([3], dtype=torch.float32)
        self.version_number = Parameter(version_number, requires_grad=False)
        
        # No recurrent memory is used, so memory_size is 0
        memory_size = torch.tensor([0], dtype=torch.float32)
        self.memory_size = Parameter(memory_size, requires_grad=False)
        
        # continuous_action_output_shape: number of continuous actions output by the policy.
        continuous_action_output_shape = torch.tensor([continuous_output_size], dtype=torch.float32)
        self.continuous_shape = Parameter(continuous_action_output_shape, requires_grad=False)
        
    def forward(self, obs: torch.Tensor):
        continuous_actions, value_estimate, log_prob = self.policy_module(obs)
        return continuous_actions, self.continuous_shape, self.version_number, self.memory_size

# Load trained PPO model
model = PPO.load("unity_rl_ckpt_530000_steps", device="cpu")
# If Unity is providing normalized images already, disable PPO image normalization:
model.policy.normalize_images = False

# Set the model's policy to evaluation mode
model.policy.eval()

# Wrap the SB3 policy for export
onnx_policy = OnnxableSB3Policy(model.policy)
onnx_policy.eval()

# Wrap further to add Unity ML-Agents constants.
# Adjust continuous_output_size if your action space differs.
wrapper_net = WrapperNet(onnx_policy, continuous_output_size=2)
wrapper_net.eval()

# Prepare dummy input
# For example, if you have an observation saved in "obs.npy" that matches your environment,
# load it, add the batch dimension, and convert to float.
dummy_input = torch.from_numpy(np.load("obs.npy")).unsqueeze(0).float() / 255.0

# Export the wrapped network to ONNX
with torch.no_grad():
    torch.onnx.export(
        wrapper_net,
        dummy_input,
        "ppo_model_530.onnx",
        opset_version=14,  # Adjust as needed for Barracuda (often opset 11 or higher)
        input_names=["obs_0"],
        output_names=["continuous_actions", "continuous_action_output_shape", "version_number", "memory_size"],
        dynamic_axes={
            'obs_0': {0: 'batch'},
            'continuous_actions': {0: 'batch'},
        }
    )

# Test the exported model using ONNX Runtime.
onnx_model = onnx.load("ppo_model_530.onnx")
onnx.checker.check_model(onnx_model)

# Create a random test observation matching the environment's shape.
observation = np.random.randn(1, *model.observation_space.shape).astype(np.float32)
ort_sess = ort.InferenceSession("ppo_model_530.onnx")
outputs = ort_sess.run(None, {"obs_0": observation})
print("ONNX Runtime outputs:", outputs)

# For comparison, get predictions directly from the PyTorch model.
with torch.no_grad():
    pytorch_out = model.policy(torch.as_tensor(observation), deterministic=False)
    print("PyTorch model outputs:", pytorch_out)
from pathlib import Path
from typing import Optional, Dict
import gym
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

class Atari_PPO_Adapted_CNN(BaseFeaturesExtractor):
    """
    Based on https://github.com/DarylRodrigo/rl_lib/blob/master/PPO/Models.py
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(Atari_PPO_Adapted_CNN, self).__init__(observation_space,features_dim)
        channels = observation_space.shape[0]

        self.network = nn.Sequential(
            # Scale(1/255),
            layer_init(nn.Conv2d(channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, features_dim)),
            # nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations=observations.view(observations.shape[0],-1,*observations.shape[3:])
        #print(observations.shape)
        return self.network(observations)

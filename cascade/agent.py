import dataclasses
from typing import Any

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal

from common import ActionAndValue, layer_init


@dataclasses.dataclass()
class ActionAndCascade(ActionAndValue):
    cascade_final_value: Any  # prediction for the final head - the timescale of interest
    cascade_values: Any


@dataclasses.dataclass()
class ValueAndCascade():
    value: Any
    cascade_final_value: Any
    cascade_values: Any


class Agent(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape, num_gammas: int):
        super().__init__()
        self.cascade = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_gammas), std=1.0),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_space_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space_shape)))

    def get_value(self, x):
        cascade_heads = self.cascade(x)
        cascade_values = torch.cumsum(cascade_heads, 1)
        # return values, values[...,-1]
        return ValueAndCascade(value=self.critic(x), cascade_final_value=cascade_values[...,-1], cascade_values=cascade_values)

    def get_action_and_value(self, x, action=None, greedy: bool = False):
        action_mean = self.actor_mean(x)

        if greedy:
            action = action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        value_and_value_heads = self.get_value(x)

        return ActionAndCascade(action=action,
                                logprob=probs.log_prob(action).sum(1),
                                entropy=probs.entropy().sum(1),
                                value=value_and_value_heads.value,
                                cascade_final_value=value_and_value_heads.cascade_final_value,
                                cascade_values=value_and_value_heads.cascade_values)

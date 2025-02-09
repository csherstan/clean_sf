import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal

from common import layer_init, ActionAndValue


class Agent(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape):
        super().__init__()
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
        return self.critic(x)

    def get_action_and_value(self, x, action=None, greedy: bool = False):
        action_mean = self.actor_mean(x)

        if greedy:
            action = action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        return ActionAndValue(action=action, logprob=probs.log_prob(action).sum(1), entropy=probs.entropy().sum(1),
                              value=self.critic(x))

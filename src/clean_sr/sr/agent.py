import dataclasses

import numpy as np
import torch
from torch import nn as nn
from torch.distributions import Normal

from clean_sr.common import layer_init, ActionAndValue

@dataclasses.dataclass
class SROutput:
    phi: torch.Tensor
    """
    The features on which the SF is based
    """
    sf: torch.Tensor
    """
    The successor features
    """
    w: torch.Tensor
    """
    Output of the ?
    """
    r: torch.Tensor
    """
    One step reward prediction
    """
    value: torch.Tensor
    """
    Standard value estimate
    """

class SRCritic(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape, phi_size: int):
        super().__init__()
        self.neck = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, phi_size)),
            nn.Tanh(),
        )

        # I've combined output of w and SR here, but maybe I shouldn't, their gradients will interact and
        # weights end up shared.
        self.head = nn.Sequential(
            layer_init(nn.Linear(phi_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, phi_size * 2))
        )

        self._phi_size = phi_size

    def forward(self, x) -> SROutput:
        phi = self.neck(x)  # [batch, phi]
        output = self.head(phi)  # [batch, phi*2]
        sf = output[:, 0:self._phi_size]  # [batch, phi]
        w = output[:, self._phi_size:]  # [batch, phi]
        r = torch.einsum("bm,bm->b", phi, w)[..., None]  # [batch, 1]
        value = torch.einsum("bm,bm->b", sf, w)[..., None]  # [batch, 1]

        return SROutput(phi=phi, sf=sf, w=w, r=r, value=value)


@dataclasses.dataclass()
class SRActionAndValue(ActionAndValue):
    sr_output: SROutput


class Agent(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape, phi_size: int):
        super().__init__()
        self.critic = SRCritic(obs_space_shape=obs_space_shape, action_space_shape=action_space_shape,
                               phi_size=phi_size)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_space_shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_space_shape)))

    def get_critic(self, x):
        return self.critic(x)

    def get_value(self, x):
        return self.critic(x)["value"]

    def get_action_and_value(self, x, action=None, greedy: bool = False):
        action_mean = self.actor_mean(x)

        if greedy:
            action = action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        sr_output = self.critic(x)

        return SRActionAndValue(action=action,
                                logprob=probs.log_prob(action).sum(1),
                                entropy=probs.entropy().sum(1),
                                value=sr_output.value,
                                sr_output=sr_output)

import dataclasses
from typing import Any, Optional

import torch
import torch.nn as nn
import tyro
from torch.distributions import Normal
from tqdm import tqdm
import gymnasium as gym
import numpy as np

from common import layer_init


class GaussianPolicy(nn.Module):

    def __init__(self, obs_space_shape, action_space_shape):
        super().__init__()
        self.mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_space_shape)), std=0.01),
        )

        self.log_std = nn.Parameter(torch.zeros(1, np.prod(action_space_shape)))


    def get_action(self, x, action=None, greedy: bool = False):
        mean = self.mean(x)
        if greedy:
            action = mean

        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        probs = Normal(mean, std)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action).sum(1)

def make_env(env_id, idx, capture_video, run_name) -> gym.Env:
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        return env

    return thunk

@dataclasses.dataclass()
class Args:
    run_name: str = "default"
    env_id: str = "dm_control/cartpole-swingup_sparse-v0"

    num_steps: int = 1000

@dataclasses.dataclass()
class Step:
    obs: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Optional[dict[str, Any]]


def one_episode(env: gym.Env, policy: GaussianPolicy):
    obs, _ = env.reset()
    done = False
    steps = [Step(obs, 0.0, False, False, {})]
    while not done:
        action, log_prob = policy.get_action(obs)
        step = Step(*env.step(action))
        steps.append(step)

        done = step.truncated or step.terminated

    return steps

def to_dict_list(steps: list[Step]) -> dict[str, list[Any]]:
    data = {}

    for step in steps:
        for key, val in dataclasses.asdict(step).items():
            data.setdefault(key, []).append(val)

    return data


def compute_returns(rewards, gammas: torch.Tensor):
    returns = []
    g = np.zeros(len(gammas))
    for reward in reversed(rewards):
        returns.append(g * gammas + reward)

    return torch.tensor(returns)


def one_update(steps: list[Step], policy: GaussianPolicy, gammas: torch.Tensor):
    step_dict = to_dict_list(steps)
    returns = compute_returns(step_dict["reward"][1:], gammas)





if __name__==("__main__"):
    args = tyro.cli(Args)

    env = make_env(args.env_id, 0, capture_video=False, run_name=args.run_name)

    policy = GaussianPolicy(env.observation_space.shape, env.action_space.shape)

    for step in range(args.num_steps):
        steps = one_episode(env, policy)


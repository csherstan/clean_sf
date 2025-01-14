import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

def make_env(env_id, idx, capture_video, run_name, gamma):
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
        # TODO: this is not good practice - gamma is part of the solution, not the problem
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # TODO: also not good to clip at this level.
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def evaluate_agent(
    agent, #state_dict,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
):
    print(f"evaluate_agent ({env_id})")
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name, gamma=0.99)])
    # agent = Model(obs_space_shape = envs.single_observation_space.shape, action_space_shape = envs.single_action_space.shape).to(device)
    # agent.load_state_dict(state_dict)
    # agent.eval()
    # agent.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_lengths = []
    while len(episodic_returns) < eval_episodes:
      actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
      next_obs, reward, terminated, truncated, infos = envs.step(actions.cpu().detach().numpy())
      if "final_info" in infos:
        for info in infos["final_info"]:
            if "episode" not in info:
                continue
            print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}, episodic_length={info['episode']['l']}")
            episodic_returns += [info["episode"]["r"]]
            episodic_lengths += [info["episode"]["l"]]
      obs = next_obs

    # agent.train()

    return episodic_returns, episodic_lengths

def eval_agent(agent: torch.nn.Module,
         env_id: str,
         eval_episodes: int,
         run_name: str,
         Model: torch.nn.Module,
         global_step: int,
         device: torch.device,
         writer):
  episode_returns, episode_lengths = evaluate_agent(agent, #state_dict=agent.state_dict(),
                make_env=make_env,
                env_id=env_id,
                eval_episodes=eval_episodes,
                run_name=run_name,
                Model=Model,
                device=device,
                capture_video=False,
                )
  writer.add_scalar("eval/episodic_return", np.array(episode_returns).mean(), global_step)
  writer.add_scalar("eval/episodic_lengths", np.array(episode_lengths).mean(), global_step)


def save_model(run_name: str, global_step: int, exp_name: str, agent) -> str:
  model_path = f"runs/{run_name}/{exp_name}_{global_step}.cleanrl_model"
  torch.save(agent.state_dict(), model_path)
  print(f"model saved to {model_path}")

  return model_path
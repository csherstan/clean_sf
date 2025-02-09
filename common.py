import dataclasses
import random
from enum import Enum
from pathlib import Path
from typing import Callable, AnyStr, Any, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

AgentGenerator = Callable


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


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


@dataclasses.dataclass
class ActionAndValue:
    action: Any
    logprob: Any
    entropy: Any
    value: Any


def eval_agent_direct(
    agent: torch.nn.Module,
    envs: gym.vector.SyncVectorEnv,
    eval_episodes: int,
    greedy: bool = True,
    device: torch.device = torch.device("cpu"),
):
    """
    Give an agent directly for evaluation. Does NOT set evaluation mode.

    :param agent:
    :param envs:
    :param eval_episodes:
    :param device:
    :return:
    """

    obs, _ = envs.reset()
    episodic_returns = []
    episodic_lengths = []
    while len(episodic_returns) < eval_episodes:
        actions = agent.get_action_and_value(torch.Tensor(obs).to(device), greedy=greedy).action
        next_obs, reward, terminated, truncated, infos = envs.step(actions.cpu().detach().numpy())
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, "
                      f"episodic_return={info['episode']['r']}, "
                      f"episodic_length={info['episode']['l']}")
                episodic_returns += [info["episode"]["r"]]
                episodic_lengths += [info["episode"]["l"]]
        obs = next_obs

    return [ret[0] for ret in episodic_returns], [l[0] for l in episodic_lengths]


# def eval_from_path(args: Any, agent: nn.Module):
#     model_dict = torch.load(args.load_model_path)
#     envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, False, "eval", gamma=1.0)])
#     agent: nn.Module = agent_generator(envs)
#     agent.load_state_dict(model_dict)
#
#     eval_returns, eval_lengths = eval_agent_direct(agent, envs, args.eval_episodes, greedy=True,
#                                                    device=torch.device("cpu"))
#     print(f"avg return {np.array(eval_returns).mean()}, avg_length {np.array(eval_lengths).mean()}")


def eval_agent_with_logging(agent: torch.nn.Module,
                            env_id: str,
                            eval_episodes: int,
                            run_name: str,
                            global_step: int,
                            device: torch.device,
                            writer):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, run_name, gamma=1.0)])
    episode_returns, episode_lengths = eval_agent_direct(agent=agent,
                                                         envs=envs,
                                                         eval_episodes=eval_episodes,
                                                         greedy=True,
                                                         device=device)
    writer.add_scalar("eval/episodic_return", np.array(episode_returns).mean(), global_step)
    writer.add_scalar("eval/episodic_lengths", np.array(episode_lengths).mean(), global_step)


def save_model(run_name: str, global_step: int, exp_name: str, agent) -> str:
    model_path = f"runs/{run_name}/{exp_name}_{global_step}.cleanrl_model"
    torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")

    return model_path


def gen_seed() -> int:
    return int(random.random() * np.iinfo(np.uint32).max)

def set_seed(seed: Optional[int], deterministic: bool) -> int:
    seed = seed if seed is not None else gen_seed()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    return seed
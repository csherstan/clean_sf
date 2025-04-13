import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from clean_sr.sr.agent import Agent
from clean_sr.sr.args import Args
from clean_sr.common import make_env, save_model, eval_agent_with_logging, Mode, eval_agent_direct, set_seed


def agent_generator(envs, args: Args):
    return Agent(
        obs_space_shape=envs.single_observation_space.shape,
        action_space_shape=envs.single_action_space.shape,
        phi_size=args.phi_size,
    )

def training_step(
    args: Args,
    agent: Agent,
    obs: torch.Tensor,
    logprobs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    rewards: torch.Tensor,
    sf_targets: torch.Tensor,
    values: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    obs_space,
    action_space):
    """
    Let phi be the "features" of the critic network,
    the \psi(\phi_t) = E[\phi_t + \gamma \psi(\phi_{t+1})]
    \bar{r}_t = \phi_t^T w
    v_t = \psi_t^T w

    :param args:
    :param agent:
    :param obs:
    :param logprobs:
    :param actions:
    :param advantages:
    :param returns:
    :param rewards:
        We need the raw rewards to be able to learn the one-step expected reward
    :param sf_targets:
        An SF target is the bootstrapped return for the SF
    :param values:
    :param optimizer:
    :param obs_space:
    :param action_space:
    :return:
    """
    # flatten the batch
    b_obs = obs.reshape((-1,) + obs_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_rewards = rewards.reshape(-1)
    b_sf = sf_targets.reshape((-1, args.phi_size))

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            action_and_value = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            newlogprob, entropy, newvalue = action_and_value.logprob, action_and_value.entropy, action_and_value.value
            new_r, new_sf = action_and_value.sr_output.r, action_and_value.sr_output.sf
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            """
            For the SF training there are potentially 3 losses we can use:
            1. Loss for the SF
            2. Loss for the reward prediction
            3. Loss for the value prediction.
            """
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            # SR loss

            # reward prediction loss
            reward_loss = 0.5 * ((new_r - b_rewards[mb_inds]) ** 2).mean()

            # sf prediction loss
            sf_loss = 0.5 * ((new_sf - b_sf[mb_inds]) ** 2).mean()

            # ---------------

            entropy_loss = entropy.mean()

            loss = (pg_loss - args.ent_coef * entropy_loss) + v_loss * args.vf_coef + args.vf_coef * (
                reward_loss + sf_loss)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    return {
        "losses/value_loss": v_loss.item(),
        "losses/policy_loss": pg_loss.item(),
        "losses/entropy": entropy_loss.item(),
        "losses/old_approx_kl": old_approx_kl.item(),
        "losses/approx_kl": approx_kl.item(),
        "losses/clipfrac": np.mean(clipfracs),
        "losses/explained_variance": explained_var,
        "losses/reward": reward_loss.item(),
        "losses/sf": sf_loss.item(),
    }

def main(args: Args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(f"num_iterations={args.num_iterations}")

    seed = set_seed(args.seed, args.torch_deterministic)
    args.seed = seed

    run_name = f"{args.env_id}__{args.exp_name}__{seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = agent_generator(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    phis = torch.zeros((args.num_steps, args.num_envs, args.phi_size)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):

        if ((iteration - 1) % args.eval_interval) == 0:
            eval_agent_with_logging(agent=agent,
                       env_id=args.env_id,
                       eval_episodes=args.eval_episodes,
                       run_name=run_name,
                       global_step=global_step,
                       device=device,
                       writer=writer)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                result = agent.get_action_and_value(next_obs)
            actions[step] = result.action
            logprobs[step] = result.logprob
            values[step] = result.value.flatten()
            phis[step] = result.sr_output.phi

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(result.action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        writer.add_scalar("charts/mean_phi_mag", phis.abs().mean(), global_step)

        # bootstrap value if not done
        # In PPO the advantage is computed once before the epoch training loop rather than after each update step.
        # I guess something similar should be done for the SF based targets:
        #   - Compute the SF return once before the epoch training loop
        # SF delta = phi_t + gamma*SF(o_{t+1}) - SF(o_t)
        with torch.no_grad():
            critic_output = agent.get_critic(next_obs)
            next_value = critic_output.value.reshape(1, -1)
            next_sf = critic_output.sf

            advantages = torch.zeros_like(rewards).to(device)
            sf_targets = torch.zeros_like(phis).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                sf_targets[t] = phis[t] + args.gamma * nextnonterminal.unsqueeze(-1) * next_sf
            returns = advantages + values

        step_results = training_step(
            args=args,
            agent=agent,
            obs=obs,
            logprobs=logprobs,
            actions=actions,
            advantages=advantages,
            returns=returns,
            rewards=rewards,
            sf_targets=sf_targets,
            values=values,
            optimizer=optimizer,
            obs_space=envs.single_observation_space,
            action_space=envs.single_action_space,
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        for key, val in step_results.items():
            writer.add_scalar(key, val, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        save_model(run_name, global_step, args.exp_name, agent)

    eval_agent_with_logging(agent=agent,
                            env_id=args.env_id,
                            eval_episodes=args.eval_episodes,
                            run_name=run_name,
                            global_step=global_step,
                            device=device,
                            writer=writer)

    # if args.upload_model:
    #     from cleanrl_utils.huggingface import push_to_hub
    #
    #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
    #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
    #     push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.mode == Mode.TRAIN:
        main(args)
    else:
        set_seed(args.seed, args.torch_deterministic)

        model_dict = torch.load(args.load_model_path)
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, True, "eval", gamma=1.0)])
        agent: nn.Module = agent_generator(envs)
        agent.load_state_dict(model_dict)

        eval_returns, eval_lengths = eval_agent_direct(agent, envs, args.eval_episodes, greedy=False,
                                                       device=torch.device("cpu"))
        print(f"avg return {np.array(eval_returns).mean()}, avg_length {np.array(eval_lengths).mean()}")

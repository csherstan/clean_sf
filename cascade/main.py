import random
import time
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cascade.agent import Agent
from cascade.args import Args
from common import make_env, Mode, set_seed, eval_agent_direct, eval_agent_with_logging, \
    save_model


def agent_generator(envs, args: Args):
    return Agent(obs_space_shape=envs.single_observation_space.shape, action_space_shape=envs.single_action_space.shape, num_gammas=len(args.gammas))

def main(args: Args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(f"num_iterations={args.num_iterations}")

    seed = set_seed(args.seed, args.torch_deterministic)

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
    cascade_final_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    cascade_values = torch.zeros((args.num_steps, args.num_envs, len(args.gammas))).to(device)
    gammas_tensor = torch.tensor(args.gammas).to(device)
    num_gammas = len(args.gammas)

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
                values[step] = result.value.flatten()
                cascade_final_values[step] = result.cascade_final_value.flatten()
                cascade_values[step] = result.cascade_values
            actions[step] = result.action
            logprobs[step] = result.logprob

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

        # bootstrap value if not done
        with (torch.no_grad()):
            val_result = agent.get_value(next_obs)
            next_value, next_cascade_final_value, next_cascade_values = val_result.value.reshape(1, -1), \
                val_result.cascade_final_value, val_result.cascade_values
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            cascade_advantages = torch.zeros_like(cascade_values).to(device)
            cascade_lastgaelam = torch.zeros((cascade_values.shape[1:])).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                    nextcascade_values = next_cascade_values
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                    nextcascade_values = cascade_values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                cascade_delta = rewards[t].unsqueeze(-1) + args.gamma*nextcascade_values*nextnonterminal.unsqueeze(-1) - cascade_values[t]
                cascade_advantages[t] = cascade_lastgaelam = cascade_delta + args.gamma*args.gae_lambda * nextnonterminal.unsqueeze(-1)*cascade_lastgaelam

            returns = advantages + values
            cascade_returns = cascade_advantages + cascade_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_value_heads = cascade_values.reshape(-1, num_gammas)
        b_cascade_returns = cascade_returns.reshape(-1, num_gammas)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                result = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                newlogprob = result.logprob
                entropy = result.entropy
                newvalue = result.value
                newcascade_values = result.cascade_values
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
                newvalue = newvalue.view(-1)
                newcascade_values = newcascade_values.view(-1, num_gammas)
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

                    cascade_v_loss_unclipped = (newcascade_values - b_cascade_returns[mb_inds])**2
                    cascade_v_clipped = b_value_heads[mb_inds] + torch.clamp(
                        newcascade_values - b_value_heads[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )

                    cascade_v_loss_clipped = (cascade_v_clipped - b_cascade_returns[mb_inds]) ** 2
                    cascade_v_loss_max = torch.max(cascade_v_loss_unclipped, cascade_v_loss_clipped)
                    cascade_v_loss = 0.5 * cascade_v_loss_max.mean(0)
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    cascade_v_loss = 0.5 * ((newcascade_values - b_cascade_returns[mb_inds])**2).mean(0)

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + (v_loss.mean() + cascade_v_loss.mean()) * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        for i in range(num_gammas):
            writer.add_scalar(f"losses/value_loss_{args.gammas[i]}", cascade_v_loss[i].item(), global_step)
        writer.add_scalar(f"losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
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


if __name__=="__main__":
    args = tyro.cli(Args)
    if args.mode == Mode.TRAIN:
        main(args)
    else:
        set_seed(args.seed, args.torch_deterministic)

        model_dict = torch.load(args.load_model_path)
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, 0, False, "eval", gamma=1.0)])
        agent: nn.Module = agent_generator(envs)
        agent.load_state_dict(model_dict)

        eval_returns, eval_lengths = eval_agent_direct(agent, envs, args.eval_episodes, greedy=False,
                                                       device=torch.device("cpu"))
        print(f"avg return {np.array(eval_returns).mean()}, avg_length {np.array(eval_lengths).mean()}")

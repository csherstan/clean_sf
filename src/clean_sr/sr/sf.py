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

from baseline import Args
from clean_sr.common import layer_init, make_env, eval_agent


class SRCritic(nn.Module):
    def __init__(self, obs_space_shape, action_space_shape, phi_size: int):
        super().__init__()
        self.neck = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, phi_size)),
            nn.Tanh(),
        )

        self.head = nn.Sequential(
            layer_init(nn.Linear(phi_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, phi_size * 2))
        )

        self._phi_size = phi_size

    def forward(self, x):
        phi = self.neck(x)  # [batch, phi]
        output = self.head(phi)  # [batch, phi*2]
        sf = output[:, 0:self._phi_size]  # [batch, phi]
        w = output[:, self._phi_size:]  # [batch, phi]
        r = torch.einsum("bm,bm->b", phi, w)[..., None]  # [batch, 1]
        value = torch.einsum("bm,bm->b", sf, w)[..., None]  # [batch, 1]

        return {"phi": phi, "sf": sf, "w": w, "r": r, "value": value}


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

    def get_action_and_value_as_dict(self, x, action=None, greedy: bool = False):

        action_mean = self.actor_mean(x)

        if greedy:
            action = action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        # TODO: would like to use tensordict instead
        outputs = {"action": action, "logprob": probs.log_prob(action).sum(1), "entropy": probs.entropy().sum(1)}
        outputs.update(self.critic(x))

        return outputs

    def get_action_and_value(self, x, action=None, greedy: bool = False):
        action_mean = self.actor_mean(x)

        if greedy:
            action = action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()

        # TODO: would like to use tensordict instead
        outputs = {"action": action, "logprob": probs.log_prob(action).sum(1), "entropy": probs.entropy().sum(1)}
        outputs.update(self.critic(x))

        return outputs["action"], outputs["logprob"], outputs["entropy"], outputs["value"]


@dataclass
class SFArgs(Args):
    phi_size: int = 64


def sf_train_loop(args: SFArgs):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(f"num_iterations={args.num_iterations}")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(obs_space_shape=envs.single_observation_space.shape,
                  action_space_shape=envs.single_action_space.shape, phi_size=args.phi_size).to(device)
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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):

        if ((iteration - 1) % args.eval_interval) == 0:
            eval_agent(agent=agent,
                       env_id=args.env_id,
                       eval_episodes=10,
                       run_name=run_name,
                       Model=Agent,
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
                outputs = agent.get_action_and_value_as_dict(next_obs)
            actions[step] = outputs["action"]
            logprobs[step] = outputs["logprob"]
            values[step] = outputs["value"].flatten()
            phis[step] = outputs["phi"]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(outputs["action"].cpu().numpy())
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
            next_value = critic_output["value"].reshape(1, -1)
            next_phi = critic_output["phi"]
            next_sf = critic_output["sf"]

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

                sf_targets[t] = phis[t] + args.gamma * nextnonterminal * next_sf  # TODO
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
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

                outputs = agent.get_action_and_value_as_dict(b_obs[mb_inds], b_actions[mb_inds])
                newlogprob = outputs["logprob"]
                entropy = outputs["entropy"]
                newvalue = outputs["value"]
                new_r = outputs["r"]
                new_sf = outputs["sf"]

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

                # TODO: clean this up
                loss = 0.0 * (pg_loss - args.ent_coef * entropy_loss) + v_loss * args.vf_coef + args.vf_coef * (
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

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/reward", reward_loss, global_step)
        writer.add_scalar("losses/sf", sf_loss, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    eval_agent(agent=agent,
               env_id=args.env_id,
               eval_episodes=10,
               run_name=run_name,
               Model=Agent,
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


sf_train_loop(SFArgs("sf"))

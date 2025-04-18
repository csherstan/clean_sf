import matplotlib.pyplot as plt

from clean_sr.sr.agent import SRCritic, StandardCritic
import torch
import gymnasium as gym
import tqdm

def normalize(x: torch.Tensor) -> torch.Tensor:
    mean, var = torch.mean(x, dim=0), torch.var(x, dim=0)
    return (x - mean) / torch.sqrt(var + 1e-8)

def compute_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute the correlation between two tensors.
    :param x: Shape [batch, num_envs]
    :param y: Shape [batch, num_envs]
    :return:
    """
    assert x.shape == y.shape
    x = normalize(x.flatten())
    y = normalize(y.flatten())
    return (x*y).mean()

def compute_standard_returns(rewards: torch.Tensor,
                    dones: torch.Tensor,
                    next_done: torch.Tensor,
                    next_value,
                    gamma: float,
                    device) -> torch.Tensor:
    """
    :param rewards: Shape [batch, num_envs]
    :param dones:   Shape [batch, num_envs]
    :param next_done:   Shape [num_envs,]
    :param next_value: Shape [num_envs,]
    :param device:
    :return:
    """
    returns = torch.zeros_like(rewards, device=device)
    num_steps = rewards.shape[0]

    g_rewards = next_value*(1.-next_done)
    non_terminal = 1. - dones

    for t in reversed(range(num_steps)):
        g_rewards = rewards[t] + gamma * g_rewards * non_terminal[t]
        returns[t] = g_rewards

    return returns


def compute_sr_returns(rewards: torch.Tensor,
                    phis: torch.Tensor,
                    dones: torch.Tensor,
                    next_done: torch.Tensor,
                    next_value, next_sf: torch.Tensor,
                    gamma: float,
                    device) -> tuple[torch.Tensor, torch.Tensor]:
    """

    :param rewards: Shape [batch, num_envs]
    :param phis: features of the network. Shape [batch, num_envs, phi_dim]
    :param dones:   Shape [batch, num_envs]
    :param values: Shape [batch, num_envs]
    :param sf_predictions: successor feature predictions: E[phi + gamma*SF(t+1)]. Shape [batch, num_envs, phi_dim]
    :param next_done:   Shape [num_envs,]
    :param next_value: Shape [num_envs,]
    :param next_sf: Shape [num_envs, phi_dim]
    :param device:
    :return:
    """
    returns = torch.zeros_like(rewards, device=device)
    sf_targets = torch.zeros_like(phis, device=device)
    num_steps = rewards.shape[0]

    g_rewards = next_value*(1.-next_done)
    g_sf = next_sf*(1.-next_done)

    non_terminal = 1. - dones

    for t in reversed(range(num_steps)):
        g_rewards = rewards[t] + gamma * g_rewards * non_terminal[t]
        g_sf = phis[t] + gamma * g_sf * non_terminal[t]
        returns[t] = g_rewards
        sf_targets[t] = g_sf

    return returns, sf_targets

if __name__=="__main__":

    gamma = 0.8
    gae_lambda = 0.95
    phi_size = 10
    value_weight = 0.5
    reward_weight = 1.0
    sf_weight = 1.0

    device = torch.device("cpu")
    # Arrange: Set up the necessary arguments, mock objects, and inputs
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)

    standard_critic = StandardCritic(obs_space_shape=obs_space.shape, action_space_shape=action_space.shape)

    sr_critic = SRCritic(
        obs_space_shape=obs_space.shape,
        action_space_shape=action_space.shape,
        phi_size=phi_size,
    )


    # prep data -------------------
    num_steps = 200
    obs = torch.arange(0, end=num_steps, dtype=torch.float32).to(device) / num_steps
    obs = obs.view(-1, 1).detach()
    rewards = torch.zeros((num_steps, 1), device=device).detach()
    rewards[100:110] = 1.0
    # rewards = torch.zeros((num_steps, 1)).to(device).detach()
    # rewards[int(num_steps / 2):] = 1.0


    dones = torch.zeros((num_steps, 1)).to(device).detach()

    true_returns = []
    g = 0
    for j in range(10):
        for r in reversed(rewards.squeeze().detach().numpy()):
            g = r.item() + gamma * g
            true_returns.append(g)
    true_returns = torch.tensor(list(reversed(true_returns)))[0:num_steps]

    # end prep data -------------------

    standard_optimizer = torch.optim.Adam(standard_critic.parameters(), lr=1e-4)
    sr_optimizer = torch.optim.Adam(sr_critic.parameters(), lr=1e-4)

    step_results = []

    def plot_predictions(step_count: int) -> None:
        with torch.no_grad():
            sr_result = sr_critic.forward(obs)
            standard_result = standard_critic.forward(obs)

        values = sr_result.value

        plt.gca().clear()
        plt.plot(values.detach().numpy() * (1 - gamma), label="predicted (SR)")
        plt.plot(standard_result.detach().numpy() * (1 - gamma), label="predicted (standard)")
        plt.plot(torch.Tensor(true_returns).detach().numpy() * (1 - gamma), label="true")
        plt.plot(rewards.detach().numpy(), label="reward")
        plt.title(f"predicted vs reward: {step_count}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_predictions(0)

    for epoch in tqdm(range(20000)):
        step_result = {}

        #-------- SR training
        result = sr_critic.forward(obs)
        values = result.value
        phis = result.phi
        sf_predictions = result.sf
        predicted_rewards = result.r

        step_result["sf_correlation"] = compute_correlation(values, true_returns.unsqueeze(-1)).item()
        step_result["sf_MSE"] = torch.nn.functional.mse_loss(values, true_returns.unsqueeze(-1)).item()

        returns, sf_targets = compute_sr_returns(
            rewards=rewards,
            phis=phis,
            dones=dones,
            next_done=dones[0],
            next_value=values[0].detach(),
            next_sf=sf_predictions[0].detach(),
            gamma=gamma,
            device=device
        )

        sf_loss = (0.5*(sf_predictions - sf_targets.detach())**2).mean()
        value_loss = (0.5*(values - returns.detach())**2).mean()
        reward_loss = (0.5*(predicted_rewards - rewards)**2).mean()
        loss = sf_loss*sf_weight + value_loss*value_weight + reward_loss*reward_weight
        sr_optimizer.zero_grad()
        loss.backward()
        sr_optimizer.step()

        step_result.update({
            "sf_loss": sf_loss.item(),
            "sf_value_loss": value_loss.item(),
            "sf_reward_loss": reward_loss.item(),
        })


        #-------- Standard training
        values = standard_critic(obs)
        step_result["standard_correlation"] = compute_correlation(values, true_returns.unsqueeze(-1)).item()
        step_result["standard_MSE"] = torch.nn.functional.mse_loss(values, true_returns.unsqueeze(-1)).item()

        returns = compute_standard_returns(
            rewards=rewards,
            dones=dones,
            next_done=dones[0],
            next_value=values[0].detach(),
            gamma=gamma,
            device=device
        )

        value_loss = (0.5*(values - returns.detach())**2).mean()
        standard_optimizer.zero_grad()
        value_loss.backward()
        standard_optimizer.step()

        step_result["standard_value_loss"] = value_loss.item()

        step_results.append(step_result)


    results_dict = {}
    for step_result in step_results:
        for key, val in step_result.items():
            results_dict.setdefault(key, []).append(val)

    plt.figure()
    for key, val in results_dict.items():
        plt.gca().clear()
        plt.plot(val, label=key)
        plt.title(key)
        plt.tight_layout()
        plt.show()

    plt.gca().clear()
    plt.plot(results_dict["sf_correlation"], label="sf correlation")
    plt.plot(results_dict["standard_correlation"], label="standard correlation")
    plt.title("correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.gca().clear()
    plt.plot(results_dict["sf_MSE"], label="sf MSE")
    plt.plot(results_dict["standard_MSE"], label="standard MSE")
    plt.title("Value MSE")
    plt.legend()
    plt.tight_layout()
    plt.show()


    plot_predictions(epoch)






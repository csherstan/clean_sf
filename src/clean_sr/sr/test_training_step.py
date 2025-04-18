import unittest

import gym
import torch

from clean_sr.sr.agent import Agent
from clean_sr.sr.args import Args
from clean_sr.sr.main import training_step, compute_returns

import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def test_training_step_sr(self):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        # Arrange: Set up the necessary arguments, mock objects, and inputs
        obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)

        gamma = 0.5
        gae_lambda = 0.5
        phi_size = 10

        num_steps = 100
        obs = torch.arange(0, end=num_steps, dtype=torch.float32).to(device) / num_steps
        obs = obs.view(-1, 1, 1)
        rewards = torch.zeros((num_steps, 1)).to(device)
        rewards[int(num_steps / 2):] = 1.0
        dones = torch.zeros((num_steps, 1)).to(device)
        logprobs = torch.full((num_steps, 1), -1.0).to(device)
        actions = torch.zeros((num_steps, 1)).to(device)

        args = Args(exp_name="test", phi_size=phi_size, batch_size=num_steps, minibatch_size=int(num_steps // 10),
                    num_minibatches=10,
                    policy_coef=0.0,
        )

        agent = Agent(obs_space_shape=obs_space.shape,
                      action_space_shape=action_space.shape,
                      phi_size=phi_size).to(device)

        optimizer = torch.optim.Adam(agent.critic.parameters(), lr=1e-3)

        step_results = []

        for epoch in range(1000):

            with torch.no_grad():
                result = agent.get_action_and_value(obs.reshape(-1, 1))

            phis = result.sr_output.phi
            values = result.value
            sf_predictions = result.sr_output.sf

            next_done = torch.zeros((1,)).to(device)
            next_value = values[0]
            next_sf = result.sr_output.sf[0]

            returns, advantages, sf_targets = compute_returns(
                rewards=rewards,
                phis=phis,
                dones=dones,
                values=values,
                sf_predictions=sf_predictions,
                next_done=next_done,
                next_value=next_value,
                next_sf=next_sf,
                gamma=gamma,
                gae_lambda=gae_lambda,
                num_steps=num_steps,
                device=device
            )

            # Act: Call the training_step function
            result = training_step(
                args=args,
                agent=agent,
                obs=obs,
                logprobs=logprobs,
                actions=actions,
                advantages=advantages,
                returns=returns,
                rewards=rewards,
                sf_targets=sf_targets,
                sf_predictions=sf_predictions,
                values=values,
                optimizer=optimizer,
                obs_space=obs_space,
                action_space=action_space,
            )

            step_results.append(result)

            # Assert: Verify the results
            self.assertIsNotNone(result)  # Example assertion
            # Add more assertions to validate the expected behavior
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



if __name__ == '__main__':
    unittest.main()

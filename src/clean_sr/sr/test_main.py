import unittest
import torch
from src.clean_sr.sr.main import compute_returns


class TestComputeReturns(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.num_steps = 5
        self.num_envs = 2
        self.phi_dim = 3
        self.gamma = 0.5
        self.gae_lambda = 0.95

        # Mock arguments
        global args
        args = type("Args", (), {})()
        args.gamma = self.gamma
        args.gae_lambda = self.gae_lambda
        args.num_steps = self.num_steps

    # def test_all_zeros(self):
    #     rewards = torch.zeros((self.num_steps, self.num_envs, 1))
    #     phis = torch.zeros((self.num_steps, self.num_envs, self.phi_dim))
    #     dones = torch.zeros((self.num_steps, self.num_envs, 1))
    #     values = torch.zeros((self.num_steps, self.num_envs, 1))
    #     sf_predictions = torch.zeros((self.num_steps, self.num_envs, self.phi_dim))
    #     next_done = torch.zeros((self.num_envs, 1))
    #     next_value = torch.zeros((self.num_envs, 1))
    #     next_sf = torch.zeros((self.num_envs, self.phi_dim))
    #
    #     returns, advantages, sf_targets = compute_returns(
    #         rewards, phis, dones, values, sf_predictions, next_done, next_value, next_sf, self.device
    #     )
    #
    #     self.assertTrue(torch.allclose(returns, torch.zeros_like(returns)))
    #     self.assertTrue(torch.allclose(advantages, torch.zeros_like(advantages)))
    #     self.assertTrue(torch.allclose(sf_targets, torch.zeros_like(sf_targets)))
    #
    # def test_terminal_state(self):
    #     rewards = torch.tensor([[[1.0]], [[1.0]], [[1.0]], [[1.0]], [[1.0]]]).repeat(1, self.num_envs, 1)
    #     phis = torch.zeros((self.num_steps, self.num_envs, self.phi_dim))
    #     dones = torch.tensor([[[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]]]).repeat(1, self.num_envs, 1)
    #     values = torch.zeros((self.num_steps, self.num_envs, 1))
    #     sf_predictions = torch.zeros((self.num_steps, self.num_envs, self.phi_dim))
    #     next_done = torch.zeros((self.num_envs, 1))
    #     next_value = torch.zeros((self.num_envs, 1))
    #     next_sf = torch.zeros((self.num_envs, self.phi_dim))
    #
    #     returns, advantages, sf_targets = compute_returns(
    #         rewards, phis, dones, values, sf_predictions, next_done, next_value, next_sf, self.device
    #     )
    #
    #     expected_returns = torch.tensor([[[2.97]], [[1.99]], [[1.0]], [[1.99]], [[1.0]]]).repeat(1, self.num_envs, 1)
    #     self.assertTrue(torch.allclose(returns, expected_returns, atol=1e-2))

    def test_sf_targets(self):
        rewards = torch.zeros((self.num_steps, self.num_envs))
        phis = torch.tensor([
            [[1., 0, 0], [0, 1, 0]],
            [[1., 0, 0], [0, 1, 0]],
            [[1., 0, 0], [0, 1, 0]],
            [[1., 0, 0], [0, 1, 0]],
            [[1., 0, 0], [0, 1, 0]],
        ])
        dones = torch.tensor([
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
        ])
        values = torch.zeros((self.num_steps, self.num_envs))
        sf_predictions = torch.tensor([
            [[0., 0, 0.], [0, 0., 0, ]],
            [[1., 0, 1], [0, 1, 1]],
            [[1., 0, 1], [0, 1, 1]],
            [[1., 0, 1], [0, 1, 1]],
            [[1., 0, 1], [0, 1, 0.]],
        ])
        next_done = torch.zeros((self.num_envs,))
        next_value = torch.zeros((self.num_envs,))
        next_sf = torch.tensor([[0., 1., 0.], [1., 1., 1.]])

        returns, advantages, sf_targets = compute_returns(
            rewards=rewards,
            phis=phis,
            dones=dones,
            values=values,
            sf_predictions=sf_predictions,
            next_done=next_done,
            next_value=next_value,
            next_sf=next_sf,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            num_steps=self.num_steps,
            device=self.device
        )

        expected_sf_targets = torch.tensor([
            [[1.5, 0, 0.5], [0, 1.5, 0.5]],
            [[1.5, 0, 0.5], [0, 1., 0.]],
            [[1.5, 0, 0.5], [0, 1.5, 0.5]],
            [[1.5, 0, 0.5], [0, 1.5, 0.0]],
            [[1., .5, 0.], [.5, 1.5, .5]],
        ])
        self.assertTrue(torch.allclose(sf_targets, expected_sf_targets, atol=1e-3))


if __name__ == "__main__":
    unittest.main()

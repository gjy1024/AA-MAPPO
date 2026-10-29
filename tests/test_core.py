"""Scientific correctness regression checks; run python -m unittest discover -s tests."""
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from aamappo.config import Config
from aamappo.environment import SwarmEnv
from aamappo.models import make_networks, transformed_log_prob
from aamappo.ppo import generalized_advantage, ppo_objective, train, load_checkpoint, evaluate


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(hidden_size=16, horizon=8, rollout_steps=8, ppo_epochs=1, minibatch_size=8)

    def test_independent_seeded_generators(self):
        a, b = SwarmEnv(self.config, 8), SwarmEnv(self.config, 8)
        a.reset(); b.reset()
        noise = SwarmEnv(self.config, 999); noise.reset()
        for _ in range(5):
            noise.step(noise.rule_actions())
            obs_a = a.step(np.zeros((3, 2)))[0]
            obs_b = b.step(np.zeros((3, 2)))[0]
            np.testing.assert_array_equal(obs_a[0], obs_b[0])

    def test_acceleration_bound_and_consistent_integration(self):
        env = SwarmEnv(self.config); env.reset()
        position = env.positions.copy()
        env.step(np.full((3, 2), 100.0))
        self.assertLessEqual(np.linalg.norm(env.velocities / self.config.dt, axis=-1).max(),
                             self.config.max_acceleration + 1e-10)
        np.testing.assert_allclose(env.positions-position, .5*env.velocities*self.config.dt)
        for _ in range(100):
            env.step(np.full((3, 2), 100.0))
        self.assertLessEqual(np.linalg.norm(env.velocities, axis=-1).max(), self.config.max_speed+1e-10)

    def test_coincident_user_is_finite_and_noise_is_not_rf(self):
        env = SwarmEnv(self.config); env.reset()
        env.positions[0] = env.users[0]
        first = env.metrics()
        env._refresh_noise()
        second = env.metrics()
        self.assertTrue(all(math.isfinite(value) for value in first.values()))
        self.assertEqual(first, second)

    def test_time_limit_is_truncation(self):
        env = SwarmEnv(self.config); env.reset()
        for _ in range(self.config.horizon):
            _, _, terminated, truncated, _ = env.step(np.zeros((3, 2)))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_config_rejects_unknown_and_invalid(self):
        with self.assertRaises(ValueError): Config.from_dict({"typo": 5})
        with self.assertRaises(ValueError): replace(self.config, height=0).validate()
        with self.assertRaises(ValueError): replace(self.config, horizon=True).validate()
        with self.assertRaises(ValueError): replace(self.config, power_tx="ten").validate()
        with self.assertRaises(ValueError): replace(self.config, seed=-1).validate()


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.config = Config(hidden_size=16, horizon=8, rollout_steps=8, ppo_epochs=1, minibatch_size=8)
        self.env = SwarmEnv(self.config)
        self.local, self.global_tokens = self.env.reset()

    def test_multiple_entity_attention_and_gradients(self):
        actor, critic = make_networks(self.config, self.env)
        local = torch.from_numpy(self.local)
        weights = actor.attention_weights(local)
        self.assertGreater(weights.shape[-1], 1)
        torch.testing.assert_close(weights.sum(-1), torch.ones(3))
        changed = local.clone(); changed[:, 2, :2] += 3
        self.assertFalse(torch.allclose(weights, actor.attention_weights(changed)))
        actor.mean(actor.encoder(local)).square().sum().backward()
        for gradient in actor.encoder.attention.in_proj_weight.grad.chunk(3, dim=0):
            self.assertGreater(gradient.abs().sum().item(), 0)
        critic(torch.from_numpy(self.global_tokens).unsqueeze(0)).square().sum().backward()
        for gradient in critic.encoder.attention.in_proj_weight.grad.chunk(3, dim=0):
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_single_token_regression(self):
        layer = torch.nn.MultiheadAttention(16, 4, batch_first=True)
        single = torch.randn(3, 1, 16)
        _, weights = layer(single, single, single, average_attn_weights=False)
        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_tanh_log_density_matches_manual_and_is_finite(self):
        dist = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
        raw = torch.tensor([.3, -.7])
        expected = (dist.log_prob(raw)-torch.log(1-raw.tanh().square())).sum()
        torch.testing.assert_close(transformed_log_prob(dist, raw), expected)
        self.assertTrue(torch.isfinite(transformed_log_prob(dist, torch.tensor([50., -50.]))))
        actor, _ = make_networks(self.config, self.env)
        action, raw, old = actor.sample(torch.from_numpy(self.local))
        new, _ = actor.evaluate_actions(torch.from_numpy(self.local), raw)
        torch.testing.assert_close(old, new)
        self.assertLess(action.abs().max().item(), 1)

    def test_gae_bootstrap_and_reset_boundary(self):
        adv, targets = generalized_advantage([1, 2], [3, 4], [4, 5], [False, False],
                                             [True, True], .9, .95)
        np.testing.assert_allclose(adv, [1+.9*4-3, 2+.9*5-4])
        terminal, _ = generalized_advantage([1], [3], [100], [True], [True], .9, .95)
        np.testing.assert_allclose(terminal, [-2])
        consecutive, _ = generalized_advantage([1, 1], [0, 0], [0, 0], [False, True],
                                               [False, True], 1, 1)
        np.testing.assert_allclose(consecutive, [2, 1])

    def test_replay_attention_and_initial_safety_are_aligned(self):
        actor, _ = make_networks(self.config, self.env)
        result = evaluate(self.config, actor, seeds=[73])
        env = SwarmEnv(self.config, seed=73)
        local, _ = env.reset()
        actor.eval()
        with torch.no_grad():
            expected = actor.attention_weights(torch.from_numpy(local)).numpy()
        np.testing.assert_allclose(result["trajectory"][0]["attention"], expected, rtol=1e-5, atol=1e-7)
        observed_minimum = min(frame["metrics"]["min_distance"] for frame in result["trajectory"])
        self.assertEqual(result["episodes"][0]["min_distance"], observed_minimum)

    def test_clip_sign_and_true_no_clip(self):
        ratios = torch.tensor([1.5, .5, .5, 1.5])
        advantages = torch.tensor([1., -1., 1., -1.])
        torch.testing.assert_close(ppo_objective(ratios, advantages, .2), torch.tensor([1.2, -.8, .5, -1.5]))
        torch.testing.assert_close(ppo_objective(ratios, advantages, .2, True), ratios*advantages)

    def test_four_ablation_modes(self):
        for mode in ("none", "actor", "critic", "both"):
            actor, critic = make_networks(replace(self.config, attention_mode=mode), self.env)
            self.assertEqual(actor.encoder.uses_attention, mode in ("actor", "both"))
            self.assertEqual(critic.encoder.uses_attention, mode in ("critic", "both"))
            self.assertEqual(actor.sample(torch.from_numpy(self.local))[0].shape, (3, 2))

    def test_short_training_checkpoint_and_outputs(self):
        temporary_root = Path.cwd() / ".test_tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temporary_root)) as temporary:
            out = Path(temporary) / "smoke"
            result = train(self.config, out, steps=16)
            self.assertEqual(result["actual_environment_steps"], 16)
            for name in ("config.json", "provenance.json", "metrics.jsonl", "checkpoint.pt", "evaluation.json"):
                self.assertTrue((out/name).is_file())
            config, actor, critic, steps = load_checkpoint(out/"checkpoint.pt")
            self.assertEqual(steps, 16)
            self.assertEqual(config.attention_mode, "both")
            expected = evaluate(config, actor, seeds=[19], trained_steps=steps)
            _, reloaded_actor, _, _ = load_checkpoint(out/"checkpoint.pt")
            actual = evaluate(config, reloaded_actor, seeds=[19], trained_steps=steps)
            self.assertEqual(expected["trajectory"], actual["trajectory"])
            with self.assertRaises(ValueError): train(self.config, out, steps=16)


if __name__ == "__main__":
    unittest.main()

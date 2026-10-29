"""Checks for the explicitly separate demonstration-assisted extension."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from aamappo.config import Config
from aamappo.environment import SwarmEnv
from aamappo.models import make_networks
from aamappo.ppo import train, load_checkpoint

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from warmstart_pilot import LocalRingTeacher


class WarmstartTests(unittest.TestCase):
    def test_teacher_uses_local_entities_and_ignores_nuisance(self):
        config = Config()
        env = SwarmEnv(config)
        local, _ = env.reset()
        teacher = LocalRingTeacher(config)
        action = teacher.actions(local)
        changed = local.copy()
        changed[:, -config.noise_dim:, :4] = 20
        np.testing.assert_array_equal(action, teacher.actions(changed))
        self.assertLessEqual(np.linalg.norm(action, axis=1).max(), 1+1e-6)

    def test_warm_actor_initialization_preserves_zero_step_policy_before_update(self):
        config = Config(hidden_size=16, horizon=4, rollout_steps=4, ppo_epochs=1, minibatch_size=4,
                        learning_rate=1e-12, entropy_coefficient=0)
        torch.manual_seed(42)
        actor, _ = make_networks(config, SwarmEnv(config))
        initial = {key: value.clone() for key, value in actor.state_dict().items()}
        temporary_root = Path.cwd()/".test_tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temporary_root)) as temporary:
            result = train(config, Path(temporary)/"initialized", 4,
                           initial_actor_state=initial,
                           initialization_provenance={"kind": "test-initializer", "samples": 4})
            self.assertEqual(result["initialization"]["kind"], "test-initializer")
            _, loaded, _, _ = load_checkpoint(Path(temporary)/"initialized/checkpoint.pt")
            for name, value in initial.items():
                torch.testing.assert_close(value, loaded.state_dict()[name], rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

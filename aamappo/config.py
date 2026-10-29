"""Explicit experiment settings and their provenance."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


PAPER_PARAMETERS = {
    "num_agents", "num_users", "map_size", "safe_distance", "max_speed",
    "learning_rate", "gamma", "clip_epsilon", "num_heads", "noise_dim",
    "train_steps",
    "collision_penalty", "speed_coefficient",
}


@dataclass
class Config:
    num_agents: int = 3
    num_users: int = 2
    map_size: float = 100.0
    safe_distance: float = 2.0
    max_speed: float = 5.0
    learning_rate: float = 5e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    num_heads: int = 4
    noise_dim: int = 4
    train_steps: int = 2000000
    # The following settings are explicit maintenance assumptions, NOT recovered
    # parameters of the published experiment.
    height: float = 10.0
    dt: float = 0.1
    max_acceleration: float = 2.0
    horizon: int = 200
    power_tx: float = 10.0
    noise_power: float = 0.1
    reference_gain: float = 1.0
    path_loss_exponent: float = 2.0
    sensing_gain: float = 1.0
    communication_weight: float = 1.0
    sensing_weight: float = 100.0
    collision_penalty: float = 50.0
    speed_coefficient: float = 0.01
    noise_amplitude: float = 1.0
    hidden_size: int = 64
    attention_mode: str = "both"
    reward_mode: str = "isac"
    no_clip: bool = False
    gae_lambda: float = 0.95
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 64
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    initial_log_std: float = -0.5
    seed: int = 7

    def validate(self):
        defaults = asdict(Config())
        for name, value in asdict(self).items():
            expected = type(defaults[name])
            if expected is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise ValueError(f"{name} must be numeric")
            if expected is int and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{name} must be an integer")
            if expected is bool and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in ("num_agents", "num_users", "horizon", "hidden_size", "num_heads",
                     "rollout_steps", "ppo_epochs", "minibatch_size", "train_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("map_size", "safe_distance", "max_speed", "height", "dt",
                     "max_acceleration", "power_tx", "noise_power", "reference_gain",
                     "path_loss_exponent", "learning_rate", "max_grad_norm"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not (0 < value < float("inf")):
                raise ValueError(f"{name} must be positive and finite")
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least two for the swarm benchmark")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.attention_mode not in ("none", "actor", "critic", "both"):
            raise ValueError("attention_mode must be none/actor/critic/both")
        if self.reward_mode not in ("isac", "naive"):
            raise ValueError("reward_mode must be isac/naive")
        if not isinstance(self.noise_dim, int) or isinstance(self.noise_dim, bool):
            raise ValueError("noise_dim must be an integer")
        if self.noise_dim < 0 or self.noise_amplitude < 0:
            raise ValueError("noise settings cannot be negative")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1]")
        for name in ("sensing_gain", "communication_weight", "sensing_weight", "collision_penalty",
                     "speed_coefficient", "entropy_coefficient", "value_coefficient"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if not 0 < self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in (0, 1); use no_clip for true ablation")
        return self

    def to_dict(self):
        result = asdict(self)
        defaults = asdict(Config())
        result["parameter_sources"] = {
            name: (("paper-reported" if result[name] == defaults[name]
                    else "maintenance-override-of-paper-value") if name in PAPER_PARAMETERS
                   else "maintenance-assumption")
            for name in asdict(self)
        }
        result["implementation"] = "maintained-reference-v0.2; not an exact historical reproduction"
        return result

    @classmethod
    def from_dict(cls, data):
        names = {field.name for field in fields(cls)}
        unknown = set(data) - names - {"parameter_sources", "implementation", "notes"}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**{key: value for key, value in data.items() if key in names}).validate()

    @classmethod
    def load(cls, path=None):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8"))) if path else cls().validate()


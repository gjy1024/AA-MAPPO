"""Small fixed-height ISAC reference environment; this is not a flight simulator."""
from __future__ import annotations

import numpy as np
from .config import Config


TOKEN_DIM = 8  # x, y, vx, vy, followed by four entity-type indicators


class SwarmEnv:
    def __init__(self, config: Config, seed=None):
        self.config = config.validate()
        self.rng = np.random.default_rng(config.seed if seed is None else seed)
        self.step_count = 0
        self.positions = np.zeros((config.num_agents, 2), dtype=np.float64)
        self.velocities = np.zeros_like(self.positions)
        self.users = np.zeros((config.num_users, 2), dtype=np.float64)
        self.nuisance = np.zeros((config.num_agents, config.noise_dim), dtype=np.float64)
        self.last_metrics = {}

    @property
    def actor_token_count(self):
        return self.config.num_agents + self.config.num_users + self.config.noise_dim

    @property
    def critic_token_count(self):
        return self.config.num_agents + self.config.num_users

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.config
        self.step_count = 0
        self.positions = self.rng.uniform(.1 * c.map_size, .9 * c.map_size, (c.num_agents, 2))
        self.velocities = np.zeros((c.num_agents, 2), dtype=np.float64)
        self.users = self.rng.uniform(.2 * c.map_size, .8 * c.map_size, (c.num_users, 2))
        self._refresh_noise()
        self.last_metrics = self.metrics()
        return self.observe()

    def _refresh_noise(self):
        c = self.config
        self.nuisance = self.rng.uniform(-c.noise_amplitude, c.noise_amplitude,
                                         (c.num_agents, c.noise_dim))

    @staticmethod
    def _token(xy, velocity, entity_type):
        token = np.zeros(TOKEN_DIM, dtype=np.float32)
        token[:2], token[2:4], token[4 + entity_type] = xy, velocity, 1.0
        return token

    def observe(self):
        """Local actor tokens use own velocity, teammate positions and users only."""
        c = self.config
        local = []
        for i in range(c.num_agents):
            tokens = [self._token(self.positions[i] / c.map_size,
                                   self.velocities[i] / c.max_speed, 0)]
            tokens += [self._token((self.positions[j] - self.positions[i]) / c.map_size,
                                    [0, 0], 1) for j in range(c.num_agents) if j != i]
            tokens += [self._token((user - self.positions[i]) / c.map_size, [0, 0], 2)
                       for user in self.users]
            tokens += [self._token([noise, 0], [0, 0], 3) for noise in self.nuisance[i]]
            local.append(tokens)
        global_tokens = [self._token(self.positions[i] / c.map_size,
                                      self.velocities[i] / c.max_speed, 0)
                         for i in range(c.num_agents)]
        global_tokens += [self._token(user / c.map_size, [0, 0], 2) for user in self.users]
        return np.asarray(local, dtype=np.float32), np.asarray(global_tokens, dtype=np.float32)

    def metrics(self, boundary_event=False):
        c = self.config
        horizontal_sq = ((self.positions[:, None] - self.users[None]) ** 2).sum(axis=-1)
        distance_sq = horizontal_sq + c.height ** 2
        gain = c.reference_gain / distance_sq ** (c.path_loss_exponent / 2)
        snr = c.power_tx * gain / c.noise_power
        # Sum over UAV-user links; no resource allocation or association model.
        communication = float(np.log2(1 + snr).sum())
        sensing_proxy = float((c.sensing_gain * c.power_tx / (c.noise_power * distance_sq ** 2)).sum())
        pairwise = np.linalg.norm(self.positions[:, None] - self.positions[None], axis=-1)
        distances = pairwise[np.triu_indices(c.num_agents, 1)]
        minimum = float(distances.min())
        collision_pairs = int((distances < c.safe_distance).sum())
        speed_cost = float(c.speed_coefficient * np.linalg.norm(self.velocities, axis=-1).sum())
        collision_cost = float(c.collision_penalty if collision_pairs else 0.0)
        physical_reward = (c.communication_weight * communication + c.sensing_weight * sensing_proxy
                           - speed_cost - collision_cost)
        nearest = np.sqrt(horizontal_sq).min(axis=1).mean()
        reward = -float(nearest / c.map_size) if c.reward_mode == "naive" else physical_reward
        return {"communication": communication, "sensing_proxy": sensing_proxy,
                "min_distance": minimum, "collision_pairs": collision_pairs,
                "collision_event": int(collision_pairs > 0), "boundary_event": int(boundary_event),
                "speed_cost": speed_cost, "collision_cost": collision_cost,
                "isac_objective": float(physical_reward), "reward": float(reward),
                "max_speed": float(np.linalg.norm(self.velocities, axis=-1).max())}

    def step(self, actions):
        c = self.config
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (c.num_agents, 2) or not np.isfinite(actions).all():
            raise ValueError(f"actions must be finite with shape {(c.num_agents, 2)}")
        normalized = np.clip(actions, -1, 1)
        normalized /= np.maximum(1.0, np.linalg.norm(normalized, axis=-1, keepdims=True))
        acceleration = normalized * c.max_acceleration
        # Define one consistent discrete update. Velocity cap limits effective
        # acceleration, and position uses average endpoint velocity.
        next_velocity = self.velocities + acceleration * c.dt
        next_velocity /= np.maximum(1.0, np.linalg.norm(next_velocity, axis=-1, keepdims=True) / c.max_speed)
        proposed = self.positions + .5 * (self.velocities + next_velocity) * c.dt
        boundary_event = bool(np.any((proposed < 0) | (proposed > c.map_size)))
        clipped = np.clip(proposed, 0, c.map_size)
        hit = clipped != proposed
        next_velocity[hit] = 0.0
        self.positions, self.velocities = clipped, next_velocity
        self.step_count += 1
        self._refresh_noise()
        self.last_metrics = self.metrics(boundary_event)
        # This continuing benchmark only truncates at the horizon. Collision is
        # measured and penalized, never presented as a hard safety guarantee.
        terminated = False
        truncated = self.step_count >= c.horizon
        return self.observe(), self.last_metrics["reward"], terminated, truncated, dict(self.last_metrics)

    def rule_actions(self):
        targets = self.users[np.arange(self.config.num_agents) % self.config.num_users]
        return np.tanh(.06 * (targets - self.positions) - .6 * self.velocities)

    def frame(self, attention=None):
        record = {"step": self.step_count, "time": self.step_count * self.config.dt,
                  "positions": self.positions.tolist(), "velocities": self.velocities.tolist(),
                  "metrics": dict(self.last_metrics)}
        if attention is not None:
            record["attention"] = attention
        return record


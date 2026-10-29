"""Correctness-oriented, single-environment shared-policy MAPPO training."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .environment import SwarmEnv
from .models import make_networks


def generalized_advantage(rewards, values, next_values, terminated, episode_ends, gamma, lam):
    """Bootstrap time limits, but never propagate GAE through a reset."""
    rewards, values, next_values = map(lambda x: np.asarray(x, dtype=np.float64),
                                       (rewards, values, next_values))
    advantages = np.zeros_like(rewards)
    accumulator = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - float(terminated[t])) * next_values[t] - values[t]
        accumulator = delta + gamma * lam * (1 - float(episode_ends[t])) * accumulator
        advantages[t] = accumulator
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


def ppo_objective(ratios, advantages, epsilon, no_clip=False):
    direct = ratios * advantages
    return direct if no_clip else torch.minimum(direct, ratios.clamp(1-epsilon, 1+epsilon) * advantages)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def train(config, out, steps=None, initial_actor_state=None, initialization_provenance=None):
    config.validate()
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Use a new run directory.")
    out.mkdir(parents=True, exist_ok=True)
    steps = config.train_steps if steps is None else int(steps)
    if steps < 1:
        raise ValueError("steps must be positive")
    config = replace(config, train_steps=steps).validate()
    seed_everything(config.seed)
    env = SwarmEnv(config)
    actor, critic = make_networks(config, env)
    if initialization_provenance is not None and initial_actor_state is None:
        raise ValueError("initialization_provenance requires initial_actor_state")
    if initial_actor_state is not None:
        actor.load_state_dict(initial_actor_state, strict=True)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    save_json(out / "config.json", config.to_dict())
    started = time.perf_counter()
    provenance = {"implementation": "maintained-reference-v0.2", "seed": config.seed,
                  "requested_environment_steps": steps, "paper_reproduction": False,
                  "historical_weights_compatible": False, "python": platform.python_version(),
                  "torch": torch.__version__, "numpy": np.__version__, "device": "cpu",
                  "actor_parameters": sum(p.numel() for p in actor.parameters()),
                  "critic_parameters": sum(p.numel() for p in critic.parameters()),
                  "notes": "Maintenance implementation with explicit assumptions; short runs verify execution only."}
    provenance["initialization"] = (
        {"kind": "external-actor-state", "critic": "random initialization",
         "optimizer_state_resumed": False, **(initialization_provenance or {})}
        if initial_actor_state is not None else {"kind": "random initialization"})
    package = Path(__file__).resolve().parent
    provenance["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.glob("*.py"))
    }
    provenance["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["command"] = sys.argv
    provenance["platform"] = platform.platform()
    try:
        provenance["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(package), stderr=subprocess.DEVNULL,
            text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        provenance["git_commit"] = None
    save_json(out / "provenance.json", provenance)
    local, global_tokens = env.reset()
    consumed, updates = 0, 0
    episode_return, completed_returns = 0.0, []
    while consumed < steps:
        buffer = {key: [] for key in ("local", "global", "raw", "old_logp", "reward", "value",
                                      "next_value", "terminated", "episode_end")}
        rollout_metrics = []
        for _ in range(min(config.rollout_steps, steps - consumed)):
            with torch.no_grad():
                action, raw, logp = actor.sample(torch.from_numpy(local))
                value = critic(torch.from_numpy(global_tokens).unsqueeze(0)).item()
            (next_local, next_global), reward, terminated, truncated, metrics = env.step(action.numpy())
            with torch.no_grad():
                next_value = critic(torch.from_numpy(next_global).unsqueeze(0)).item()
            for key, item in (("local", local), ("global", global_tokens), ("raw", raw.numpy()),
                              ("old_logp", logp.numpy()), ("reward", reward), ("value", value),
                              ("next_value", next_value), ("terminated", terminated),
                              ("episode_end", terminated or truncated)):
                buffer[key].append(item)
            rollout_metrics.append(metrics)
            episode_return += reward
            consumed += 1
            if terminated or truncated:
                completed_returns.append(episode_return)
                episode_return = 0.0
                local, global_tokens = env.reset()
            else:
                local, global_tokens = next_local, next_global
        advantage, targets = generalized_advantage(buffer["reward"], buffer["value"],
                                                    buffer["next_value"], buffer["terminated"],
                                                    buffer["episode_end"], config.gamma, config.gae_lambda)
        advantage = (advantage - advantage.mean()) / max(float(advantage.std()), 1e-8)
        data = {key: torch.as_tensor(np.asarray(buffer[key]), dtype=torch.float32)
                for key in ("local", "global", "raw", "old_logp")}
        advantage, targets = torch.from_numpy(advantage), torch.from_numpy(targets)
        total = len(targets)
        diagnostics = []
        for _ in range(config.ppo_epochs):
            for indices in torch.randperm(total).split(config.minibatch_size):
                tokens = data["local"][indices].flatten(0, 1)
                raw_actions = data["raw"][indices].flatten(0, 1)
                logp, entropy = actor.evaluate_actions(tokens, raw_actions)
                old_logp = data["old_logp"][indices].flatten()
                # Guard exponentiation against numeric overflow; this does not
                # pretend to constrain the policy ratio to PPO clip boundaries.
                ratios = (logp - old_logp).clamp(-20, 20).exp()
                expanded_advantage = advantage[indices, None].expand(-1, config.num_agents).flatten()
                policy_loss = -ppo_objective(ratios, expanded_advantage,
                                             config.clip_epsilon, config.no_clip).mean()
                actor_loss = policy_loss - config.entropy_coefficient * entropy.mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), config.max_grad_norm)
                actor_optimizer.step()
                predictions = critic(data["global"][indices])
                value_loss = .5 * (predictions - targets[indices]).square().mean()
                critic_optimizer.zero_grad()
                (config.value_coefficient * value_loss).backward()
                critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                critic_optimizer.step()
                diagnostics.append({"policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
                                    "entropy_mc": entropy.mean().item(), "approx_kl": (old_logp-logp).mean().item(),
                                    "clip_fraction": ((ratios-1).abs() > config.clip_epsilon).float().mean().item(),
                                    "actor_gradient_norm": float(actor_grad), "critic_gradient_norm": float(critic_grad)})
        updates += 1
        record = {"environment_steps": consumed, "updates": updates,
                  "elapsed_seconds": time.perf_counter()-started, "completed_episodes": len(completed_returns),
                  "last_completed_episode_return": completed_returns[-1] if completed_returns else None,
                  "mean_step_reward": float(np.mean(buffer["reward"])),
                  "communication": float(np.mean([m["communication"] for m in rollout_metrics])),
                  "sensing_proxy": float(np.mean([m["sensing_proxy"] for m in rollout_metrics])),
                  "collision_step_fraction": float(np.mean([m["collision_event"] for m in rollout_metrics]))}
        record.update({key: float(np.mean([d[key] for d in diagnostics])) for key in diagnostics[0]})
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    checkpoint = {"schema_version": "1.0", "implementation": "maintained-reference-v0.2",
                  "config": config.to_dict(), "steps": consumed, "actor": actor.state_dict(),
                  "critic": critic.state_dict(), "seed": config.seed,
                  "initialization": provenance["initialization"]}
    torch.save(checkpoint, out / "checkpoint.pt")
    provenance.update(actual_environment_steps=consumed, training_seconds=time.perf_counter()-started,
                      completed_episodes=len(completed_returns))
    save_json(out / "provenance.json", provenance)
    evaluation = evaluate(config, actor, [1001, 1002, 1003],
                          checkpoint=str(out / "checkpoint.pt"), trained_steps=consumed)
    save_json(out / "evaluation.json", evaluation)
    return provenance


def evaluate(config, actor=None, seeds=None, checkpoint=None, trained_steps=0, illustrative=False):
    seeds = list(seeds if seeds is not None else [1001, 1002, 1003])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide at least one distinct evaluation seed")
    if actor is not None:
        actor.eval()
    episodes, first_trajectory, first_users = [], None, None
    for seed in seeds:
        env = SwarmEnv(config, seed=seed)
        local, _ = env.reset()
        frames = [env.frame()]
        episode_metrics = []
        for _ in range(config.horizon):
            if actor is None:
                actions, attention = env.rule_actions(), None
            else:
                with torch.no_grad():
                    tokens = torch.from_numpy(local)
                    actions = actor.sample(tokens, deterministic=True)[0].numpy()
                    weights = actor.attention_weights(tokens)
                    attention = weights.tolist() if weights is not None else None
            # These weights describe the observation used to choose this action,
            # so attach them to the current frame, not the resulting next state.
            if attention is not None:
                frames[-1]["attention"] = attention
            (local, _), _, terminated, truncated, metrics = env.step(actions)
            episode_metrics.append(metrics)
            frames.append(env.frame())
            if terminated or truncated:
                break
        if actor is not None:
            with torch.no_grad():
                final_weights = actor.attention_weights(torch.from_numpy(local))
                if final_weights is not None:
                    frames[-1]["attention"] = final_weights.tolist()
        sampled_metrics = [frames[0]["metrics"]] + episode_metrics
        episodes.append({"evaluation_seed": int(seed), "steps": len(episode_metrics),
                         "episode_return": float(sum(m["reward"] for m in episode_metrics)),
                         "communication_mean": float(np.mean([m["communication"] for m in episode_metrics])),
                         "sensing_proxy_mean": float(np.mean([m["sensing_proxy"] for m in episode_metrics])),
                         "speed_cost_mean": float(np.mean([m["speed_cost"] for m in episode_metrics])),
                         "min_distance": float(min(m["min_distance"] for m in sampled_metrics)),
                         "collision_episode": int(any(m["collision_event"] for m in sampled_metrics)),
                         "boundary_episode": int(any(m["boundary_event"] for m in sampled_metrics))})
        if first_trajectory is None:
            first_trajectory, first_users = frames, env.users.tolist()
    summary = {}
    for key in ("episode_return", "communication_mean", "sensing_proxy_mean", "speed_cost_mean",
                "min_distance", "collision_episode", "boundary_episode"):
        numbers = [episode[key] for episode in episodes]
        summary[key + "_mean"] = float(np.mean(numbers))
        summary[key + "_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
    summary.update(evaluation_episodes=len(seeds), independent_training_runs=0 if actor is None else 1,
                   uncertainty_unit="evaluation episodes of one fixed policy; NOT independent training seeds")
    return {"schema_version": "1.0", "kind": "evaluation",
            "provenance": {"status": "illustrative" if illustrative or actor is None else "maintenance-smoke",
                           "policy": "rule-based teaching controller" if actor is None else f"maintained MAPPO reference; attention={config.attention_mode}",
                           "seed": int(seeds[0]), "training_seed": config.seed if actor is not None else None,
                           "checkpoint": checkpoint, "trained_environment_steps": trained_steps,
                           "paper_reproduction": False,
                           "notes": "The replay shows only the first evaluation seed. New architecture and explicit assumptions; no claim of paper-level performance."},
            "config": config.to_dict(), "users": first_users, "trajectory": first_trajectory,
            "episodes": episodes, "summary": summary,
            "metric_definitions": {"communication": "sum-link spectral efficiency, bit/s/Hz; no bandwidth/association model",
                                   "sensing_proxy": "dimensionless SNR proxy; not CRB or positioning RMSE",
                                   "speed_cost": "weighted sum of UAV speeds; not energy in joules",
                                   "collision_episode": "at least one sampled distance below safe_distance, including the reset state; not continuous-time safety",
                                   "attention": "weights computed from the observation at this same replay frame"}}


def load_checkpoint(path):
    from .config import Config
    # This loader accepts maintained state dictionaries only. Never use arbitrary
    # pickle objects. weights_only is supported in the tested PyTorch version.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("implementation") != "maintained-reference-v0.2":
        raise ValueError("Historical checkpoints are incompatible with the maintained entity-token architecture")
    config = Config.from_dict(data["config"])
    actor, critic = make_networks(config, SwarmEnv(config))
    actor.load_state_dict(data["actor"])
    critic.load_state_dict(data["critic"])
    return config, actor, critic, data["steps"]


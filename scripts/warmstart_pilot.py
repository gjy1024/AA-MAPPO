"""Exploratory local-teacher behavior cloning followed by PPO fine tuning.

This is a distinct demonstration-assisted extension, NOT original AA-MAPPO.
Environment physics, reward and Actor architecture are kept unchanged. The
teacher reads only the same local entity tokens available to that Actor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aamappo.config import Config
from aamappo.environment import SwarmEnv
from aamappo.models import make_networks
from aamappo.ppo import evaluate, load_checkpoint, save_json, seed_everything, train
from benchmark_pilot import aggregate, ui_metrics


class LocalRingTeacher:
    """Nearest-user 6m ring, neighbor repulsion, velocity damping; no safety proof."""
    def __init__(self, config, ring_radius=6.0, repulsion_radius=12.0):
        self.config = config
        self.ring_radius = ring_radius
        self.repulsion_radius = repulsion_radius

    def actions(self, tokens):
        c = self.config
        tokens = np.asarray(tokens, dtype=np.float64)
        velocity = tokens[:, 0, 2:4] * c.max_speed
        user_relative = tokens[:, c.num_agents:c.num_agents+c.num_users, :2] * c.map_size
        neighbor_relative = tokens[:, 1:c.num_agents, :2] * c.map_size
        distance = np.linalg.norm(user_relative, axis=-1)
        nearest = np.argmin(distance, axis=1)
        target_relative = user_relative[np.arange(len(tokens)), nearest]
        target_distance = np.linalg.norm(target_relative, axis=-1, keepdims=True)
        # The chosen point is on the current approach ray, not at the user itself.
        direction = target_relative / np.maximum(target_distance, 1e-8)
        direction[target_distance[:, 0] < 1e-8] = np.array([1.0, 0.0])
        ring_error = direction * (target_distance - self.ring_radius)
        neighbor_distance = np.linalg.norm(neighbor_relative, axis=-1, keepdims=True)
        away = -neighbor_relative / np.maximum(neighbor_distance, 1e-8)
        strength = np.maximum(0, 1-neighbor_distance/self.repulsion_radius) ** 2
        desired_velocity = .8 * ring_error + 8.0 * (away * strength).sum(axis=1)
        desired_velocity /= np.maximum(1.0, np.linalg.norm(desired_velocity, axis=-1, keepdims=True)/c.max_speed)
        acceleration = (desired_velocity-velocity)/.7
        action = acceleration/c.max_acceleration
        action /= np.maximum(1.0, np.linalg.norm(action, axis=-1, keepdims=True))
        return action.astype(np.float32)

    def eval(self):
        return self

    def sample(self, tokens, deterministic=False):
        actions = torch.from_numpy(self.actions(tokens.detach().cpu().numpy()))
        return actions, torch.atanh(actions.clamp(-.9999, .9999)), torch.zeros(len(actions))

    def attention_weights(self, tokens):
        return None


def evaluate_stage(config, policy, label, seeds, out, trained_steps=0, checkpoint=None, training_runs=1):
    out.mkdir(parents=True, exist_ok=True)
    episodes, first, paths = [], None, []
    for seed in seeds:
        result = evaluate(config, policy, seeds=[seed], checkpoint=checkpoint, trained_steps=trained_steps)
        result["provenance"].update(status="maintenance-pilot", policy=label,
                                    paper_reproduction=False,
                                    notes="Demonstration-assisted extension pilot; all predefined validation scenarios retained; final test set unused.")
        if training_runs == 0:
            result["provenance"]["training_seed"] = None
        result["summary"]["independent_training_runs"] = training_runs
        path = out/f"seed-{seed}.json"
        save_json(path, result)
        paths.append(str(path))
        episodes.extend(result["episodes"])
        if first is None:
            first = result
    first["episodes"] = episodes
    first["summary"] = aggregate(episodes)
    first["summary"]["independent_training_runs"] = training_runs
    first["all_replays"] = paths
    first["provenance"]["overview_selection"] = "first predefined validation seed, not best outcome"
    save_json(out/"validation.json", first)
    return first


def collect_demonstrations(config, teacher, episodes, out, seed_start=10000):
    observations, actions = [], []
    for seed in range(seed_start, seed_start+episodes):
        env = SwarmEnv(config, seed=seed)
        local, _ = env.reset()
        for _ in range(config.horizon):
            target = teacher.actions(local)
            observations.append(local.copy())
            actions.append(target.copy())
            (local, _), _, terminated, truncated, _ = env.step(target)
            if terminated or truncated:
                break
    observations = np.concatenate(observations).astype(np.float32)
    actions = np.concatenate(actions).astype(np.float32)
    path = out/"demonstrations.npz"
    np.savez_compressed(path, local_tokens=observations, teacher_actions=actions)
    metadata = {"training_scenario_seeds": list(range(seed_start, seed_start+episodes)),
                "episodes": episodes, "agent_observation_action_samples": len(actions),
                "teacher": "local nearest-user 6m ring + 12m neighbor repulsion + velocity damping",
                "teacher_has_global_critic_state": False,
                "validation_scenario_seeds_used_for_demonstrations": False,
                "dataset": str(path), "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    save_json(out/"demonstrations.json", metadata)
    return observations, actions, metadata


def behavior_clone(config, observations, actions, epochs, out):
    seed_everything(config.seed)
    actor, critic = make_networks(config, SwarmEnv(config))
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    inputs, targets = torch.from_numpy(observations), torch.from_numpy(actions)
    raw_targets = torch.atanh(targets.clamp(-.9999, .9999))
    started = time.perf_counter()
    logs = []
    actor.train()
    for epoch in range(epochs):
        totals, batches = np.zeros(3), 0
        for indices in torch.randperm(len(inputs)).split(512):
            distribution = actor.distribution(inputs[indices])
            mse = (distribution.mean.tanh()-targets[indices]).square().mean()
            nll = -distribution.log_prob(raw_targets[indices]).mean()
            loss = mse + .02*nll
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
            optimizer.step()
            totals += [loss.item(), mse.item(), nll.item()]
            batches += 1
        record = {"epoch": epoch+1, "loss": float(totals[0]/batches),
                  "action_mse": float(totals[1]/batches), "raw_action_nll": float(totals[2]/batches),
                  "log_std": actor.log_std.detach().tolist(), "elapsed_seconds": time.perf_counter()-started}
        logs.append(record)
        with (out/"bc-metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False)+"\n")
    checkpoint = {"schema_version": "1.0", "implementation": "maintained-reference-v0.2",
                  "config": config.to_dict(), "steps": 0, "training_stage": "behavior-cloning",
                  "actor": actor.state_dict(), "critic": critic.state_dict(), "seed": config.seed}
    torch.save(checkpoint, out/"bc-checkpoint.pt")
    return actor, {"epochs": epochs, "samples": len(inputs), "learning_rate": 1e-3,
                   "batch_size": 512, "loss": "action MSE + 0.02 * raw-action Gaussian NLL",
                   "seconds": time.perf_counter()-started, "final_metrics": logs[-1]}


def row(identifier, label, result, steps, source):
    return {"id": identifier, "label": label, "policy": result["provenance"]["policy"],
            "training_seed": (result["provenance"].get("training_seed")
                              if result["summary"]["independent_training_runs"] else None),
            "training_steps": steps,
            "metrics": ui_metrics(result["summary"]), "source": str(source)}


def run(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite existing pilot: {out}")
    config = Config.load(args.config)
    if not config.attention_mode == "both":
        raise ValueError("This pilot uses the unchanged dual-attention Actor/Critic architecture")
    if any(3001 <= seed <= 3020 for seed in args.validation_seeds):
        raise ValueError("Final test seeds 3001..3020 are reserved and must remain unused")
    seed_everything(config.seed)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out/"config.json", config.to_dict())
    teacher = LocalRingTeacher(config)
    provenance = {"kind": "demonstration-assisted-PPO-extension", "paper_reproduction": False,
                  "started_at_utc": datetime.now(timezone.utc).isoformat(),
                  "config": args.config, "validation_seeds": args.validation_seeds,
                  "environment_reward_actor_structure_unchanged": True,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "teacher": {"ring_radius": 6.0, "repulsion_radius": 12.0,
                              "observation": "only local entity tokens; no user assignment index/global critic state",
                              "safety": "heuristic repulsion; no hard guarantee"}}
    save_json(out/"provenance.json", provenance)
    teacher_result = evaluate_stage(config, teacher, "local-ring-repulsion-teacher", args.validation_seeds,
                                    out/"teacher", training_runs=0)
    print(json.dumps({"stage": "teacher", "summary": teacher_result["summary"]}), flush=True)
    observations, actions, metadata = collect_demonstrations(config, teacher, args.demo_episodes, out)
    actor, pretraining = behavior_clone(config, observations, actions, args.bc_epochs, out)
    bc_result = evaluate_stage(config, actor, "behavior-cloned local teacher; no PPO yet", args.validation_seeds,
                              out/"bc", checkpoint=str(out/"bc-checkpoint.pt"))
    print(json.dumps({"stage": "behavior-cloning", "summary": bc_result["summary"]}), flush=True)
    baseline_config, baseline_actor, _, baseline_steps = load_checkpoint(args.baseline)
    excluded = {"train_steps", "seed", "rollout_steps", "gae_lambda", "minibatch_size"}
    mismatches = {key for key, value in asdict(config).items()
                  if key not in excluded and value != asdict(baseline_config)[key]}
    if mismatches:
        raise ValueError(f"Baseline task/network differs: {sorted(mismatches)}")
    baseline_result = evaluate_stage(baseline_config, baseline_actor, "maintenance short-training baseline",
                                    args.validation_seeds, out/"baseline", baseline_steps, str(args.baseline))
    rows = [row("short-training", "短训练基准", baseline_result, baseline_steps, out/"baseline/validation.json"),
            row("teacher", "局部环形停靠与排斥控制（未训练）", teacher_result, 0, out/"teacher/validation.json"),
            row("behavior-cloning", "示范预训练扩展（仅BC）", bc_result, 0, out/"bc/validation.json")]
    before, after = baseline_result["summary"], bc_result["summary"]
    improved = (after["episode_return_mean"] > before["episode_return_mean"] + max(10, abs(before["episode_return_mean"])*.1)
                and after["collision_episode_mean"] <= before["collision_episode_mean"]
                and after["boundary_episode_mean"] <= before["boundary_episode_mean"]
                and after["communication_mean_mean"] >= before["communication_mean_mean"]
                and after["sensing_proxy_mean_mean"] >= before["sensing_proxy_mean_mean"])
    provenance.update(demonstrations=metadata, behavior_cloning=pretraining,
                      bc_clear_improvement_over_short_training=bool(improved),
                      ppo_finetuning_started=bool(improved))
    save_json(out/"provenance.json", provenance)
    if improved:
        initialization = {"kind": "behavior-cloned-local-teacher", "teacher": provenance["teacher"],
                          "pretraining": pretraining, "demonstration_metadata": metadata,
                          "source_checkpoint": str(out/"bc-checkpoint.pt"),
                          "source_sha256": hashlib.sha256((out/"bc-checkpoint.pt").read_bytes()).hexdigest()}
        print(json.dumps({"stage": "PPO-finetuning-started", "steps": args.ppo_steps}), flush=True)
        train(config, out/"ppo", args.ppo_steps,
              initial_actor_state=actor.state_dict(), initialization_provenance=initialization)
        ppo_config, ppo_actor, _, ppo_steps = load_checkpoint(out/"ppo/checkpoint.pt")
        ppo_result = evaluate_stage(ppo_config, ppo_actor, "local-teacher BC + PPO extension", args.validation_seeds,
                                   out/"ppo/validation", ppo_steps, str(out/"ppo/checkpoint.pt"))
        rows.append(row("bc-ppo", "示范预训练＋PPO微调扩展", ppo_result, ppo_steps,
                        out/"ppo/validation/validation.json"))
        print(json.dumps({"stage": "PPO-finetuning-completed", "summary": ppo_result["summary"]}), flush=True)
    comparison = {"schema_version": "1.0", "kind": "pilot-comparison",
                  "provenance": {"status": "maintenance-pilot", "training_steps": args.ppo_steps if improved else 0,
                                 "training_seeds": [config.seed], "evaluation_seeds": args.validation_seeds,
                                 "config": args.config, "notes": "示范预训练扩展，区别于原版AA-MAPPO；只用验证集诊断，非多训练种子定论；全部轨迹保留。"},
                  "rows": rows}
    save_json(out/"comparison.json", comparison)
    print(json.dumps({"comparison": str(out/"comparison.json"), "ppo_executed": bool(improved)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot-long-rollout.json")
    parser.add_argument("--out", default="artifacts/warmstart-pilot-seed7")
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/maintenance-smoke-v2/checkpoint.pt"))
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=list(range(2001, 2011)))
    parser.add_argument("--demo-episodes", type=int, default=64)
    parser.add_argument("--bc-epochs", type=int, default=40)
    parser.add_argument("--ppo-steps", type=int, default=30000)
    args = parser.parse_args()
    if min(args.demo_episodes, args.bc_epochs, args.ppo_steps) < 1:
        parser.error("All experiment budgets must be positive")
    run(args)


if __name__ == "__main__":
    main()

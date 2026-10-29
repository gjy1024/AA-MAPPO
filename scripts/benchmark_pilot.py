"""Paired scenario benchmark of two maintained checkpoints; never selects runs.

This evaluates one trained policy per checkpoint. Scenario variation is NOT
independent training-run replication. Each scenario's complete replay is saved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aamappo.ppo import evaluate, load_checkpoint, save_json, seed_everything


METRICS = {
    "episode_return": "higher",
    "communication_mean": "higher",
    "sensing_proxy_mean": "higher",
    "speed_cost_mean": "lower",
    "min_distance": "higher",
    "collision_episode": "lower",
    "boundary_episode": "lower",
}

TRAINING_PARAMETERS = {
    "seed", "train_steps", "learning_rate", "gamma", "clip_epsilon", "no_clip",
    "gae_lambda", "rollout_steps", "ppo_epochs", "minibatch_size",
    "entropy_coefficient", "value_coefficient", "max_grad_norm", "initial_log_std",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(episodes):
    result = {}
    for metric in METRICS:
        values = np.asarray([episode[metric] for episode in episodes], dtype=np.float64)
        result[metric + "_mean"] = float(values.mean())
        result[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
    result.update(evaluation_episodes=len(episodes), independent_training_runs=1,
                  uncertainty_unit="fixed evaluation scenarios of one checkpoint; not training seeds")
    return result


def ui_metrics(summary):
    return {"communication": summary["communication_mean_mean"],
            "sensing_proxy": summary["sensing_proxy_mean_mean"],
            "collision_episode_rate": summary["collision_episode_mean"],
            "min_distance": summary["min_distance_mean"],
            "mean_return": summary["episode_return_mean"],
            "boundary_episode_rate": summary["boundary_episode_mean"]}


def write_ui_comparison(report, report_path, baseline_results=None):
    report_path = Path(report_path)
    rows = []
    for role, label in (("baseline", "维护版短训练"), ("candidate", "探索性延长训练")):
        record = report[role]
        rows.append({"id": role, "label": f"{label}（{record['trained_environment_steps']:,}步）",
                     "policy": "maintained MAPPO entity-attention policy",
                     "training_seed": record["training_seed"],
                     "training_steps": record["trained_environment_steps"],
                     "metrics": ui_metrics(record["summary"]),
                     "source": (report_path.parent / f"{role}.json").as_posix()})
    for source in baseline_results or []:
        result = json.loads(Path(source).read_text(encoding="utf-8"))
        result_seeds = [episode["evaluation_seed"] for episode in result["episodes"]]
        if result_seeds != report["protocol"]["evaluation_seeds"]:
            raise ValueError(f"Baseline evaluation seeds differ: {source}")
        policy = result["provenance"]["policy"]
        rows.append({"id": Path(source).stem, "label": policy, "policy": policy,
                     "training_seed": None, "training_steps": 0,
                     "metrics": ui_metrics(result["summary"]), "source": Path(source).as_posix()})
    comparison = {"schema_version": "1.0", "kind": "pilot-comparison",
                  "provenance": {"status": "maintenance-pilot",
                                 "notes": "探索性训练，非多训练种子定论；验证集用于诊断，全部场景保留，未使用最终测试集。",
                                 "training_steps": report["candidate"]["trained_environment_steps"],
                                 "training_seeds": [report["candidate"]["training_seed"]],
                                 "evaluation_seeds": report["protocol"]["evaluation_seeds"],
                                 "config": "configs/maintenance.json",
                                 "config_differences": report["protocol"]["config_differences"],
                                 "benchmark_report": report_path.as_posix()},
                  "rows": rows}
    destination = report_path.parent / "comparison.json"
    save_json(destination, comparison)
    return destination


def diagnose_replays(report, out):
    diagnostics = {}
    for role in ("baseline", "candidate"):
        records = []
        for replay_file in report[role]["replays"]:
            replay = json.loads((out / replay_file).read_text(encoding="utf-8"))
            frames = replay["trajectory"]
            metrics = [frame["metrics"] for frame in frames[1:]]
            positions = np.asarray([frame["positions"] for frame in frames])
            records.append({
                "evaluation_seed": replay["episodes"][0]["evaluation_seed"],
                "collision_step_fraction": float(np.mean([m["collision_event"] for m in metrics])),
                "boundary_step_fraction": float(np.mean([m["boundary_event"] for m in metrics])),
                "mean_collision_penalty": float(np.mean([m["collision_cost"] for m in metrics])),
                "mean_speed_cost": float(np.mean([m["speed_cost"] for m in metrics])),
                "mean_communication": float(np.mean([m["communication"] for m in metrics])),
                "mean_sensing_proxy": float(np.mean([m["sensing_proxy"] for m in metrics])),
                "total_path_length_all_agents": float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum()),
            })
        diagnostics[role] = {"per_scenario": records,
                             "means": {key: float(np.mean([record[key] for record in records]))
                                       for key in records[0] if key != "evaluation_seed"}}
    diagnostics["interpretation"] = [
        "These diagnostics preserve the predefined scenarios; no attractive trajectory was selected.",
        "Lower return must be read with communication, sensing, collision and speed components.",
        "Boundary clipping is an idealized environment event, not a real-flight success criterion.",
        "One trained checkpoint per condition cannot establish training-seed robustness.",
    ]
    save_json(out / "diagnostics.json", diagnostics)
    return diagnostics


def evaluate_all(label, config, actor, steps, checkpoint, seeds, out):
    seed_everything(config.seed)
    episodes, replay_files, overview = [], [], None
    for seed in seeds:
        replay = evaluate(config, actor, seeds=[seed], checkpoint=str(checkpoint), trained_steps=steps)
        replay["provenance"]["benchmark_role"] = label
        replay["provenance"]["checkpoint_sha256"] = digest(checkpoint)
        replay["provenance"]["notes"] += " All fixed-scenario replays are archived; no best-run selection."
        relative = f"{label}/seed-{seed}.json"
        save_json(out / relative, replay)
        replay_files.append(relative)
        episodes.extend(replay["episodes"])
        if overview is None:
            overview = replay
    # The first predefined scenario remains the overview replay; all others are
    # retained as separate files. It is never chosen based on observed performance.
    overview["episodes"] = episodes
    overview["summary"] = aggregate(episodes)
    overview["all_replays"] = replay_files
    overview["provenance"]["overview_selection"] = "first predefined evaluation seed, not the best outcome"
    save_json(out / f"{label}.json", overview)
    return episodes, replay_files, overview["summary"]


def benchmark(baseline, candidate, out, seeds):
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Evaluation seeds must be nonempty and unique")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty benchmark directory: {out}")
    base_config, base_actor, _, base_steps = load_checkpoint(baseline)
    candidate_config, candidate_actor, _, candidate_steps = load_checkpoint(candidate)
    left, right = asdict(base_config), asdict(candidate_config)
    differences = {key: {"baseline": left[key], "candidate": right[key]}
                   for key in left if left[key] != right[key]}
    unfair_differences = set(differences) - TRAINING_PARAMETERS
    if unfair_differences:
        raise ValueError(f"Environment/reward/network configs differ: {sorted(unfair_differences)}")
    out.mkdir(parents=True, exist_ok=True)
    base_episodes, base_files, base_summary = evaluate_all(
        "baseline", base_config, base_actor, base_steps, baseline, seeds, out)
    candidate_episodes, candidate_files, candidate_summary = evaluate_all(
        "candidate", candidate_config, candidate_actor, candidate_steps, candidate, seeds, out)
    pairs = []
    for base_episode, candidate_episode in zip(base_episodes, candidate_episodes):
        if base_episode["evaluation_seed"] != candidate_episode["evaluation_seed"]:
            raise ValueError("Scenario alignment failed")
        pairs.append({"evaluation_seed": base_episode["evaluation_seed"],
                      "baseline": base_episode, "candidate": candidate_episode,
                      "delta_candidate_minus_baseline": {
                          metric: candidate_episode[metric] - base_episode[metric] for metric in METRICS}})
    comparisons = {}
    for metric, preferred in METRICS.items():
        deltas = np.asarray([pair["delta_candidate_minus_baseline"][metric] for pair in pairs])
        oriented = deltas if preferred == "higher" else -deltas
        comparisons[metric] = {
            "preferred_direction": preferred,
            "baseline_mean": base_summary[metric + "_mean"],
            "candidate_mean": candidate_summary[metric + "_mean"],
            "mean_delta_candidate_minus_baseline": float(deltas.mean()),
            "scenario_delta_std": float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
            "better_scenarios": int((oriented > 1e-10).sum()),
            "tied_scenarios": int((np.abs(oriented) <= 1e-10).sum()),
            "worse_scenarios": int((oriented < -1e-10).sum()),
        }
    report = {
        "schema_version": "1.0", "kind": "paired-pilot-benchmark",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pilot; not a paper reproduction or a multi-training-seed significance test",
        "protocol": {"evaluation_seeds": seeds, "deterministic_policy": True,
                     "scenario_selection": "predefined seeds; every scenario retained",
                     "independent_training_runs_per_condition": 1,
                     "same_environment_reward_network_config": True,
                     "config_differences": differences,
                     "uncertainty_unit": "evaluation scenarios, not independent training seeds",
                     "limitations": ["Only one training seed per condition",
                                     "Validation scenarios are not an untouched final test set",
                                     "Safety measured at sampled states, not a hard guarantee",
                                     "Sensing uses an SNR proxy, not localization RMSE or CRB"]},
        "baseline": {"checkpoint": str(baseline), "checkpoint_sha256": digest(baseline),
                     "trained_environment_steps": base_steps, "training_seed": base_config.seed,
                     "summary": base_summary, "replays": base_files},
        "candidate": {"checkpoint": str(candidate), "checkpoint_sha256": digest(candidate),
                      "trained_environment_steps": candidate_steps, "training_seed": candidate_config.seed,
                      "summary": candidate_summary, "replays": candidate_files},
        "comparisons": comparisons, "paired_episodes": pairs,
    }
    save_json(out / "benchmark.json", report)
    diagnose_replays(report, out)
    write_ui_comparison(report, out / "benchmark.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--from-report", type=Path, help="Only derive UI comparison/diagnostics from an existing full report")
    parser.add_argument("--baseline-results", type=Path, nargs="*", default=[],
                        help="Optional stationary/rule evaluations on exactly the same seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(2001, 2011)))
    args = parser.parse_args()
    if args.from_report:
        report = json.loads(args.from_report.read_text(encoding="utf-8"))
        diagnose_replays(report, args.from_report.parent)
        comparison = write_ui_comparison(report, args.from_report, args.baseline_results)
        print(json.dumps({"comparison": str(comparison)}, ensure_ascii=False))
        return
    if not args.baseline or not args.candidate or not args.out:
        parser.error("--baseline, --candidate, --out are required unless --from-report is used")
    report = benchmark(args.baseline, args.candidate, args.out, args.seeds)
    print(json.dumps({"report": str(args.out / "benchmark.json"),
                      "baseline_steps": report["baseline"]["trained_environment_steps"],
                      "candidate_steps": report["candidate"]["trained_environment_steps"],
                      "comparisons": report["comparisons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))

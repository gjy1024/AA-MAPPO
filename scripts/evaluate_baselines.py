"""Evaluate non-learning controls on the maintained task without changing rewards.

The stationary and existing round-robin PD policies are measured baselines, not
AA-MAPPO results. All use the same evaluator, reset seeds and environment config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from aamappo.config import Config
from aamappo.ppo import evaluate, save_json


class StationaryActor:
    """Zero acceleration; reset velocity is zero in this benchmark."""

    def eval(self):
        return self

    def sample(self, tokens, deterministic=False):
        actions = torch.zeros((tokens.shape[0], 2), dtype=tokens.dtype, device=tokens.device)
        return actions, actions, torch.zeros(tokens.shape[0], dtype=tokens.dtype, device=tokens.device)

    def attention_weights(self, tokens):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/maintenance.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2001, 2011)))
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/baselines")
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide distinct evaluation seeds")
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    source_files = ["aamappo/config.py", "aamappo/environment.py", "aamappo/ppo.py",
                    "scripts/evaluate_baselines.py"]
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in source_files}
    policies = [
        ("stationary.json", "stationary-zero-acceleration", StationaryActor(),
         "Zero acceleration with zero initial velocity; no learned parameters."),
        ("rule.json", "round-robin-pd-rule", None,
         "Existing SwarmEnv.rule_actions(): target user i modulo user count; "
         "fixed gains 0.06 and 0.6, no collision avoidance. Uses agent-index assignment."),
    ]
    comparison = {"schema_version": "1.0", "kind": "baseline-comparison",
                  "config": config.to_dict(), "evaluation_seeds": args.seeds, "policies": [],
                  "notes": "Unchanged physical environment/reward. These are fixed-policy scenario "
                           "evaluations, not independent training seeds, and not AA-MAPPO results."}
    for filename, label, actor, description in policies:
        started = time.perf_counter()
        result = evaluate(config, actor=actor, seeds=args.seeds)
        duration = time.perf_counter() - started
        result["provenance"].update({
            "status": "baseline-evaluation", "policy": label, "training_seed": None,
            "evaluation_seeds": args.seeds, "trained_environment_steps": 0,
            "checkpoint": None, "paper_reproduction": False, "source_sha256": source_hashes,
            "notes": description + " Same evaluator and seeds as maintained-model validation; "
                     "replay contains only the first evaluation scenario.",
        })
        result["summary"].update({
            "independent_training_runs": 0,
            "uncertainty_unit": "scenario variation of one fixed non-learning policy; NOT training seeds",
            "evaluation_seconds": duration,
        })
        save_json(args.out / filename, result)
        comparison["policies"].append({"policy": label, "path": filename,
                                      "summary": result["summary"]})
    save_json(args.out / "comparison.json", comparison)
    print(json.dumps({"output": str(args.out), "evaluation_seeds": args.seeds,
                      "policies": comparison["policies"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2)

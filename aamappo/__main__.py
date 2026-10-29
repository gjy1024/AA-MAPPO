"""Run with python -m aamappo; no hidden working-directory assumptions."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shlex
import sys
from .config import Config


def parser():
    root = argparse.ArgumentParser(description="AA-MAPPO maintained reference tools (not exact historical reproduction)")
    commands = root.add_subparsers(dest="command", required=True)
    for command in ("train", "smoke", "demo"):
        item = commands.add_parser(command)
        item.add_argument("--config", default=None)
        item.add_argument("--seed", type=int, default=None)
        item.add_argument("--out", default=f"runs/{command}")
        item.add_argument("--attention-mode", choices=("none", "actor", "critic", "both"), default=None)
        if command != "demo":
            item.add_argument("--steps", type=int, default=512 if command == "smoke" else None)
    item = commands.add_parser("evaluate")
    item.add_argument("--checkpoint", required=True)
    item.add_argument("--out", default="runs/evaluation.json")
    item.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    item = commands.add_parser("sweep")
    item.add_argument("--config", default="configs/maintenance.json")
    item.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    item.add_argument("--steps", type=int, default=2000000)
    item.add_argument("--out", default="runs/ablation")
    item.add_argument("--dry-run", action="store_true", help="Explicitly print the experiment plan (also the default)")
    item.add_argument("--execute", action="store_true", help="Actually run the full plan; potentially expensive")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "sweep":
        from .ppo import train
        config = Config.load(args.config)
        if args.execute and args.dry_run:
            raise ValueError("--execute and --dry-run are mutually exclusive")
        for mode in ("none", "actor", "critic", "both"):
            for seed in args.seeds:
                destination = str(Path(args.out) / mode / f"seed-{seed}")
                if args.execute:
                    train(replace(config, attention_mode=mode, seed=seed), destination, args.steps)
                else:
                    command = [sys.executable, "-m", "aamappo", "train", "--config", args.config,
                               "--attention-mode", mode, "--seed", str(seed), "--steps", str(args.steps),
                               "--out", destination]
                    print(shlex.join(command))
        return 0
    from .ppo import evaluate, load_checkpoint, save_json, seed_everything, train
    if args.command == "evaluate":
        config, actor, _, steps = load_checkpoint(args.checkpoint)
        seed_everything(config.seed)
        result = evaluate(config, actor, args.seeds, args.checkpoint, steps)
        save_json(args.out, result)
        print(json.dumps({"evaluation": args.out, "summary": result["summary"]}, ensure_ascii=False))
        return 0
    config = Config.load(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.attention_mode is not None:
        config.attention_mode = args.attention_mode
    config.validate()
    if args.command == "demo":
        seed_everything(config.seed)
        path = Path(args.out)
        if path.suffix.lower() != ".json":
            path = path / "evaluation.json"
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing output: {path}")
        save_json(path, evaluate(config, seeds=[config.seed], illustrative=True))
        print(json.dumps({"evaluation": str(path), "status": "illustrative; rule controller, not trained AA-MAPPO"}))
    else:
        result = train(config, args.out, args.steps)
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)


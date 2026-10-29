"""Freeze validation-based selection, then evaluate once on untouched test seeds."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aamappo.ppo import evaluate, load_checkpoint, save_json, seed_everything
from benchmark_pilot import aggregate, ui_metrics
from warmstart_pilot import LocalRingTeacher


class Stationary:
    def eval(self): return self
    def sample(self, tokens, deterministic=False):
        import torch
        zero = torch.zeros((len(tokens), 2))
        return zero, zero, torch.zeros(len(tokens))
    def attention_weights(self, tokens): return None


def main():
    out = ROOT / "artifacts/final-test"
    if out.exists(): raise ValueError("Final test already exists; do not silently repeat selection")
    selected = ROOT / "artifacts/pilot-long-rollout-seed7/checkpoint.pt"
    out.mkdir(parents=True)
    seeds = list(range(3001, 3021))
    save_json(out / "selection.json", {"selected_before_test": True,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_checkpoint": selected.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "selection": "Among learned candidates: collision episode rate, then boundary episode rate, then return on validation seeds 2001..2010",
        "test_seeds": seeds, "training_seed": 7,
        "validation_results": {"default_30k_collision_rate": .5, "long_rollout_30k_collision_rate": .1,
                               "behavior_cloning_collision_rate": .2},
        "notes": "BC did not qualify for PPO fine tuning. Rule teacher is a separate non-learning reference."})
    cfg, actor, _, steps = load_checkpoint(selected)
    short_path = ROOT / "artifacts/maintenance-smoke-v2/checkpoint.pt"
    short_cfg, short_actor, _, short_steps = load_checkpoint(short_path)
    policies = [("short", "维护版短训练 · 512步", short_cfg, short_actor, short_steps, short_path, 1),
                ("long-rollout", "改进PPO · 30,000步", cfg, actor, steps, selected, 1),
                ("stationary", "静止参考 · 未训练", cfg, Stationary(), 0, None, 0),
                ("ring-teacher", "避碰环形规则参考 · 未训练", cfg, LocalRingTeacher(cfg), 0, None, 0)]
    rows = []
    for name, label, config, policy, budget, checkpoint, training_runs in policies:
        seed_everything(config.seed)
        episodes, first = [], None
        for seed in seeds:
            data = evaluate(config, policy, seeds=[seed], checkpoint=checkpoint.relative_to(ROOT).as_posix() if checkpoint else None, trained_steps=budget)
            data["provenance"].update(status="maintenance-pilot", policy=label,
                training_seed=config.seed if training_runs else None,
                evaluation_split="held-out-test", paper_reproduction=False,
                notes="独立测试3001–3020；检查点已在测试前按验证集固定。回放显示预先指定的首个测试场景。规则参考未训练，不是AA-MAPPO。" if not training_runs else "独立测试3001–3020；检查点已在测试前按验证集固定。回放显示预先指定的首个测试场景。单训练seed探索结果。")
            if checkpoint: data["provenance"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            data["summary"]["independent_training_runs"] = training_runs
            save_json(out / name / f"seed-{seed}.json", data)
            episodes.extend(data["episodes"])
            if first is None: first = data
        first["episodes"] = episodes
        first["summary"] = aggregate(episodes)
        first["summary"]["independent_training_runs"] = training_runs
        first["all_replays"] = [f"{name}/seed-{s}.json" for s in seeds]
        save_json(out / f"{name}.json", first)
        rows.append({"id": name, "label": label, "policy": label,
            "training_seed": config.seed if training_runs else None, "training_steps": budget,
            "metrics": ui_metrics(first["summary"]), "source": f"artifacts/final-test/{name}.json"})
        print(json.dumps({"policy": name, "metrics": rows[-1]["metrics"]}, ensure_ascii=False), flush=True)
    save_json(out / "comparison.json", {"schema_version":"1.0", "kind":"pilot-comparison",
        "provenance":{"status":"maintenance-pilot", "training_steps":steps,"training_seeds":[7],
            "evaluation_seeds":seeds, "config":"configs/pilot-long-rollout.json",
            "notes":"独立测试20个固定场景；学习策略只有1个训练seed，不能说明多训练seed稳定性。最小间距为各回合最小值的均值；碰撞与触边为含事件回合比例。规则控制单独标注未训练。"},
        "rows":rows})


if __name__ == "__main__": main()

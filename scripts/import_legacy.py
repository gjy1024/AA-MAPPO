"""Import supplied historical artifacts without inventing seeds or measurements.

Raw logs/checkpoints are copied byte-for-byte. Public chart data are a deterministic
transformation of TensorBoard's exported [wall_time, step, value] records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path


ALGORITHMS = {
    "6G_ISAC_Attention": ("aa_mappo", "AA-MAPPO（历史单次运行）"),
    "Standard_MAPPO": ("standard_mappo", "Standard MAPPO（历史单次运行）"),
    "No_Clip_MAPPO": ("no_clip_mappo", "No-Clip MAPPO（历史单次运行）"),
    "Naive_Reward_MAPPO": ("naive_reward", "Naive Reward（历史单次运行）"),
}
METRICS = {
    "episode_reward": "episode_reward", "value_loss": "value_loss",
    "dist_entropy": "entropy", "ratio": "ratio", "policy_loss": "policy_loss",
    "actor_grad_norm": "actor_grad_norm", "critic_grad_norm": "critic_grad_norm",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalize_summary(raw: dict) -> tuple[dict, dict]:
    """Keep all per-agent records and derive explicitly named within-run means."""
    full, groups = {}, defaultdict(lambda: defaultdict(list))
    for tag, records in raw.items():
        portable_tag = tag.replace("\\", "/").split("/logs/")[-1]
        metric = METRICS.get(portable_tag.split("/")[-1])
        points = []
        for record in records:
            if not isinstance(record, list) or len(record) != 3:
                raise ValueError("Expected [wall_time, step, value] records")
            timestamp, step, value = record
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in record):
                raise ValueError("Non-finite/non-numeric historical record")
            points.append({"step": int(step), "value": float(value), "wall_time": float(timestamp)})
            if metric:
                groups[metric][int(step)].append(float(value))
        full[portable_tag] = points
    means = {
        metric: [{"step": step, "value": sum(values) / len(values)} for step, values in sorted(steps.items())]
        for metric, steps in groups.items()
    }
    return means, full


def import_artifacts(source: Path, destination: Path) -> dict:
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    runs, inventory = [], []
    for original_name, (run_id, label) in ALGORITHMS.items():
        candidates = [p for p in source.rglob("summary.json") if original_name in p.parts and p.parent.name == "logs"]
        if not candidates:
            raise ValueError(f"Missing supplied log: {original_name}")
        # Prefer the dedicated experiment archive, preserving any differing copies.
        candidates.sort(key=lambda p: (p.relative_to(source).parts[0] == "AA-MAPPO", str(p)))
        selected = candidates[0]
        hashes = {sha256(p) for p in candidates}
        raw = json.loads(selected.read_text(encoding="utf-8-sig"))
        series, per_agent = normalize_summary(raw)
        archive = destination / "artifacts" / "legacy" / run_id
        archive.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected, archive / "summary.raw.json")
        for model in sorted((selected.parent.parent / "models").glob("*.pt")):
            target = archive / "models" / model.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model, target)
            inventory.append({"path": target.relative_to(destination).as_posix(), "sha256": sha256(target), "bytes": target.stat().st_size, "source": model.relative_to(source).as_posix()})
        source_info = {"path": selected.relative_to(source).as_posix(), "sha256": sha256(selected), "copies_found": len(candidates), "copies_identical": len(hashes) == 1}
        if len(hashes) > 1:
            for index, candidate in enumerate(candidates):
                shutil.copy2(candidate, archive / f"summary.source-{index + 1}.json")
        write_json(archive / "provenance.json", {
            "status": "historical-single-run", "algorithm_label_from_directory": original_name,
            "seed": None, "training_config_complete": False, "source": source_info,
            "notes": ["Folder label is historical metadata, not a verification of the exact training architecture.", "run1 does not identify a random seed.", "Original architecture, all hyperparameters and correspondence to published figures still require author confirmation.", "These checkpoints are not compatible with the maintained entity-token architecture."],
        })
        inventory.append({"path": (archive / "summary.raw.json").relative_to(destination).as_posix(), "sha256": sha256(selected), "bytes": selected.stat().st_size, "source": source_info["path"]})
        runs.append({
            "id": run_id, "label": label, "seed": None, "series": series, "per_agent_series": per_agent,
            "source": source_info, "aggregation": "episode_reward is preserved; agent diagnostics are arithmetic means across agents at the same training step, NOT independent training seeds",
            "notes": "历史单次运行；种子与完整配置未确认；不是维护版验证结果。不同奖励定义的曲线不能直接用来判断算法优劣。",
        })
    data = {
        "schema_version": "1.0", "kind": "legacy-training",
        "provenance": {"status": "historical-single-run", "notes": "来自用户提供的原始summary.json，保留异常值和原始步数。没有多种子置信区间，也没有据此推定通信、定位或安全率。"},
        "runs": runs,
    }
    write_json(destination / "site" / "data" / "legacy-training.json", data)
    screenshots = []
    for image in sorted(source.rglob("*.png")):
        if image.relative_to(source).parts[0] == "AA-MAPPO":
            continue
        if image.stem == "更改命令":
            continue
        name = hashlib.sha256(image.relative_to(source).as_posix().encode("utf-8")).hexdigest()[:10] + "-" + image.name
        target = destination / "site" / "assets" / "historical" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, target)
        screenshots.append({"path": "assets/historical/" + name, "source": image.relative_to(source).as_posix(), "sha256": sha256(image), "status": "historical-figure", "caption": image.parent.name + " / " + image.name})
    write_json(destination / "site" / "data" / "historical-figures.json", {"schema_version": "1.0", "figures": screenshots, "notes": "原始截图；不由图片反推逐步轨迹或安全率。"})
    code_inventory = [{"path": p.relative_to(source).as_posix(), "sha256": sha256(p)} for p in sorted(source.rglob("*.py")) if "__pycache__" not in p.parts]
    write_json(destination / "legacy" / "SOURCE_SNAPSHOT.json", {"source_label": source.name, "status": "supplied-original", "files": code_inventory, "note": "Byte hashes identify supplied files; this is not a claim that they generated every historical run."})
    write_json(destination / "artifacts" / "legacy" / "inventory.json", {"schema_version": "1.0", "artifacts": inventory})
    return {"runs": len(runs), "raw_logs": len(runs), "checkpoints": sum(1 for p in inventory if p["path"].endswith(".pt")), "figures": len(screenshots), "code_files_hashed": len(code_inventory)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Supplied AA-MAPPO_Project directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(import_artifacts(args.source.resolve(), args.out.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()

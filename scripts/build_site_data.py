"""Build a small, provenance-labelled public manifest from existing run outputs."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--pilot-evaluation", type=Path, help="Fixed first-scene replay of the selected exploratory policy")
    parser.add_argument("--pilot-comparison", type=Path, help="Measured comparison JSON; never generated from screenshots")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    runs = []
    if (root / "site/data/demo.json").is_file():
        runs.append({"id": "teaching-demo", "label": "规则控制教学演示", "path": "data/demo.json", "kind": "evaluation", "status": "illustrative", "description": "解释场景、物理指标与回放接口；不是训练后的AA-MAPPO策略。"})
    if args.evaluation:
        evaluation_path = args.evaluation.resolve()
        data = read(evaluation_path)
        if data.get("kind") != "evaluation":
            raise ValueError("Expected evaluation JSON")
        checkpoint = evaluation_path.parent / "checkpoint.pt"
        if checkpoint.is_file():
            data["provenance"]["checkpoint"] = checkpoint.relative_to(root).as_posix()
            data["provenance"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        data["provenance"]["notes"] = "维护版短训练后的实际确定性评估；只验证训练、保存、加载和评估流程，不支持论文级性能结论。回放仅展示第一个评估场景。"
        write(root / "site/data/maintenance-smoke.json", data)
    if (root / "site/data/maintenance-smoke.json").is_file():
        runs.append({"id": "maintenance-smoke", "label": "维护版短训练 · 实际评估回放", "path": "data/maintenance-smoke.json", "kind": "evaluation", "status": "maintenance-smoke", "description": "真实短训练输出，非完整复现实验；与历史权重和原论文结果分开。"})
    if args.pilot_evaluation:
        data = read(args.pilot_evaluation)
        if data.get("kind") != "evaluation" or data.get("provenance", {}).get("status") != "maintenance-pilot":
            raise ValueError("Pilot replay requires kind=evaluation and explicit maintenance-pilot provenance")
        write(root / "site/data/pilot-evaluation.json", data)
    if args.pilot_comparison:
        data = read(args.pilot_comparison)
        if data.get("kind") != "pilot-comparison" or not data.get("rows"):
            raise ValueError("Expected a nonempty measured pilot-comparison")
        write(root / "site/data/pilot-comparison.json", data)
    if (root / "site/data/pilot-evaluation.json").is_file():
        runs.append({"id": "maintenance-pilot", "label": "探索性改进 · 固定测试场景", "path": "data/pilot-evaluation.json", "kind": "evaluation", "status": "maintenance-pilot", "description": "新探索方案的实际轨迹；具体训练方法与预算见来源记录，非原论文结果。默认回放预先指定的首个测试场景，不挑选最好看的轨迹。"})
    manifest = {"schema_version": "1.0", "notes": "已接入四组历史日志及维护版运行；历史单次记录、教学示例、短训练验证与探索性实验分别标注。完整多种子性能实验仍待执行。", "runs": runs,
                "artifacts": [
                    {"title": "四组历史训练日志", "status": "historical-single-run", "path": "data/legacy-training.json", "description": "原始summary逐条转换，保留异常值；每组run1，种子与完整配置待核对。"},
                    {"title": "历史截图档案", "status": "historical-figure", "path": "data/historical-figures.json", "description": "原始图像与源文件校验信息；不由静态图片推断碰撞率。"},
                    {"title": "24份历史Actor/Critic检查点", "status": "historical-artifact", "description": "在仓库artifacts/legacy保留；结构不兼容新的实体注意力网络，不加载为维护版模型。"},
                    {"title": "完整多种子消融", "status": "pending", "description": "提供四种注意力模式与五种子计划；尚未据短训练宣称完整实验通过。"}
                ]}
    if (root / "artifacts/final-test/ring-teacher.json").is_file():
        write(root / "site/data/ring-reference.json", read(root / "artifacts/final-test/ring-teacher.json"))
        runs.append({"id":"ring-reference", "label":"避碰环形规则参考 · 未训练", "path":"data/ring-reference.json", "kind":"evaluation", "status":"maintenance-pilot", "description":"相同环境的非学习控制参照，用于解释可达到的轨迹形态；不是AA-MAPPO训练结果。"})
    manifest["default_run"] = "maintenance-pilot" if any(r["id"] == "maintenance-pilot" for r in runs) else "teaching-demo"
    write(root / "site/data/manifest.json", manifest)
    print(json.dumps({"manifest_runs": len(runs), "legacy_data_present": (root / "site/data/legacy-training.json").is_file()}))


if __name__ == "__main__":
    main()

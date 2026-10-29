"""Render provenance-labelled research figures from actual evaluation/log files.

Optional dependency: python -m pip install -r requirements-figures.txt
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


COLORS = ["#267a9b", "#dc874b", "#478877", "#9066a2", "#bb5963"]


def style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"] if name in available]
    plt.rcParams.update({"font.family": fonts + ["DejaVu Sans"], "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#b9c8c3", "text.color": "#1c373b",
                         "axes.labelcolor": "#1c373b", "axes.titleweight": "bold",
                         "grid.color": "#e1e8e4", "grid.alpha": .8,
                         "axes.unicode_minus": False, "savefig.facecolor": "white",
                         "svg.fonttype": "none"})


def figure_evaluation(data, destination):
    frames = data["trajectory"]
    coordinates = np.asarray([f["positions"] for f in frames])
    times = np.asarray([f["time"] for f in frames])
    size = data["config"]["map_size"]
    provenance = data["provenance"]
    status = provenance.get("status", "evaluation")
    label = {"maintenance-smoke": "短训练验证", "maintenance-pilot": "探索性训练评估",
             "illustrative": "规则控制教学示例", "baseline": "基线评估"}.get(status, "实际评估")
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
    ax = fig.add_subplot(grid[:, 0])
    for i in range(coordinates.shape[1]):
        color = COLORS[i % len(COLORS)]
        xy = coordinates[:, i]
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2, label=f"UAV {i+1}")
        ax.scatter(*xy[0], facecolor="white", edgecolor=color, s=70, lw=1.8, zorder=4)
        ax.scatter(*xy[-1], marker="^", color=color, s=75, zorder=4)
        for frame in range(0, len(frames)-1, max(1, len(frames)//6)):
            end = min(frame+4, len(frames)-1)
            if np.linalg.norm(xy[end]-xy[frame]) > .1:
                ax.annotate("", xy[end], xy[frame], arrowprops={"arrowstyle":"->", "color":color, "lw":1.5})
    users = np.asarray(data["users"])
    ax.scatter(users[:, 0], users[:, 1], marker="*", s=210, color="#b04a40", edgecolor="white", lw=.7, label="地面用户", zorder=5)
    for i, xy in enumerate(users):
        ax.annotate(f"UE {i+1}", xy, xytext=(7, 8), textcoords="offset points", fontsize=10)
    ax.set(xlim=(0, size), ylim=(0, size), xlabel="X / m", ylabel="Y / m", title="同一场景的实际飞行轨迹")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, lw=.7)
    ax.legend(loc="upper left", ncol=2, framealpha=.95, fontsize=9)
    ax.text(.02, .02, "○ 起点   ▲ 终点   → 时间方向\n坐标等比例，无轨迹平滑", transform=ax.transAxes, fontsize=9, color="#57706b")
    for row, key, title, unit in [(0,"communication","通信频谱效率","bit/s/Hz"),
                                  (1,"sensing_proxy","感知 SNR 代理","代理量"),
                                  (2,"min_distance","最小机间距离","m")]:
        plot = fig.add_subplot(grid[row, 1])
        values = [frame["metrics"][key] for frame in frames]
        plot.plot(times, values, color=COLORS[row], lw=2)
        plot.set(title=title, xlabel="时间 / s", ylabel=unit)
        plot.set_xlim(times[0], times[-1])
        plot.grid(True, lw=.7)
        if key == "min_distance":
            threshold = data["config"]["safe_distance"]
            plot.axhline(threshold, color="#b04a40", ls="--", lw=1.3, label=f"阈值 {threshold:g} m")
            plot.set_ylim(bottom=0)
            plot.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{label}  |  训练步数 {provenance.get('trained_environment_steps', 0):,}  |  场景 seed {provenance.get('seed', '未记录')}", fontsize=16)
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def figure_training(path, destination, window):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    steps = [record["environment_steps"] for record in records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, key, title, unit, color in zip(axes, ["mean_step_reward", "communication", "collision_step_fraction"],
            ["采样策略的平均每步奖励", "训练期间通信频谱效率", "训练期间碰撞步比例"],
            ["reward / step", "bit/s/Hz", "fraction"], COLORS):
        values = np.asarray([record[key] for record in records])
        smooth = [values[max(0,i-window+1):i+1].mean() for i in range(len(values))]
        axis.plot(steps, values, color=color, alpha=.25, lw=.8, label="原始记录")
        axis.plot(steps, smooth, color=color, lw=2, label=f"{window}点向后均值")
        axis.set(title=title, xlabel="训练环境步", ylabel=unit)
        axis.grid(True, lw=.7)
        axis.legend(frameon=False, fontsize=9)
        if key == "collision_step_fraction": axis.set_ylim(0, max(.05, float(values.max())*1.1))
    fig.suptitle("单次训练诊断 · 全部记录点保留 · 平滑线不是置信区间或独立测试结果", fontsize=13)
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".svg"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--training-log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()
    if args.window < 1: raise ValueError("window must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    data = json.loads(args.evaluation.read_text(encoding="utf-8"))
    figure_evaluation(data, args.out / "trajectory-and-metrics")
    sources = {str(args.evaluation): hashlib.sha256(args.evaluation.read_bytes()).hexdigest()}
    if args.training_log:
        figure_training(args.training_log, args.out / "training-diagnostics", args.window)
        sources[str(args.training_log)] = hashlib.sha256(args.training_log.read_bytes()).hexdigest()
    (args.out / "figure-provenance.json").write_text(json.dumps({"source_sha256": sources,
        "trajectory_smoothing": False, "diagnostic_moving_average_window": args.window,
        "notes": "Replay only the first evaluation scene; no confidence intervals inferred from a single training run."},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"figures": str(args.out), "sources": len(sources)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

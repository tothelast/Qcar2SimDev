import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def load_results(path: Path) -> List[dict]:
    """Load a JSON list of result dicts.

    Expected format:
      [
        {"annotations": int, "mean_ADE": float, "mean_reward": float, "label": str},
        ...
      ]
    """
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of result objects")
    for item in data:
        for key in ["annotations", "mean_ADE", "mean_reward"]:
            if key not in item:
                raise ValueError(f"Missing key '{key}' in result item: {item}")
    return data


def plot_reward_vs_annotations(results: List[dict], output_path: Path) -> None:
    # Sort by annotation count for a monotonic x-axis
    results_sorted = sorted(results, key=lambda r: r["annotations"])

    x = [r["annotations"] for r in results_sorted]
    y_reward = [r["mean_reward"] for r in results_sorted]
    labels = [r.get("label", "") for r in results_sorted]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y_reward, marker="o")

    for xi, yi, label in zip(x, y_reward, labels):
        if label:
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=8)

    plt.xlabel("Number of annotations (frames)")
    plt.ylabel("Offline reward (exp(-ADE/τ))")
    plt.title("Offline QLabs reward vs annotations")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.axhline(1.0, color="gray", linestyle="--", label="Expert (human)")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved reward vs annotations plot to {output_path}")


def plot_ade_vs_annotations(results: List[dict], output_path: Path) -> None:
    results_sorted = sorted(results, key=lambda r: r["annotations"])

    x = [r["annotations"] for r in results_sorted]
    labels = [r.get("label", "") for r in results_sorted]

    # Primary metric: model predictions vs teleop waypoints (backwards-compatible)
    y_model_vs_teleop = [
        r.get("mean_ADE_model_vs_teleop", r["mean_ADE"]) for r in results_sorted
    ]

    has_model_vs_route = all("mean_ADE_model_vs_route" in r for r in results_sorted)
    has_teleop_vs_route = all("mean_ADE_teleop_vs_route" in r for r in results_sorted)

    plt.figure(figsize=(6, 4))

    # Model vs teleop ADE (current metric)
    plt.plot(
        x,
        y_model_vs_teleop,
        marker="o",
        linestyle="-",
        color="C0",
        label="Model vs teleop (ADE)",
    )

    # Model vs route ADE (route-following metric)
    if has_model_vs_route:
        y_model_vs_route = [r["mean_ADE_model_vs_route"] for r in results_sorted]
        plt.plot(
            x,
            y_model_vs_route,
            marker="s",
            linestyle="--",
            color="C1",
            label="Model vs route (ADE)",
        )

    # Teleop vs route ADE (human route-following baseline)
    if has_teleop_vs_route:
        teleop_vs_route_vals = [r["mean_ADE_teleop_vs_route"] for r in results_sorted]
        human_route_ade = float(sum(teleop_vs_route_vals) / len(teleop_vs_route_vals))
        plt.axhline(
            human_route_ade,
            color="gray",
            linestyle=":",
            label="Teleop vs route (ADE)",
        )

    # Annotate model-vs-teleop points with budget labels
    for xi, yi, label in zip(x, y_model_vs_teleop, labels):
        if label:
            plt.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=8)

    plt.xlabel("Number of annotations (frames)")
    plt.ylabel("Mean ADE (m)")
    plt.title("Offline QLabs ADE vs annotations")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved ADE vs annotations plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot offline QLabs reward vs annotations curve.")
    parser.add_argument("results_json", type=str,
                        help="Path to JSON file with list of results (annotations, mean_ADE, mean_reward, label)")
    parser.add_argument("--out_reward", type=str, default="reward_vs_annotations.png",
                        help="Path to save reward vs annotations plot")
    parser.add_argument("--out_ade", type=str, default="ade_vs_annotations.png",
                        help="Path to save ADE vs annotations plot")

    args = parser.parse_args()

    results = load_results(Path(args.results_json))

    plot_reward_vs_annotations(results, Path(args.out_reward))
    plot_ade_vs_annotations(results, Path(args.out_ade))


if __name__ == "__main__":
    main()


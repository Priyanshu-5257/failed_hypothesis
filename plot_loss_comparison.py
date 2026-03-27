import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_csv_rows(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def aggregate_mean_by_step(rows: List[Dict[str, str]], metric: str) -> Tuple[List[int], List[float]]:
    step_to_values: Dict[int, List[float]] = {}
    for row in rows:
        if metric not in row or row[metric] in {"", None}:
            continue
        step = int(row["global_step"])
        value = float(row[metric])
        step_to_values.setdefault(step, []).append(value)

    steps = sorted(step_to_values.keys())
    means = [sum(step_to_values[s]) / len(step_to_values[s]) for s in steps]
    return steps, means


def aggregate_eval_split_by_step(rows: List[Dict[str, str]], split: str, metric: str) -> Tuple[List[int], List[float]]:
    filtered = [row for row in rows if row.get("split") == split]
    return aggregate_mean_by_step(filtered, metric)


def load_train_metric(output_dir: str, metric: str) -> Tuple[List[int], List[float]]:
    file_path = os.path.join(output_dir, "train_metrics.csv")
    rows = read_csv_rows(file_path)
    return aggregate_mean_by_step(rows, metric)


def load_val_metric(output_dir: str, metric: str) -> Tuple[List[int], List[float]]:
    file_path = os.path.join(output_dir, "eval_metrics.csv")
    rows = read_csv_rows(file_path)
    return aggregate_eval_split_by_step(rows, split="val", metric=metric)


def safe_series(
    output_dir: str,
    series_loader,
    metric: str,
) -> Optional[Tuple[List[int], List[float]]]:
    try:
        steps, values = series_loader(output_dir, metric)
        if not steps:
            return None
        return steps, values
    except (FileNotFoundError, KeyError, ValueError):
        return None


def plot_comparison(
    metric: str,
    split: str,
    vanilla_dir: str,
    ortho_dir: str,
    output_dir: str,
) -> str:
    loader = load_train_metric if split == "train" else load_val_metric

    vanilla_series = safe_series(vanilla_dir, loader, metric)
    ortho_series = safe_series(ortho_dir, loader, metric)

    if vanilla_series is None and ortho_series is None:
        raise ValueError(f"No data available for metric='{metric}' split='{split}'")

    plt.figure(figsize=(9.5, 5.5))

    if vanilla_series is not None:
        v_steps, v_values = vanilla_series
        plt.plot(v_steps, v_values, linewidth=2.0, label="outputs_vanilla")

    if ortho_series is not None:
        o_steps, o_values = ortho_series
        plt.plot(o_steps, o_values, linewidth=2.0, label="outputs_ortho_act")

    metric_title = "Ortho Loss" if metric == "ortho_loss" else "Loss"
    split_title = "Train" if split == "train" else "Val"
    plt.title(f"{split_title} {metric_title}: outputs_vanilla vs outputs_ortho_act")
    plt.xlabel("Global Step")
    plt.ylabel(metric_title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"compare_{split}_{metric}.png"
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare train/val loss and ortho_loss between outputs_vanilla and outputs_ortho_act."
    )
    parser.add_argument(
        "--vanilla-dir",
        type=str,
        default="outputs_vanilla",
        help="Path to vanilla outputs directory (expects train_metrics.csv and eval_metrics.csv).",
    )
    parser.add_argument(
        "--ortho-dir",
        type=str,
        default="outputs_ortho_act",
        help="Path to orthogonal outputs directory (expects train_metrics.csv and eval_metrics.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_plots",
        help="Directory where comparison PNG plots are saved.",
    )
    args = parser.parse_args()

    targets = [
        ("loss", "train"),
        ("loss", "val"),
        ("ortho_loss", "train"),
        ("ortho_loss", "val"),
    ]

    for metric, split in targets:
        try:
            path = plot_comparison(
                metric=metric,
                split=split,
                vanilla_dir=args.vanilla_dir,
                ortho_dir=args.ortho_dir,
                output_dir=args.output_dir,
            )
            print(f"Saved: {path}")
        except ValueError as e:
            print(f"Skipped ({split}, {metric}): {e}")


if __name__ == "__main__":
    main()

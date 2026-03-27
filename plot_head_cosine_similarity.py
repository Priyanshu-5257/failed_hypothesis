import argparse
import csv
import glob
import math
import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_csv_rows(file_path: str) -> List[Dict[str, str]]:
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_token_char(token_char: str) -> str:
    if token_char in {"", "\n", "\r"}:
        return "⏎"
    return token_char


def sample_token_keys(rows: List[Dict[str, str]], sample_size: int, seed: int) -> List[Tuple[str, str]]:
    unique_tokens = sorted({(row["token_id"], row.get("token_char", "")) for row in rows})
    if not unique_tokens:
        return []
    rng = random.Random(seed)
    k = min(sample_size, len(unique_tokens))
    return rng.sample(unique_tokens, k=k)


def extract_cosine_columns(rows: List[Dict[str, str]]) -> List[str]:
    if not rows:
        return []
    cosine_cols = [col for col in rows[0].keys() if col.startswith("cosine_head_")]
    cosine_cols.sort()
    return cosine_cols


def aggregate_token_pair_by_step(
    rows: List[Dict[str, str]],
    token_id: str,
    token_char: str,
    cosine_col: str,
) -> Tuple[List[int], List[float]]:
    step_to_values: Dict[int, List[float]] = {}
    for row in rows:
        if row["token_id"] != token_id or row.get("token_char", "") != token_char:
            continue
        step = int(row["global_step"])
        value = float(row[cosine_col])
        step_to_values.setdefault(step, []).append(value)

    sorted_steps = sorted(step_to_values.keys())
    mean_values = [sum(step_to_values[s]) / len(step_to_values[s]) for s in sorted_steps]
    return sorted_steps, mean_values


def plot_block_cosine_similarity(block_csv_path: str, output_dir: str, sample_size: int, seed: int) -> str:
    rows = read_csv_rows(block_csv_path)
    sampled_tokens = sample_token_keys(rows, sample_size=sample_size, seed=seed)
    cosine_cols = extract_cosine_columns(rows)

    if not rows:
        raise ValueError(f"No rows found in {block_csv_path}")
    if not cosine_cols:
        raise ValueError(f"No cosine_head_* columns found in {block_csv_path}")
    if not sampled_tokens:
        raise ValueError(f"No tokens found in {block_csv_path}")

    num_pairs = len(cosine_cols)
    ncols = min(3, num_pairs)
    nrows = math.ceil(num_pairs / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 4.2 * nrows), sharey=True)
    if num_pairs == 1:
        axes_list = [axes]
    elif nrows == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row_axes in axes for ax in row_axes]

    token_labels = [f"id={token_id} '{normalize_token_char(token_char)}'" for token_id, token_char in sampled_tokens]

    for pair_idx, col in enumerate(cosine_cols):
        ax = axes_list[pair_idx]
        for token_idx, (token_id, token_char) in enumerate(sampled_tokens):
            steps, y = aggregate_token_pair_by_step(rows, token_id, token_char, col)
            if steps:
                ax.plot(
                    steps,
                    y,
                    marker="o",
                    linewidth=1.8,
                    label=token_labels[token_idx],
                )

        pair_name = col.replace("cosine_", "")
        ax.set_title(pair_name)
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True, alpha=0.3)

    for unused_ax in axes_list[num_pairs:]:
        unused_ax.axis("off")

    block_name = os.path.basename(block_csv_path).replace("_head_activations.csv", "")
    fig.suptitle(
        f"{block_name}: Cosine Similarity vs Step (sampled {len(sampled_tokens)} tokens)",
        fontsize=14,
        y=0.995,
    )
    handles, labels = axes_list[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(labels), 3),
            frameon=False,
        )
    fig.tight_layout(rect=[0, 0, 1, 0.86])

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{block_name}_cosine_similarity.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-block cosine similarity plots from head activation CSV files."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="outputs_ortho_act",
        help="Directory containing block*_head_activations.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_ortho_act",
        help="Directory to save PNG plots.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of random tokens to sample for each block plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducible token sampling.",
    )
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "block*_head_activations.csv")
    block_files = sorted(glob.glob(pattern))

    if not block_files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    for block_idx, block_file in enumerate(block_files):
        output_path = plot_block_cosine_similarity(
            block_csv_path=block_file,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            seed=args.seed + block_idx,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

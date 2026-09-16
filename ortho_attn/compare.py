"""Summarize matched runs into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="output directories")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--require-matched-params",
        action="store_true",
        help="Exit non-zero if n_params differs across runs.",
    )
    args = p.parse_args()

    rows = []
    for run in args.runs:
        path = Path(run)
        s = load_summary(path)
        final = s["final_eval"]
        val = final["val"]
        rows.append(
            {
                "run": path.name,
                "ortho": s["ortho"],
                "n_params": s["n_params"],
                "n_embd": s.get("n_embd"),
                "n_head": s.get("n_head"),
                "proj_mode": s.get("proj_mode"),
                "ortho_lambda": s.get("ortho_lambda"),
                "mixer": s.get("mixer"),
                "seconds": s["seconds"],
                "train_loss": final["train"]["loss"],
                "val_loss": val["loss"],
                "val_mean_abs_head_cosine": val["mean_abs_head_cosine"],
                "val_mean_abs_head_cosine_post_proj": val.get(
                    "mean_abs_head_cosine_post_proj"
                ),
                "val_ortho_loss": val["ortho_loss"],
            }
        )
    param_set = {r["n_params"] for r in rows}
    report = {
        "ok": True,
        "matched_param_count": len(param_set) == 1,
        "n_params": sorted(param_set),
        "runs": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.require_matched_params and not report["matched_param_count"]:
        raise SystemExit("param counts differ across runs; comparison is confounded")


if __name__ == "__main__":
    main()

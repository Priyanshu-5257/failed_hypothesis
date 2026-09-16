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
    args = p.parse_args()

    rows = []
    for run in args.runs:
        path = Path(run)
        s = load_summary(path)
        final = s["final_eval"]
        rows.append(
            {
                "run": path.name,
                "ortho": s["ortho"],
                "n_params": s["n_params"],
                "mixer": s.get("mixer"),
                "seconds": s["seconds"],
                "train_loss": final["train"]["loss"],
                "val_loss": final["val"]["loss"],
                "val_mean_abs_head_cosine": final["val"]["mean_abs_head_cosine"],
                "val_ortho_loss": final["val"]["ortho_loss"],
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
    if not report["matched_param_count"]:
        raise SystemExit("param counts differ across runs; comparison is confounded")


if __name__ == "__main__":
    main()

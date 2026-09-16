"""Follow-up jobs: identity proj, thinner width, stronger activation penalty."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

JOBS = [
    {
        "name": "id_none",
        "ortho": "none",
        "proj": "identity",
        "n_embd": 64,
        "ortho_lambda": 0.1,
    },
    {
        "name": "id_act",
        "ortho": "act",
        "proj": "identity",
        "n_embd": 64,
        "ortho_lambda": 0.1,
    },
    {
        "name": "e32_none",
        "ortho": "none",
        "proj": "linear",
        "n_embd": 32,
        "ortho_lambda": 0.1,
    },
    {
        "name": "e32_act",
        "ortho": "act",
        "proj": "linear",
        "n_embd": 32,
        "ortho_lambda": 0.1,
    },
    {
        "name": "lam1_act",
        "ortho": "act",
        "proj": "linear",
        "n_embd": 64,
        "ortho_lambda": 1.0,
    },
]


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("+", *cmd, flush=True)
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(rc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    run_dirs: list[str] = []
    for job in JOBS:
        out = work / job["name"]
        run_dirs.append(str(out))
        run(
            [
                sys.executable,
                "-m",
                "ortho_attn.train",
                "--data",
                args.data,
                "--output-dir",
                str(out),
                "--ortho",
                job["ortho"],
                "--proj",
                job["proj"],
                "--n-embd",
                str(job["n_embd"]),
                "--ortho-lambda",
                str(job["ortho_lambda"]),
                "--max-iters",
                str(args.max_iters),
                "--eval-iters",
                str(args.eval_iters),
                "--eval-interval",
                str(args.eval_interval),
            ]
        )
    report = work / "comparison.json"
    run(
        [
            sys.executable,
            "-m",
            "ortho_attn.compare",
            "--runs",
            *run_dirs,
            "--out",
            str(report),
        ]
    )


if __name__ == "__main__":
    main()

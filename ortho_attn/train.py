"""Train one matched-architecture variant.

Eval sampling uses a dedicated CPU Generator so it does not steal RNG from
the training stream. Variants differ only in the extra loss term.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .data import CharData, get_batch, load_char_data
from .model import GPT, GPTConfig, count_parameters
from .ortho import (
    activation_orthogonality_loss,
    mean_abs_head_cosine,
    value_weight_orthogonality_loss,
)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_lambda(step: int, ortho_lambda: float, burn_in: int) -> float:
    if burn_in <= 0:
        return ortho_lambda
    return ortho_lambda * min(1.0, step / burn_in)


def extra_loss(
    ortho: str,
    model: GPT,
    block_head_outputs: list[torch.Tensor],
    margin: float,
) -> torch.Tensor:
    if ortho == "none":
        return torch.zeros((), device=next(model.parameters()).device)
    if ortho == "act":
        return activation_orthogonality_loss(block_head_outputs, margin=margin)
    if ortho == "wv":
        return value_weight_orthogonality_loss(model, margin=margin)
    raise ValueError(f"unknown ortho variant: {ortho}")


@torch.no_grad()
def evaluate(
    model: GPT,
    data: CharData,
    *,
    step: int,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    eval_seed: int,
    device: torch.device,
    ortho: str,
    margin: float,
    lambda_weight: float,
) -> dict[str, dict[str, float]]:
    model.eval()
    out: dict[str, dict[str, float]] = {}
    for split_name, split_data in (("train", data.train), ("val", data.val)):
        gen = torch.Generator()
        gen.manual_seed(eval_seed + step * 1_000_003 + (0 if split_name == "train" else 1))
        losses = []
        orthos = []
        cosines = []
        for _ in range(eval_iters):
            x, y = get_batch(split_data, batch_size, block_size, device, generator=gen)
            _logits, loss, heads = model(x, y)
            o = extra_loss(ortho, model, heads, margin)
            losses.append(loss.item())
            orthos.append(o.item())
            cosines.append(mean_abs_head_cosine(heads).item())
        mean_loss = sum(losses) / len(losses)
        mean_ortho = sum(orthos) / len(orthos)
        out[split_name] = {
            "loss": mean_loss,
            "ortho_loss": mean_ortho,
            "total_loss": mean_loss + lambda_weight * mean_ortho,
            "mean_abs_head_cosine": sum(cosines) / len(cosines),
        }
    model.train()
    return out


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Controlled orthogonal-head LM run")
    p.add_argument("--data", type=str, default="input.txt")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--ortho", choices=("none", "act", "wv"), default="none")
    p.add_argument("--max-iters", type=int, default=5000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ortho-lambda", type=float, default=0.1)
    p.add_argument("--ortho-margin", type=float, default=0.0)
    p.add_argument("--burn-in", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--eval-seed", type=int, default=2026)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    seed_all(args.seed)
    data = load_char_data(args.data)
    config = GPTConfig(
        vocab_size=data.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    n_params = count_parameters(model)
    print(
        f"ortho={args.ortho} params={n_params} device={device} "
        f"proj={model.blocks[0].sa.proj.in_features}->{model.blocks[0].sa.proj.out_features}",
        flush=True,
    )
    if model.blocks[0].sa.proj.in_features != args.n_embd:
        raise RuntimeError("concat mixer broken: proj.in_features != n_embd")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **vars(args),
        "device": str(device),
        "n_params": n_params,
        "vocab_size": data.vocab_size,
        "mixer": "concat",
        "proj": [args.n_embd, args.n_embd],
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    last_eval: dict[str, dict[str, float]] | None = None
    t0 = time.perf_counter()

    for step in range(args.max_iters):
        lam = current_lambda(step, args.ortho_lambda, args.burn_in)
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            last_eval = evaluate(
                model,
                data,
                step=step,
                batch_size=args.batch_size,
                block_size=args.block_size,
                eval_iters=args.eval_iters,
                eval_seed=args.eval_seed,
                device=device,
                ortho=args.ortho,
                margin=args.ortho_margin,
                lambda_weight=lam,
            )
            for split_name, metrics in last_eval.items():
                eval_rows.append({"global_step": step, "split": split_name, **metrics})
            print(
                f"step {step}: train {last_eval['train']['loss']:.4f} "
                f"val {last_eval['val']['loss']:.4f} "
                f"val_cos {last_eval['val']['mean_abs_head_cosine']:.4f} "
                f"val_ortho {last_eval['val']['ortho_loss']:.4f}",
                flush=True,
            )

        xb, yb = get_batch(data.train, args.batch_size, args.block_size, device)
        _logits, loss, heads = model(xb, yb)
        o = extra_loss(args.ortho, model, heads, args.ortho_margin)
        total = loss + lam * o
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq += p.grad.detach().pow(2).sum().item()
        optimizer.step()
        train_rows.append(
            {
                "global_step": step,
                "loss": loss.item(),
                "ortho_loss": o.item(),
                "total_loss": total.item(),
                "lambda_weight": lam,
                "gradient_magnitude": grad_sq**0.5,
            }
        )

    elapsed = time.perf_counter() - t0
    write_csv(
        out_dir / "train_metrics.csv",
        train_rows,
        ["global_step", "loss", "ortho_loss", "total_loss", "lambda_weight", "gradient_magnitude"],
    )
    write_csv(
        out_dir / "eval_metrics.csv",
        eval_rows,
        ["global_step", "split", "loss", "ortho_loss", "total_loss", "mean_abs_head_cosine"],
    )
    summary = {
        "ok": True,
        "ortho": args.ortho,
        "n_params": n_params,
        "seconds": elapsed,
        "final_eval": last_eval,
        "mixer": "concat",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

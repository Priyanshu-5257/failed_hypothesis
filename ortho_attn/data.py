from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CharData:
    train: torch.Tensor
    val: torch.Tensor
    vocab_size: int
    stoi: dict[str, int]
    itos: dict[int, str]


def load_char_data(path: str | Path, train_frac: float = 0.9) -> CharData:
    text = Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(train_frac * len(data))
    return CharData(
        train=data[:n],
        val=data[n:],
        vocab_size=len(chars),
        stoi=stoi,
        itos=itos,
    )


def get_batch(
    split_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(
        len(split_data) - block_size,
        (batch_size,),
        generator=generator,
    )
    x = torch.stack([split_data[i : i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

"""
training/tiny_pretrain.py — small-scale pretraining of OpenMythos on FineWeb-Edu.

Single-GPU script sized for ~16GB VRAM (RTX 4070/5070-class). No FSDP.
Same checkpoint format as training/3b_fine_web_edu.py, so the resulting
step_*.pt files are loadable by examples/chat_example.py.

Usage:
    python training/tiny_pretrain.py

What you should expect:
    Loss should drop from ~12-13 (random init over a ~200k vocab) toward
    ~5-6 within a few thousand steps. The model is tiny (~50-100M params),
    so generations will be locally coherent but not conversational without
    much more compute and an instruction-tuning stage on top.
"""

import math
import os
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from open_mythos import MythosConfig, OpenMythos
from open_mythos.tokenizer import MythosTokenizer


def tiny_config(vocab_size: int, max_seq_len: int) -> MythosConfig:
    """Tiny config sized to fit on a single ~16GB GPU with bf16 + AdamW state."""
    return MythosConfig(
        vocab_size=vocab_size,
        dim=384,
        n_heads=6,
        n_kv_heads=2,
        max_seq_len=max_seq_len,
        max_loop_iters=4,
        prelude_layers=1,
        coda_layers=1,
        attn_type="gqa",
        n_experts=4,
        n_shared_experts=1,
        n_experts_per_tok=2,
        expert_dim=256,
        lora_rank=8,
        act_threshold=0.99,
    )


class FineWebEduDataset(IterableDataset):
    """Streaming FineWeb-Edu loader — packs documents into fixed seq_len chunks."""

    def __init__(self, encoding, seq_len: int, subset: str):
        self.encoding = encoding
        self.seq_len = seq_len
        self.subset = subset

    def __iter__(self):
        worker = get_worker_info()
        nw = worker.num_workers if worker else 1
        wid = worker.id if worker else 0
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",
            streaming=True,
        ).shard(num_shards=nw, index=wid)

        buf: list[int] = []
        for sample in ds:
            buf.extend(self.encoding.encode(sample["text"]))
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1 :]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    # ------------------------------------------------------------------
    # Hyperparameters — small enough for ~16GB VRAM
    # ------------------------------------------------------------------
    seq_len = 512
    micro_batch = 8
    grad_accum = 4
    total_steps = 5000
    warmup_steps = 200
    lr = 3e-4
    weight_decay = 0.1
    log_every = 10
    ckpt_every = 500
    ckpt_dir = "checkpoints"
    dataset_subset = "sample-10BT"

    encoding = MythosTokenizer()
    vocab_size = encoding.vocab_size
    logger.info(f"Device: {device} | AMP: {amp_dtype} | Vocab: {vocab_size:,}")

    cfg = tiny_config(vocab_size, seq_len)
    model = OpenMythos(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")
    logger.info(
        f"seq_len={seq_len} | micro_batch={micro_batch} | grad_accum={grad_accum} | "
        f"global_batch_tokens={micro_batch * grad_accum * seq_len:,} | "
        f"total_steps={total_steps:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        fused=(device == "cuda"),
    )

    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        if device == "cuda"
        else nullcontext()
    )

    dataset = FineWebEduDataset(encoding, seq_len, dataset_subset)
    loader = DataLoader(dataset, batch_size=micro_batch, num_workers=2, pin_memory=True)

    os.makedirs(ckpt_dir, exist_ok=True)
    model.train()
    data_iter = iter(loader)
    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        cur_lr = get_lr(step, warmup_steps, total_steps, lr, lr * 0.1)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        optimizer.zero_grad()
        loss_acc = 0.0

        for _ in range(grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp_ctx:
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                ) / grad_accum
            loss.backward()
            loss_acc += loss.item()

        gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % log_every == 0:
            dt = time.perf_counter() - t0
            tok_per_sec = micro_batch * grad_accum * seq_len * log_every / dt
            logger.info(
                f"step {step:5d}/{total_steps} | loss {loss_acc:.4f} | "
                f"gnorm {float(gnorm):.2f} | lr {cur_lr:.2e} | "
                f"{tok_per_sec / 1e3:.1f}k tok/s"
            )
            t0 = time.perf_counter()

        if step % ckpt_every == 0:
            path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
            tmp = path + ".tmp"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cfg": cfg,
                    "vocab_size": vocab_size,
                },
                tmp,
            )
            os.replace(tmp, path)
            logger.success(f"Checkpoint saved -> {path}")

    logger.success("Training complete.")


if __name__ == "__main__":
    main()

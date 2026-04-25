"""
examples/chat_example.py — text-in / text-out demo for OpenMythos.

This example supports three paths:

  1. Trained OpenMythos checkpoint:
        python examples/chat_example.py --checkpoint checkpoints/step_0005000.pt

  2. Small pretrained Hugging Face instruct fallback (default when no checkpoint
     is provided). This gives a useful answer out of the box:
        python examples/chat_example.py

  3. Random-init OpenMythos smoke test. This only verifies the pipeline runs and
     will produce gibberish:
        python examples/chat_example.py --random-init
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Windows consoles default to cp1252; tokenizer output is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from open_mythos import MythosConfig, OpenMythos
from open_mythos.tokenizer import DEFAULT_MODEL_ID, MythosTokenizer


DEFAULT_FALLBACK_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def tiny_config(vocab_size: int, max_seq_len: int = 512) -> MythosConfig:
    """Default tiny config; must match training/tiny_pretrain.py for ckpt loads."""
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


def resolve_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def run_mythos(args: argparse.Namespace, device: str) -> None:
    print(f"Device: {device}")
    print(f"Loading tokenizer ({DEFAULT_MODEL_ID})...")
    tok = MythosTokenizer()
    vocab_size = tok.vocab_size
    print(f"Vocab size: {vocab_size:,}")

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise SystemExit(
                f"Checkpoint not found: {args.checkpoint!r}\n"
                "Pass a valid --checkpoint, omit it to use the pretrained HF fallback, "
                "or use --random-init for a smoke test."
            )
        print(f"Loading OpenMythos checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        model = OpenMythos(cfg)
        model.load_state_dict(ckpt["model"])
        print(f"Resumed from step {ckpt['step']}")
    else:
        print(
            "Building tiny RANDOM-INIT OpenMythos model. "
            "This is a smoke test only; output will be gibberish."
        )
        model = OpenMythos(tiny_config(vocab_size))
        cfg = model.cfg

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print(f"\nPrompt   : {args.prompt!r}")
    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit("Tokenizer produced an empty prompt; nothing to generate.")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    if input_ids.shape[1] + args.max_new_tokens > cfg.max_seq_len:
        raise SystemExit(
            f"Prompt ({input_ids.shape[1]}) + max_new_tokens ({args.max_new_tokens}) "
            f"exceeds cfg.max_seq_len ({cfg.max_seq_len}). Shorten one of them."
        )

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            n_loops=args.n_loops,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    completion_ids = out[0, input_ids.shape[1] :].tolist()
    completion = tok.decode(completion_ids)
    full = tok.decode(out[0].tolist())

    print(f"\nResponse : {completion}")
    print(f"\nFull text: {full}")


def run_hf_fallback(args: argparse.Namespace, device: str) -> None:
    dtype = resolve_dtype(device)
    model_id = args.hf_model
    print(f"Device: {device}")
    print(f"Loading pretrained HF fallback model ({model_id})...")
    print(
        "No OpenMythos checkpoint was provided, so this example is using a small "
        "instruction-tuned baseline for a sensible chat response."
    )

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    print(f"\nPrompt   : {args.prompt!r}")
    if tok.chat_template:
        prompt_text = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = args.prompt

    inputs = tok(prompt_text, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
    }
    if args.temperature > 0:
        generate_kwargs.update(
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    with torch.no_grad():
        out = model.generate(**inputs, **generate_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    completion_ids = out[0, prompt_len:]
    completion = tok.decode(completion_ids, skip_special_tokens=True).strip()
    full = f"User: {args.prompt}\nAssistant: {completion}"

    print(f"\nResponse : {completion}")
    print(f"\nFull text: {full}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a step_*.pt produced by tiny_pretrain.py",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use an untrained tiny OpenMythos model as a smoke test",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=DEFAULT_FALLBACK_MODEL,
        help="HF fallback model to use when no OpenMythos checkpoint is provided",
    )
    parser.add_argument("--prompt", type=str, default="Hi how are you?")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.checkpoint or args.random_init:
        run_mythos(args, device)
    else:
        run_hf_fallback(args, device)


if __name__ == "__main__":
    os.system("cls")
    main()

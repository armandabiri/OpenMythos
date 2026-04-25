"""
examples/chat_example.py — interactive terminal chat demo for OpenMythos.

This example supports four paths:

  1. Interactive chat with a trained OpenMythos checkpoint:
        python examples/chat_example.py --checkpoint checkpoints/step_0005000.pt

  2. Interactive chat with a stronger pretrained Hugging Face instruct fallback
     (default when no checkpoint is provided):
        python examples/chat_example.py

  3. Random-init OpenMythos smoke test. This only verifies the pipeline runs
     and will produce gibberish:
        python examples/chat_example.py --random-init

     Specific OpenMythos variant config:
        python examples/chat_example.py --random-init --variant mythos_100b

  4. Single-turn mode:
        python examples/chat_example.py --prompt "Hi how are you?"
"""

import argparse
import os
import re
import sys
from collections.abc import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Windows consoles default to cp1252; tokenizer output is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from open_mythos import (
    MythosConfig,
    OpenMythos,
    mythos_1b,
    mythos_1t,
    mythos_3b,
    mythos_10b,
    mythos_50b,
    mythos_100b,
    mythos_500b,
)
from open_mythos.tokenizer import DEFAULT_MODEL_ID, MythosTokenizer

DEFAULT_FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROJECT_PRIMER = (
    "Project context: OpenMythos is a Python and PyTorch research project that "
    "implements a theoretical recurrent-depth transformer architecture. It includes "
    "a Prelude, a looped Recurrent Block, and a Coda, with MLA (Multi-Latent Attention) "
    "or GQA (Grouped Query Attention), "
    "sparse mixture-of-experts routing, LoRA depth adaptation, and ACT-style halting. "
    "It is not a game setting, lore project, or fantasy world. When answering "
    "questions about OpenMythos internals, prefer this project context over prior knowledge."
)
DEFAULT_SYSTEM_PROMPT = (
    "You are Mythos Chat, a concise and technically strong AI assistant for the "
    "OpenMythos project. Give direct helpful answers with minimal filler. "
    "For code or math, be precise. If the user asks for code, return the code first "
    "and avoid extra explanation unless they ask for it. "
    "If something depends on current real-world facts and you are unsure, say that clearly "
    "instead of guessing. "
    f"{PROJECT_PRIMER}"
)
EXIT_COMMANDS = {"exit", "quit", ":q"}
RESET_COMMANDS = {"/reset", ":reset"}
HELP_COMMANDS = {"/help", ":help"}
Conversation = list[tuple[str, str]]
ChatResponder = Callable[[Conversation, str], tuple[str, str]]
CODE_REQUEST_TERMS = (
    "code",
    "function",
    "script",
    "class",
    "python",
    "javascript",
    "typescript",
    "sql",
    "regex",
    "bash",
    "shell",
    "implement",
    "write a",
    "write an",
    "give me the code",
)
VARIANT_BUILDERS: dict[str, Callable[[], MythosConfig]] = {
    "mythos_1b": mythos_1b,
    "mythos_3b": mythos_3b,
    "mythos_10b": mythos_10b,
    "mythos_50b": mythos_50b,
    "mythos_100b": mythos_100b,
    "mythos_500b": mythos_500b,
    "mythos_1t": mythos_1t,
}
VARIANT_PARAM_HINTS = {
    "mythos_1b": 1_000_000_000,
    "mythos_3b": 3_000_000_000,
    "mythos_10b": 10_000_000_000,
    "mythos_50b": 50_000_000_000,
    "mythos_100b": 100_000_000_000,
    "mythos_500b": 500_000_000_000,
    "mythos_1t": 1_000_000_000_000,
}
BYTES_PER_PARAM = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


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


def dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    return "float32"


def resolve_variant_config(variant: str, vocab_size: int) -> MythosConfig:
    if variant == "tiny":
        return tiny_config(vocab_size)
    try:
        return VARIANT_BUILDERS[variant]()
    except KeyError as exc:
        supported = ", ".join(["tiny", *VARIANT_BUILDERS.keys()])
        raise SystemExit(f"Unknown variant {variant!r}. Supported variants: {supported}") from exc


def ensure_variant_fits_or_explain(
    variant: str,
    device: str,
    model_dtype: torch.dtype,
) -> None:
    if variant == "tiny" or device != "cuda":
        return
    if not torch.cuda.is_available():
        return

    total_vram = torch.cuda.get_device_properties(0).total_memory
    param_hint = VARIANT_PARAM_HINTS.get(variant)
    if param_hint is None:
        return

    requested_dtype = dtype_label(model_dtype)
    estimated_weight_bytes = param_hint * BYTES_PER_PARAM[requested_dtype]
    usable_vram_bytes = int(total_vram * 0.85)

    if estimated_weight_bytes <= usable_vram_bytes:
        return

    gpu_name = torch.cuda.get_device_name(0)
    need_gb = estimated_weight_bytes / (1024**3)
    have_gb = total_vram / (1024**3)
    raise SystemExit(
        f"{variant} cannot be instantiated on this device.\n"
        f"GPU: {gpu_name} ({have_gb:.1f} GiB VRAM)\n"
        f"Estimated weights only ({requested_dtype}): ~{need_gb:.1f} GiB\n"
        "This example does not support tensor parallelism, quantized loading, or CPU offload for "
        "OpenMythos variants yet. Use a much smaller variant, the HF fallback path, or a sharded "
        "multi-GPU loader."
    )


def format_plain_chat(
    history: Conversation,
    user_prompt: str,
    system_prompt: str | None = None,
) -> str:
    lines: list[str] = []
    if system_prompt:
        lines.append(f"System: {system_prompt}")
    for previous_user, previous_assistant in history:
        lines.append(f"User: {previous_user}")
        lines.append(f"Assistant: {previous_assistant}")
    lines.append(f"User: {user_prompt}")
    lines.append("Assistant:")
    return "\n".join(lines)


def clean_completion(text: str) -> str:
    for marker in ("\nUser:", "\nAssistant:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    text = text.strip()
    return text or "[empty response]"


def is_code_request(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(term in lowered for term in CODE_REQUEST_TERMS)


def extract_complete_code_block(text: str) -> str | None:
    match = re.search(r"```[\w+-]*\n.*?\n```", text, flags=re.DOTALL)
    return match.group(0).strip() if match else None


def resolve_context_window(model: AutoModelForCausalLM, tok: AutoTokenizer) -> int | None:
    candidates: list[int] = []
    for value in (
        getattr(model.config, "max_position_embeddings", None),
        getattr(tok, "model_max_length", None),
    ):
        if isinstance(value, int) and 0 < value < 1_000_000:
            candidates.append(value)
    return min(candidates) if candidates else None


def trim_mythos_prompt(
    tok: MythosTokenizer,
    cfg: MythosConfig,
    history: Conversation,
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    active_history = list(history)
    while True:
        prompt_text = format_plain_chat(active_history, user_prompt, system_prompt)
        prompt_ids = tok.encode(prompt_text)
        if not prompt_ids:
            raise SystemExit("Tokenizer produced an empty prompt; nothing to generate.")
        if len(prompt_ids) + max_new_tokens <= cfg.max_seq_len or not active_history:
            return prompt_ids, prompt_text
        active_history = active_history[1:]


def trim_hf_prompt(
    tok: AutoTokenizer,
    history: Conversation,
    user_prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    max_context: int | None,
) -> tuple[dict[str, torch.Tensor], str]:
    active_history = list(history)
    while True:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for previous_user, previous_assistant in active_history:
            messages.append({"role": "user", "content": previous_user})
            messages.append({"role": "assistant", "content": previous_assistant})

        if tok.chat_template:
            prompt_text = tok.apply_chat_template(
                messages + [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = format_plain_chat(active_history, user_prompt, system_prompt)

        inputs = tok(prompt_text, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        if max_context is None or prompt_len + max_new_tokens <= max_context or not active_history:
            return inputs, format_plain_chat(active_history, user_prompt, system_prompt)
        active_history = active_history[1:]


def create_mythos_responder(args: argparse.Namespace, device: str) -> ChatResponder:
    print(f"Device: {device}")
    print(f"Loading tokenizer ({DEFAULT_MODEL_ID})...")
    tok = MythosTokenizer()
    vocab_size = tok.vocab_size
    print(f"Vocab size: {vocab_size:,}")
    model_dtype = resolve_dtype(device)

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
        cfg = resolve_variant_config(args.variant, vocab_size)
        ensure_variant_fits_or_explain(args.variant, device, model_dtype)
        print(
            f"Building RANDOM-INIT OpenMythos model for variant {args.variant}. "
            "This is a smoke test only; output will be gibberish."
        )
        model = OpenMythos(cfg)
        cfg = model.cfg

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    def respond(history: Conversation, user_prompt: str) -> tuple[str, str]:
        prompt_ids, prompt_text = trim_mythos_prompt(
            tok,
            cfg,
            history,
            user_prompt,
            args.system,
            args.max_new_tokens,
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        temperature = args.temperature if args.temperature > 0 else 1.0
        top_k = args.top_k if args.temperature > 0 else 1

        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                n_loops=args.n_loops,
                temperature=temperature,
                top_k=top_k,
            )

        completion_ids = out[0, input_ids.shape[1] :].tolist()
        completion = clean_completion(tok.decode(completion_ids))
        full = f"{prompt_text} {completion}".rstrip()
        return completion, full

    return respond


def create_hf_responder(args: argparse.Namespace, device: str) -> ChatResponder:
    dtype = resolve_dtype(device)
    model_id = args.hf_model
    print(f"Device: {device}")
    print(f"Loading pretrained HF fallback model ({model_id})...")
    print(
        "No OpenMythos checkpoint was provided, so this example is using a stronger "
        "instruction-tuned baseline for smarter chat responses."
    )

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    max_context = resolve_context_window(model, tok)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    def respond(history: Conversation, user_prompt: str) -> tuple[str, str]:
        code_request = is_code_request(user_prompt)
        max_new_tokens = max(args.max_new_tokens, 512) if code_request else args.max_new_tokens
        inputs, prompt_text = trim_hf_prompt(
            tok,
            history,
            user_prompt,
            args.system,
            max_new_tokens,
            max_context,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_token_id,
            "repetition_penalty": args.repetition_penalty,
        }
        if args.temperature > 0:
            generate_kwargs.update(
                do_sample=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        with torch.no_grad():
            out = model.generate(**inputs, **generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        completion_ids = out[0, prompt_len:].tolist()
        completion = clean_completion(tok.decode(completion_ids, skip_special_tokens=True))
        if code_request:
            code_block = extract_complete_code_block(completion)
            if code_block is not None:
                completion = code_block
        full = f"{prompt_text} {completion}".rstrip()
        return completion, full

    return respond


def run_single_turn(respond: ChatResponder, prompt: str) -> None:
    print(f"\nPrompt   : {prompt!r}")
    completion, full = respond([], prompt)
    print(f"\nResponse : {completion}")
    print(f"\nFull text: {full}")


def run_interactive_chat(respond: ChatResponder, initial_prompt: str | None = None) -> None:
    history: Conversation = []
    queued_prompt = initial_prompt

    print("\nInteractive chat ready.")
    print("Commands: `exit` to quit, `/reset` to clear history, `/help` to show commands.")

    while True:
        if queued_prompt is None:
            try:
                user_prompt = input("\nYou       : ").strip()
            except EOFError:
                print("\nExiting chat.")
                return
        else:
            user_prompt = queued_prompt.strip()
            queued_prompt = None
            print(f"\nYou       : {user_prompt}")

        lower_prompt = user_prompt.lower()
        if not user_prompt or lower_prompt in EXIT_COMMANDS:
            print("\nExiting chat.")
            return
        if lower_prompt in RESET_COMMANDS:
            history.clear()
            print("Assistant : Conversation history cleared.")
            continue
        if lower_prompt in HELP_COMMANDS:
            print("Assistant : Commands: `exit`, `/reset`, `/help`.")
            continue

        completion, _full = respond(history, user_prompt)
        print(f"Assistant : {completion}")
        history.append((user_prompt, completion))


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
        help="Use an untrained OpenMythos variant as a smoke test",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tiny",
        help=(
            "OpenMythos variant to use with --random-init. "
            "Supported: tiny, mythos_1b, mythos_3b, mythos_10b, mythos_50b, "
            "mythos_100b, mythos_500b, mythos_1t"
        ),
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default=DEFAULT_FALLBACK_MODEL,
        help="HF fallback model to use when no OpenMythos checkpoint is provided",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used to steer the assistant",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single-turn prompt. If omitted, start an interactive chat session",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep chatting after the initial prompt",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-loops", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    respond = (
        create_mythos_responder(args, device)
        if args.checkpoint or args.random_init
        else create_hf_responder(args, device)
    )

    if args.prompt is None or args.interactive:
        run_interactive_chat(respond, initial_prompt=args.prompt)
    else:
        run_single_turn(respond, args.prompt)


if __name__ == "__main__":
    os.system("cls")
    main()

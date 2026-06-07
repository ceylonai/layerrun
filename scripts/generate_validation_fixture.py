#!/usr/bin/env python3
"""Generate LayerRun validation fixtures from Hugging Face Transformers.

Example:
  python3 scripts/generate_validation_fixture.py \
    --model-dir /path/to/transformers/model \
    --output fixtures/gemma.json \
    --prompt "The capital of France is" \
    --max-new-tokens 4 \
    --top-n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision")
    parser.add_argument("--logit-atol", type=float, default=0.5)
    parser.add_argument("--logit-rtol", type=float, default=0.05)
    return parser.parse_args()


def torch_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype {name}")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output = Path(args.output)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        revision=args.revision,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        revision=args.revision,
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    cases = []
    with torch.no_grad():
        for index, prompt in enumerate(args.prompt):
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(args.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1].float().cpu()
            values, token_ids = torch.topk(logits, k=args.top_n)

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )[0].detach().cpu().tolist()

            prompt_len = input_ids.shape[-1]
            cases.append(
                {
                    "name": f"case_{index}",
                    "prompt": prompt,
                    "token_ids": input_ids[0].detach().cpu().tolist(),
                    "first_token_top_logits": [
                        {"token_id": int(token_id), "logit": float(value)}
                        for token_id, value in zip(token_ids.tolist(), values.tolist())
                    ],
                    "greedy": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": 0.0,
                        "generated_ids": generated[prompt_len:],
                        "mismatches": [],
                    },
                }
            )

    fixture = {
        "reference": {
            "runtime": "huggingface-transformers",
            "model": str(model_dir),
            "dtype": args.dtype,
            "revision": args.revision,
        },
        "top_n": args.top_n,
        "logit_atol": args.logit_atol,
        "logit_rtol": args.logit_rtol,
        "require_top_token_ids": True,
        "cases": cases,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()

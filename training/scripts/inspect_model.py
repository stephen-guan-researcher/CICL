"""Probe a Qwen model to find Linear module names suitable for LoRA targets.

Run after the model finishes downloading; the output guides the
``target_modules`` choice in the QLoRA config.
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path or HF id")
    parser.add_argument("--load-weights", action="store_true",
                        help="Actually load weights (slow). Default: config-only.")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    print("=== Architecture ===")
    print(f"model_type:    {config.model_type}")
    print(f"architectures: {getattr(config, 'architectures', None)}")
    for attr in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                 "intermediate_size", "vocab_size", "max_position_embeddings"):
        if hasattr(config, attr):
            print(f"{attr}: {getattr(config, attr)}")
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        print("--- text_config ---")
        for k, v in text_cfg.to_dict().items():
            if not isinstance(v, (dict, list)):
                print(f"  {k}: {v}")

    if not args.load_weights:
        print("\n(pass --load-weights to load weights and dump module names)")
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",  # avoid GPU OOM during inspection
        low_cpu_mem_usage=True,
    )
    leaf_kinds: Counter[str] = Counter()
    linear_names: Counter[str] = Counter()
    for name, module in model.named_modules():
        cls = type(module).__name__
        leaf_kinds[cls] += 1
        if "Linear" in cls or "Linear4bit" in cls:
            leaf_name = name.rsplit(".", 1)[-1]
            linear_names[leaf_name] += 1

    print("\n=== Top module classes ===")
    for cls, n in leaf_kinds.most_common(20):
        print(f"  {n:>6}  {cls}")

    print("\n=== Linear leaf names (LoRA target candidates) ===")
    for leaf, n in linear_names.most_common():
        print(f"  {n:>6}  {leaf}")


if __name__ == "__main__":
    main()

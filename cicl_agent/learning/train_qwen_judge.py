"""QLoRA SFT entry point for the Qwen causal-judge student.

Trains a Qwen-family LM to reproduce the counterfactual JSON judgments that
Opus (or the deterministic simulator, for pipeline smoke testing) emits for
``(task, context_unit)`` pairs.

Target: Qwen3.5-9B (Qwen3_5ForConditionalGeneration) on 2 x V100-SXM2-16GB
on Alibaba PAI-DSW. The vision tower is loaded but never receives input;
LoRA adapters are attached only to the text decoder Linear modules.

    * 4-bit NF4 quantization (bitsandbytes)
    * fp16 compute (Volta has no bf16)
    * gradient checkpointing
    * paged AdamW 8bit
    * SDPA / eager attention (V100 has no FlashAttention 2)

Smoke run:

    PYTHONPATH=. accelerate launch --num_processes 1 \
        -m cicl_agent.learning.train_qwen_judge \
        --model Qwen/Qwen3.5-9B \
        --train-data training/data/synthetic_smoke/synthetic_train.jsonl \
        --val-data   training/data/synthetic_smoke/synthetic_val.jsonl \
        --output-dir training/outputs/smoke_run_qwen35_9b \
        --max-steps 4 \
        --per-device-batch-size 1 \
        --grad-accum 4 \
        --max-seq-length 1024
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer


# Text-decoder Linear leaf names in Qwen3.5 (verified against the meta-loaded
# model). The vision tower uses different leaf names (qkv / proj /
# linear_fc1 / linear_fc2) so listing the text-decoder names below leaves
# the vision tower un-LoRAed without needing module-path filters.
#
# - full-attention layers (8): q_proj / k_proj / v_proj / o_proj
# - linear-attention layers (24): in_proj_qkv / in_proj_z / in_proj_a /
#   in_proj_b / out_proj
# - all 32 MLPs: gate_proj / up_proj / down_proj
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class Args:
    model: str
    train_data: str
    val_data: str | None
    output_dir: str
    max_seq_length: int = 1024
    per_device_batch_size: int = 1
    grad_accum: int = 8
    learning_rate: float = 1e-4
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    logging_steps: int = 1
    save_steps: int = 50
    eval_steps: int = 50
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_TARGET_MODULES)
    seed: int = 17
    # If True, do not load 4-bit; useful only when bitsandbytes is unavailable.
    no_quant: bool = False


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local path or HF id")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-quant", action="store_true")
    ns = parser.parse_args()
    return Args(
        model=ns.model,
        train_data=ns.train_data,
        val_data=ns.val_data,
        output_dir=ns.output_dir,
        max_seq_length=ns.max_seq_length,
        per_device_batch_size=ns.per_device_batch_size,
        grad_accum=ns.grad_accum,
        learning_rate=ns.learning_rate,
        num_train_epochs=ns.num_train_epochs,
        max_steps=ns.max_steps,
        warmup_ratio=ns.warmup_ratio,
        logging_steps=ns.logging_steps,
        save_steps=ns.save_steps,
        eval_steps=ns.eval_steps,
        lora_rank=ns.lora_rank,
        lora_alpha=ns.lora_alpha,
        lora_dropout=ns.lora_dropout,
        target_modules=tuple(ns.target_modules),
        seed=ns.seed,
        no_quant=ns.no_quant,
    )


def _load_qwen3_5(model_path: str, quant_config: BitsAndBytesConfig | None):
    """Force the multimodal class so the safetensors checkpoint (which has a
    ``model.visual.*`` branch) loads without mismatches.
    """
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    # Always device_map={"":0} — under DDP, each rank's CUDA_VISIBLE_DEVICES is
    # masked to its own physical GPU (see launch_train_rank.sh), so cuda:0 is
    # the only visible device from the rank's perspective.
    kwargs = dict(
        trust_remote_code=False,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",  # V100: no flash-attn 2
        low_cpu_mem_usage=True,
    )
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = {"": 0}
    return Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **kwargs)


def build_model_and_tokenizer(args: Args):
    quant_config = None
    if not args.no_quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # V100: no bf16
            bnb_4bit_use_double_quant=True,
        )

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    architectures = getattr(cfg, "architectures", None) or []
    if any("Qwen3_5" in a for a in architectures):
        model = _load_qwen3_5(args.model, quant_config)
    else:
        from transformers import AutoModelForCausalLM
        kwargs = dict(
            trust_remote_code=False,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
            kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

    model.config.use_cache = False

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
        )
        # prepare_model_for_kbit_training upcasts every non-quantized fp16
        # param to fp32 (embeddings, lm_head, layer norms, vision tower).
        # On Qwen3.5-9B that's ~4 GB extra per rank — V100-16GB OOMs at LoRA
        # injection. Recast big tensors back to fp16, keep small (<100k)
        # layer-norm params in fp32 so loss is stable in mixed precision.
        recast_bytes = 0
        kept_ln_bytes = 0
        for name, p in model.named_parameters():
            if p.dtype != torch.float32:
                continue
            is_ln = (
                "norm" in name.lower()
                or "layernorm" in name.lower()
                or "rmsnorm" in name.lower()
            )
            if is_ln and p.numel() < 100_000:
                kept_ln_bytes += p.numel() * 4
                continue
            recast_bytes += p.numel() * 4
            p.data = p.data.to(torch.float16)
        print(
            f"[train_qwen_judge] post-prepare fp16 recast: "
            f"recast={recast_bytes / 2**30:.2f} GiB, "
            f"kept LN in fp32={kept_ln_bytes / 2**20:.2f} MiB"
        )
    else:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def build_lora_config(args: Args) -> LoraConfig:
    return LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_chat_dataset(args: Args):
    """Load JSONL with a 'messages' column. SFTTrainer applies the chat
    template automatically."""
    data_files = {"train": args.train_data}
    if args.val_data and Path(args.val_data).exists():
        data_files["validation"] = args.val_data
    return load_dataset("json", data_files=data_files)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Each DDP rank's CUDA_VISIBLE_DEVICES is masked to its own physical GPU
    # by launch_train_rank.sh, so cuda:0 from the rank's view is the rank's
    # dedicated device. Required to make bitsandbytes 4-bit place weights
    # correctly under DDP.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"[train_qwen_judge] model={args.model}")
    print(f"[train_qwen_judge] train_data={args.train_data}")
    print(f"[train_qwen_judge] output_dir={args.output_dir}")
    print(f"[train_qwen_judge] cuda available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()} local_rank={local_rank} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    model, tokenizer = build_model_and_tokenizer(args)
    peft_config = build_lora_config(args)
    dataset = load_chat_dataset(args)

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if "validation" in dataset else "no",
        save_strategy="steps",
        save_total_limit=2,
        fp16=True,  # V100: bf16 unsupported
        bf16=False,
        optim="paged_adamw_8bit" if not args.no_quant else "adamw_torch",
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        max_length=args.max_seq_length,
        packing=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    print("[train_qwen_judge] training done:", json.dumps(metrics, indent=2, ensure_ascii=False))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (Path(args.output_dir) / "train_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

"""Field-level eval for a trained Qwen judge adapter against Opus gold.

For each val example, render the (system, user) prompt, generate greedy with
the LoRA adapter applied, parse the assistant JSON, compare field-by-field to
the gold assistant message. Reports:

  - parse_rate: fraction of generations that yield valid JSON
  - field_coverage: fraction with all 8 expected fields present
  - per-field numeric MAE (action_shift, necessity, expected_outcome_uplift,
    negative_transfer_risk, confidence)
  - per-field exact-match (no_context_action, with_context_action)
  - reason: char-length ratio + non-empty rate

Mirrors the recast logic in train_qwen_judge.py so memory fits 16 GB.

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python -m \
        training.scripts.eval_qwen_judge \
        --base Qwen/Qwen3.5-9B \
        --adapter XinyuGuan/CICL \
        --val training/data/opus_v1/val.jsonl \
        --output artifacts/outputs/latest/qwen_local_eval/eval_field_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig


NUMERIC_FIELDS = (
    "action_shift",
    "necessity",
    "expected_outcome_uplift",
    "negative_transfer_risk",
    "confidence",
)
STRING_FIELDS = (
    "no_context_action",
    "with_context_action",
)
ALL_FIELDS = NUMERIC_FIELDS + STRING_FIELDS + ("reason",)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Base Qwen3.5-9B local path or HF id")
    p.add_argument("--adapter", required=True, help="LoRA adapter local directory or HF id")
    p.add_argument("--val", required=True, help="val.jsonl with messages")
    p.add_argument("--output", required=True, help="Where to write report JSON")
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--max-input-tokens", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=-1, help="Eval first N (debug)")
    p.add_argument("--no-quant", action="store_true")
    return p.parse_args()


def load_model(base_path: str, adapter_path: str, no_quant: bool):
    quant_config = None
    if not no_quant:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    cfg = AutoConfig.from_pretrained(base_path, trust_remote_code=False)
    architectures = getattr(cfg, "architectures", None) or []

    kwargs: dict[str, Any] = dict(
        trust_remote_code=False,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = {"": 0}

    if any("Qwen3_5" in a for a in architectures):
        from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

        base = Qwen3_5ForConditionalGeneration.from_pretrained(base_path, **kwargs)
    else:
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(base_path, **kwargs)

    # Same fp16 recast as training, so the 16 GB budget holds for inference.
    if quant_config is not None:
        for name, p in base.named_parameters():
            if p.dtype != torch.float32:
                continue
            is_ln = (
                "norm" in name.lower()
                or "layernorm" in name.lower()
                or "rmsnorm" in name.lower()
            )
            if is_ln and p.numel() < 100_000:
                continue
            p.data = p.data.to(torch.float16)

    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgment(text: str) -> dict | None:
    """Extract the first balanced JSON object from generated text."""
    # Strip markdown code fence
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[-1]
        if text.startswith("json"):
            text = text[4:]
    match = _JSON_OBJ_RE.search(text)
    if not match:
        return None
    raw = match.group(0)
    # Greedy match may overshoot — try shrinking until parseable
    while raw:
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
            return None
        except json.JSONDecodeError as e:
            # Try truncating to the last `}` before the failing position
            if e.pos <= 0:
                return None
            cut = raw.rfind("}", 0, e.pos)
            if cut <= 0:
                return None
            raw = raw[: cut + 1]
    return None


@dataclass
class FieldStats:
    numeric_abs_errors: dict[str, list[float]] = dc_field(default_factory=dict)
    string_exact: dict[str, list[int]] = dc_field(default_factory=dict)
    reason_pred_lens: list[int] = dc_field(default_factory=list)
    reason_gold_lens: list[int] = dc_field(default_factory=list)
    reason_nonempty: list[int] = dc_field(default_factory=list)
    missing_fields: dict[str, int] = dc_field(default_factory=dict)

    def add(self, pred: dict, gold: dict) -> None:
        for fld in NUMERIC_FIELDS:
            self.numeric_abs_errors.setdefault(fld, [])
            if fld not in pred:
                self.missing_fields[fld] = self.missing_fields.get(fld, 0) + 1
                continue
            try:
                self.numeric_abs_errors[fld].append(abs(float(pred[fld]) - float(gold[fld])))
            except (TypeError, ValueError):
                self.missing_fields[fld] = self.missing_fields.get(fld, 0) + 1
        for fld in STRING_FIELDS:
            self.string_exact.setdefault(fld, [])
            if fld not in pred:
                self.missing_fields[fld] = self.missing_fields.get(fld, 0) + 1
                continue
            self.string_exact[fld].append(1 if str(pred[fld]).strip() == str(gold[fld]).strip() else 0)
        if "reason" in pred:
            r = str(pred.get("reason", ""))
            g = str(gold.get("reason", ""))
            self.reason_pred_lens.append(len(r))
            self.reason_gold_lens.append(len(g))
            self.reason_nonempty.append(1 if r.strip() else 0)
        else:
            self.missing_fields["reason"] = self.missing_fields.get("reason", 0) + 1

    def summary(self, n_parsed: int) -> dict:
        out: dict = {"n_parsed": n_parsed}
        for fld in NUMERIC_FIELDS:
            errs = self.numeric_abs_errors.get(fld, [])
            if errs:
                errs_t = torch.tensor(errs)
                out[fld] = {
                    "mae": float(errs_t.mean()),
                    "p50": float(errs_t.median()),
                    "p90": float(errs_t.quantile(0.9)),
                    "max": float(errs_t.max()),
                    "n": len(errs),
                }
        for fld in STRING_FIELDS:
            ex = self.string_exact.get(fld, [])
            if ex:
                out[fld] = {"exact_match": sum(ex) / len(ex), "n": len(ex)}
        if self.reason_pred_lens:
            pred_t = torch.tensor(self.reason_pred_lens, dtype=torch.float32)
            gold_t = torch.tensor(self.reason_gold_lens, dtype=torch.float32)
            out["reason"] = {
                "nonempty_rate": sum(self.reason_nonempty) / len(self.reason_nonempty),
                "pred_char_mean": float(pred_t.mean()),
                "gold_char_mean": float(gold_t.mean()),
                "len_ratio_mean": float((pred_t / gold_t.clamp(min=1)).mean()),
                "n": len(self.reason_pred_lens),
            }
        if self.missing_fields:
            out["missing_field_counts"] = self.missing_fields
        return out


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[eval] base={args.base}")
    print(f"[eval] adapter={args.adapter}")
    print(f"[eval] val={args.val}")

    model, tokenizer = load_model(args.base, args.adapter, args.no_quant)

    with open(args.val) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        examples = examples[: args.limit]
    print(f"[eval] n_examples={len(examples)}")

    stats = FieldStats()
    parse_failures: list[dict] = []
    per_example_records: list[dict] = []
    n_parsed = 0

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Pre-render prompts and parse golds once.
    prompts: list[str] = []
    golds: list[dict | None] = []
    for i, ex in enumerate(examples):
        prompt_msgs = ex["messages"][:-1]
        gold_text = ex["messages"][-1]["content"]
        golds.append(parse_judgment(gold_text))
        # enable_thinking=False — model was SFT'd on raw JSON assistants, so we
        # suppress Qwen3.5's default reasoning preamble or the model burns its
        # token budget on chain-of-thought and never reaches the JSON.
        prompts.append(
            tokenizer.apply_chat_template(
                prompt_msgs,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        )

    t0 = time.time()
    for batch_start in range(0, len(examples), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(examples))
        batch_prompts = prompts[batch_start:batch_end]

        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
            add_special_tokens=False,
        )
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)

        with torch.inference_mode():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        gen_tokens = out[:, input_ids.shape[-1]:]
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for j, gen_text in enumerate(gen_texts):
            i = batch_start + j
            ex = examples[i]
            gold = golds[i]
            if gold is None:
                parse_failures.append({"i": i, "phase": "gold", "task_id": ex.get("task_id")})
                continue
            pred = parse_judgment(gen_text)
            record = {
                "i": i,
                "task_id": ex.get("task_id"),
                "context_id": ex.get("context_id"),
                "gen_text_head": gen_text[:240],
                "pred": pred,
                "gold": gold,
            }
            per_example_records.append(record)
            if pred is None:
                parse_failures.append({
                    "i": i, "phase": "pred", "task_id": ex.get("task_id"),
                    "gen_text_head": gen_text[:240],
                })
                continue
            n_parsed += 1
            stats.add(pred, gold)

        elapsed = time.time() - t0
        done = batch_end
        rate = done / max(elapsed, 1e-6)
        eta = (len(examples) - done) / max(rate, 1e-6)
        print(f"[eval] {done}/{len(examples)}  parsed={n_parsed}  rate={rate:.2f} ex/s  ETA={eta:.0f}s")

    summary = stats.summary(n_parsed)
    report = {
        "adapter": args.adapter,
        "val": args.val,
        "n_examples": len(examples),
        "n_parsed": n_parsed,
        "parse_rate": n_parsed / len(examples) if examples else 0.0,
        "n_parse_failures": len(parse_failures),
        "elapsed_sec": time.time() - t0,
        "field_stats": summary,
        "parse_failures_head": parse_failures[:10],
    }

    print("[eval] report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    per_example_path = output_path.with_suffix(".per_example.jsonl")
    with per_example_path.open("w", encoding="utf-8") as f:
        for r in per_example_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[eval] wrote {output_path} and {per_example_path}")


if __name__ == "__main__":
    main()

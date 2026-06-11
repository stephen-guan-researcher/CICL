"""Structural quality audit for causal memory cards."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from cicl_agent.core.io import read_jsonl


REQUIRED_FIELDS = ("trigger", "evidence", "action_hint", "failure_if_ignored", "scope")
ACTION_VERBS = (
    "inspect",
    "read",
    "open",
    "search",
    "check",
    "update",
    "fix",
    "call",
    "run",
    "use",
    "compare",
    "look",
)
PLACEHOLDER_VALUES = {"todo", "tbd", "unknown", "not available", "n/a", "none", ""}
PLACEHOLDER_PREFIX = re.compile(r"^(todo|tbd|unknown|not available|n/a|none)\b", re.IGNORECASE)


@dataclass(slots=True)
class CardQualityRow:
    source: str
    cards: int
    status: str
    required_field_rate: float
    actionable_hint_rate: float
    compression_success_rate: float
    placeholder_rate: float
    avg_compression_ratio: float
    avg_compressed_tokens: float
    avg_original_tokens: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "source": self.source,
            "cards": self.cards,
            "status": self.status,
            "required_field_rate": round(self.required_field_rate, 6),
            "actionable_hint_rate": round(self.actionable_hint_rate, 6),
            "compression_success_rate": round(self.compression_success_rate, 6),
            "placeholder_rate": round(self.placeholder_rate, 6),
            "avg_compression_ratio": round(self.avg_compression_ratio, 6),
            "avg_compressed_tokens": round(self.avg_compressed_tokens, 3),
            "avg_original_tokens": round(self.avg_original_tokens, 3),
        }


def audit_card_file(path: str | Path) -> CardQualityRow:
    path = Path(path)
    rows = list(read_jsonl(path))
    if not rows:
        return CardQualityRow(str(path), 0, "FAIL", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    required = [_has_required_fields(row) for row in rows]
    actionable = [_has_actionable_hint(row) for row in rows]
    compressed = [_is_compressed(row) for row in rows]
    placeholders = [_has_placeholder(row) for row in rows]
    ratios = [_compression_ratio(row) for row in rows]
    compressed_tokens = [float(row.get("compressed_token_cost") or 0.0) for row in rows]
    original_tokens = [float(row.get("original_token_cost") or 0.0) for row in rows]

    required_rate = _rate(required)
    actionable_rate = _rate(actionable)
    compression_rate = _rate(compressed)
    placeholder_rate = _rate(placeholders)
    avg_ratio = _avg(ratios)
    status = _status(required_rate, actionable_rate, compression_rate, placeholder_rate, avg_ratio)
    return CardQualityRow(
        source=str(path),
        cards=len(rows),
        status=status,
        required_field_rate=required_rate,
        actionable_hint_rate=actionable_rate,
        compression_success_rate=compression_rate,
        placeholder_rate=placeholder_rate,
        avg_compression_ratio=avg_ratio,
        avg_compressed_tokens=_avg(compressed_tokens),
        avg_original_tokens=_avg(original_tokens),
    )


def audit_card_files(paths: list[str | Path]) -> list[CardQualityRow]:
    return [audit_card_file(path) for path in paths]


def write_csv_report(rows: list[CardQualityRow], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(CardQualityRow("", 0, "", 0, 0, 0, 0, 0, 0, 0).to_dict().keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return output_path


def write_markdown_report(rows: list[CardQualityRow], output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall = "PASS" if rows and all(row.status == "PASS" for row in rows) else "WARN"
    if any(row.status == "FAIL" for row in rows):
        overall = "FAIL"
    lines = [
        "# Causal Memory Card 质量审计",
        "",
        f"Overall status: **{overall}**",
        "",
        "这是一个确定性的结构性审计。它不能替代人工或 LLM 语义质量评审，但可以验证生成的 causal memory cards 是否字段完整、是否压缩有效、是否足够行动导向，从而作为可复现实验包的一部分被引用。",
        "",
        "| 来源 | Cards | 状态 | 必需字段 | 动作提示 | 已压缩 | 占位符 | 平均压缩比 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.source}`",
                    str(row.cards),
                    row.status,
                    f"{row.required_field_rate:.3f}",
                    f"{row.actionable_hint_rate:.3f}",
                    f"{row.compression_success_rate:.3f}",
                    f"{row.placeholder_rate:.3f}",
                    f"{row.avg_compression_ratio:.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- `必需字段` 检查 trigger、evidence、action hint、failure-if-ignored 和 scope 是否非空。",
            "- `动作提示` 检查 action hint 是否包含 inspect、read、search、update、run、use 等明确下一步动作词。",
            "- `已压缩` 检查 compressed token count 是否小于原始 context token count。",
            "- `平均压缩比` 是 compressed tokens / original tokens，越低表示越短。",
            "- 这是保守证据：它支持 artifact hygiene，不证明每张 card 的语义最优性。",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _has_required_fields(row: dict) -> bool:
    return all(str(row.get(field, "")).strip() for field in REQUIRED_FIELDS)


def _has_actionable_hint(row: dict) -> bool:
    hint = str(row.get("action_hint", "")).lower()
    return any(verb in hint for verb in ACTION_VERBS)


def _is_compressed(row: dict) -> bool:
    original = float(row.get("original_token_cost") or 0.0)
    compressed = float(row.get("compressed_token_cost") or 0.0)
    return original > 0 and 0 < compressed < original


def _has_placeholder(row: dict) -> bool:
    for field in REQUIRED_FIELDS:
        value = str(row.get(field, "")).strip()
        normalized = value.lower().strip(" .:-")
        if normalized in PLACEHOLDER_VALUES or bool(PLACEHOLDER_PREFIX.search(value)):
            return True
    return False


def _compression_ratio(row: dict) -> float:
    original = float(row.get("original_token_cost") or 0.0)
    compressed = float(row.get("compressed_token_cost") or 0.0)
    if original <= 0 or compressed <= 0:
        return 0.0
    return compressed / original


def _status(
    required_rate: float,
    actionable_rate: float,
    compression_rate: float,
    placeholder_rate: float,
    avg_ratio: float,
) -> str:
    if required_rate < 0.98:
        return "FAIL"
    if actionable_rate >= 0.80 and compression_rate >= 0.70 and placeholder_rate <= 0.02 and avg_ratio < 0.80:
        return "PASS"
    return "WARN"


def _rate(values: list[bool]) -> float:
    return sum(1 for value in values if value) / len(values) if values else 0.0


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit causal memory-card structural quality.")
    parser.add_argument("cards", nargs="+", help="causal_memory_cards.jsonl files to audit")
    parser.add_argument("--output", default="reports/card_quality_audit.md")
    parser.add_argument("--csv-output", default="reports/card_quality_audit.csv")
    args = parser.parse_args()

    rows = audit_card_files(args.cards)
    write_markdown_report(rows, args.output)
    write_csv_report(rows, args.csv_output)
    print(args.output)
    print(args.csv_output)


if __name__ == "__main__":
    main()

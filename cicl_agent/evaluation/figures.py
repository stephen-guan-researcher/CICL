"""Shared SVG drawing primitives for CICL reports."""

from __future__ import annotations

import html
from collections import defaultdict
from pathlib import Path
from statistics import mean


METHOD_COLORS: dict[str, str] = {
    "CICL": "#0f766e",
    "LLM-CICL": "#9333ea",
    "CICL_Distilled": "#0e7490",
    "LearnedCICL": "#0891b2",
    "VanillaRAG": "#2563eb",
    "AutoContextKG": "#f97316",
    "GraphMemory": "#7c3aed",
    "FullContext": "#64748b",
    "NoContext": "#111827",
    "OracleGoldContext": "#be123c",
    "CICL_full": "#0f766e",
    "CICL_remove_top": "#dc2626",
    "CICL_remove_random": "#2563eb",
}


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.0f}" y="32" text-anchor="middle" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>',
    ]


def axis_svg(x: int, y: int, width: int, height: int, ylabel: str, max_value: float, min_value: float = 0.0) -> str:
    parts: list[str] = []
    value_range = max(1e-9, max_value - min_value)
    for idx in range(6):
        value = min_value + value_range * idx / 5
        yy = y + height - height * idx / 5
        parts.append(f'<line x1="{x}" y1="{yy:.2f}" x2="{x + width}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{x - 8}" y="{yy + 4:.2f}" text-anchor="end" font-size="11" fill="#6b7280">{value:.2g}</text>')
    parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#111827"/>')
    parts.append(f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#111827"/>')
    parts.append(
        f'<text x="20" y="{y + height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 20,{y + height / 2:.2f})" font-size="13" fill="#374151">{html.escape(ylabel)}</text>'
    )
    return "\n".join(parts)


def legend_svg(x: int, y: int, items: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for idx, (label, color) in enumerate(items):
        yy = y + idx * 22
        parts.append(f'<rect x="{x}" y="{yy - 11}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{x + 20}" y="{yy}" font-size="12" fill="#374151">{html.escape(label)}</text>')
    return "\n".join(parts)


def _pick_color(label: str, default: str = "#2563eb") -> str:
    if label in METHOD_COLORS:
        return METHOD_COLORS[label]
    if "Oracle" in label:
        return "#7c3aed"
    if label == "NoContext":
        return "#64748b"
    if label == "CICL":
        return "#0f766e"
    return default


def svg_bar_chart(
    path: str | Path,
    rows: list[dict],
    label_key: str,
    value_key: str,
    title: str,
    ylabel: str,
) -> None:
    data = [(str(row[label_key]), float(row[value_key])) for row in rows]
    width = max(760, 86 * len(data) + 180)
    height = 520
    margin_left = 80
    margin_bottom = 150
    plot_width = width - margin_left - 50
    plot_height = height - 90 - margin_bottom
    values = [value for _, value in data]
    min_value = min(values + [0.0])
    max_value = max(values + [1.0])
    value_range = max(1e-9, max_value - min_value)
    zero_y = 60 + plot_height - ((0.0 - min_value) / value_range) * plot_height
    bar_width = plot_width / max(1, len(data)) * 0.68
    gap = plot_width / max(1, len(data))

    parts = svg_header(width, height, title)
    parts.append(axis_svg(margin_left, 60, plot_width, plot_height, ylabel, max_value, min_value))
    if min_value < 0 < max_value:
        parts.append(
            f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{margin_left + plot_width}" y2="{zero_y:.2f}" '
            'stroke="#374151" stroke-dasharray="3 3"/>'
        )
    for idx, (label, value) in enumerate(data):
        x = margin_left + idx * gap + (gap - bar_width) / 2
        value_y = 60 + plot_height - ((value - min_value) / value_range) * plot_height
        y = min(value_y, zero_y)
        bar_height = abs(zero_y - value_y)
        color = _pick_color(label)
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" rx="3"/>')
        value_label_y = y - 6 if value >= 0 else y + bar_height + 16
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{value_label_y:.2f}" '
            f'text-anchor="middle" font-size="12" fill="#111827">{value:.3g}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{60 + plot_height + 18:.2f}" '
            f'text-anchor="end" transform="rotate(-38 {x + bar_width / 2:.2f},{60 + plot_height + 18:.2f})" '
            f'font-size="12" fill="#374151">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def svg_grouped_precision_recall(path: str | Path, rows: list[dict]) -> None:
    data = [(str(row["method"]), float(row["precision"]), float(row["recall"])) for row in rows]
    width = max(800, 94 * len(data) + 180)
    height = 540
    margin_left = 80
    margin_bottom = 160
    plot_width = width - margin_left - 50
    plot_height = height - 90 - margin_bottom
    parts = svg_header(width, height, "Context Precision and Recall")
    parts.append(axis_svg(margin_left, 60, plot_width, plot_height, "Score", 1.0))
    group_gap = plot_width / max(1, len(data))
    bar_width = group_gap * 0.28
    for idx, (label, precision, recall) in enumerate(data):
        center = margin_left + idx * group_gap + group_gap / 2
        for offset, value, color in [(-bar_width / 1.8, precision, "#2563eb"), (bar_width / 1.8, recall, "#f97316")]:
            x = center + offset - bar_width / 2
            bar_height = value * plot_height
            y = 60 + plot_height - bar_height
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}" rx="3"/>')
        parts.append(
            f'<text x="{center:.2f}" y="{60 + plot_height + 18:.2f}" text-anchor="end" '
            f'transform="rotate(-38 {center:.2f},{60 + plot_height + 18:.2f})" font-size="12" fill="#374151">{html.escape(label)}</text>'
        )
    parts.append(legend_svg(width - 220, 70, [("Precision", "#2563eb"), ("Recall", "#f97316")]))
    parts.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def svg_causal_components(path: str | Path, scores: list[dict]) -> None:
    if not scores:
        Path(path).write_text("", encoding="utf-8")
        return
    metrics = [
        ("action_delta", "ActionDelta", "#2563eb"),
        ("outcome_uplift", "OutcomeUplift", "#0f766e"),
        ("necessity_score", "Necessity", "#f97316"),
        ("cost_penalty", "CostPenalty", "#64748b"),
        ("final_score", "Final", "#7c3aed"),
    ]
    rows = [
        {"metric": label, "value": mean(float(score[key]) for score in scores), "color": color}
        for key, label, color in metrics
    ]
    width = 760
    height = 440
    margin_left = 90
    margin_bottom = 90
    plot_width = width - margin_left - 60
    plot_height = height - 90 - margin_bottom
    max_value = max([row["value"] for row in rows] + [1.0])
    parts = svg_header(width, height, "Mean Causal Score Components")
    parts.append(axis_svg(margin_left, 60, plot_width, plot_height, "Mean Score", max_value))
    gap = plot_width / len(rows)
    bar_width = gap * 0.58
    for idx, row in enumerate(rows):
        x = margin_left + idx * gap + (gap - bar_width) / 2
        bar_height = row["value"] / max_value * plot_height
        y = 60 + plot_height - bar_height
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{row["color"]}" rx="3"/>')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="12" fill="#111827">{row["value"]:.3g}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{60 + plot_height + 24:.2f}" text-anchor="middle" font-size="12" fill="#374151">{row["metric"]}</text>')
    parts.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def svg_line_chart(
    path: str | Path,
    rows: list[dict],
    x_key: str,
    y_key: str,
    group_key: str,
    title: str,
    ylabel: str,
) -> None:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append((float(row[x_key]), float(row[y_key])))
    for points in groups.values():
        points.sort()
    x_values = [x for points in groups.values() for x, _ in points]
    y_values = [y for points in groups.values() for _, y in points]
    if not x_values or not y_values:
        Path(path).write_text("", encoding="utf-8")
        return
    width, height = 860, 520
    left, top, plot_w, plot_h = 82, 60, 700, 330
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min([0.0] + y_values), max([1.0] + y_values)
    x_range = max(1e-9, max_x - min_x)
    y_range = max(1e-9, max_y - min_y)

    def sx(x: float) -> float:
        return left + ((x - min_x) / x_range) * plot_w

    def sy(y: float) -> float:
        return top + plot_h - ((y - min_y) / y_range) * plot_h

    parts = svg_header(width, height, title)
    parts.append(axis_svg(left, top, plot_w, plot_h, ylabel, max_y, min_y))
    for name, points in groups.items():
        color = METHOD_COLORS.get(name, "#2563eb")
        poly = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="{color}"/>')
    parts.append(legend_svg(630, 80, [(name, METHOD_COLORS.get(name, "#2563eb")) for name in groups]))
    parts.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def svg_simple_bar_chart(
    path: str | Path,
    rows: list[dict],
    label_key: str,
    value_key: str,
    title: str,
    ylabel: str,
    label_strip_prefix: str | None = None,
) -> None:
    width, height = 760, 460
    left, top, plot_w, plot_h = 80, 60, 600, 280
    max_y = max([float(row[value_key]) for row in rows] + [1.0])
    parts = svg_header(width, height, title)
    parts.append(axis_svg(left, top, plot_w, plot_h, ylabel, max_y))
    gap = plot_w / max(1, len(rows))
    bar_w = gap * 0.58
    for idx, row in enumerate(rows):
        label = str(row[label_key])
        value = float(row[value_key])
        x = left + idx * gap + (gap - bar_w) / 2
        h = value / max_y * plot_h
        y = top + plot_h - h
        color = METHOD_COLORS.get(label, "#2563eb")
        display_label = label.replace(label_strip_prefix, "") if label_strip_prefix else label
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="12">{value:.3g}</text>')
        parts.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 24:.2f}" text-anchor="middle" font-size="12">{html.escape(display_label)}</text>'
        )
    parts.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(parts), encoding="utf-8")


__all__ = [
    "METHOD_COLORS",
    "axis_svg",
    "legend_svg",
    "svg_bar_chart",
    "svg_causal_components",
    "svg_grouped_precision_recall",
    "svg_header",
    "svg_line_chart",
    "svg_simple_bar_chart",
]

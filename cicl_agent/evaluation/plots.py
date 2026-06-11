"""Optional plotting utilities for publication-style figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_metrics(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_metrics(metrics_csv: str | Path, output_dir: str | Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install the optional 'plots' dependency to render figures.") from exc

    rows = read_metrics(metrics_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures: list[Path] = []

    for metric, ylabel, filename in [
        ("success_rate", "Success Rate", "success_rate_by_method.png"),
        ("context_f1", "Context F1", "context_f1_by_method.png"),
        ("avg_tokens", "Average Tokens", "avg_tokens_by_method.png"),
    ]:
        methods = [row["method"] for row in rows]
        values = [float(row[metric]) for row in rows]
        plt.figure(figsize=(max(8, len(methods) * 0.55), 4.8))
        plt.bar(methods, values, color="#3b82f6")
        plt.ylabel(ylabel)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        path = output / filename
        plt.savefig(path, dpi=220)
        plt.close()
        figures.append(path)

    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CICL metrics plots.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv")
    parser.add_argument("--output", required=True, help="Output figure directory")
    args = parser.parse_args()
    for path in plot_metrics(args.metrics, args.output):
        print(path)


if __name__ == "__main__":
    main()


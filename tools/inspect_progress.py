"""Utility script to inspect training progress logs and export them to TensorBoard."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_METRICS = (
    "success_rate",
    "eval/success_rate_mean",
    "eval/reward_mean",
    "reward",
    "SPS",
)


@dataclass
class ProgressEntry:
    step: int
    metrics: Dict[str, float]


def parse_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value: Optional[str]) -> Optional[int]:
    float_val = parse_float(value)
    if float_val is None:
        return None
    try:
        return int(float_val)
    except (ValueError, TypeError):
        return None


def detect_step(row: Dict[str, str], fallback_index: int) -> int:
    for candidate in ("epoch", "timesteps", "episodes", "grad_steps"):
        step_value = parse_int(row.get(candidate))
        if step_value is not None:
            return step_value
    return fallback_index


def load_progress(progress_file: Path) -> Tuple[List[ProgressEntry], Dict[str, List[Tuple[int, float]]]]:
    entries: List[ProgressEntry] = []
    series: Dict[str, List[Tuple[int, float]]] = {}

    with progress_file.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            step = detect_step(row, index)
            metrics: Dict[str, float] = {}
            for key, raw_value in row.items():
                numeric = parse_float(raw_value)
                if numeric is None:
                    continue
                metrics[key] = numeric
                series.setdefault(key, []).append((step, numeric))
            if metrics:
                entries.append(ProgressEntry(step=step, metrics=metrics))

    return entries, series


def build_summary(series: Dict[str, List[Tuple[int, float]]], metrics: Sequence[str]) -> List[Dict[str, float]]:
    summaries: List[Dict[str, float]] = []

    for metric in metrics:
        points = series.get(metric)
        if not points:
            continue
        points = sorted(points, key=lambda item: item[0])
        steps, values = zip(*points)
        best_value = max(values)
        best_index = values.index(best_value)
        summaries.append(
            {
                "metric": metric,
                "latest_step": steps[-1],
                "latest_value": values[-1],
                "best_step": steps[best_index],
                "best_value": best_value,
            }
        )

    return summaries


def print_summary(summary: Iterable[Dict[str, float]]) -> None:
    rows = list(summary)
    if not rows:
        print("No metrics to summarise. Try using --list-metrics to inspect available keys.")
        return

    metric_width = max(len("Metric"), max(len(row["metric"]) for row in rows))
    latest_step_width = max(len("Latest Step"), max(len(str(int(row["latest_step"]))) for row in rows))
    latest_value_width = max(len("Latest Value"), max(len(f"{row['latest_value']:.4f}") for row in rows))
    best_step_width = max(len("Best Step"), max(len(str(int(row["best_step"]))) for row in rows))
    best_value_width = max(len("Best Value"), max(len(f"{row['best_value']:.4f}") for row in rows))

    header = (
        f"{'Metric':<{metric_width}}  "
        f"{'Latest Step':>{latest_step_width}}  "
        f"{'Latest Value':>{latest_value_width}}  "
        f"{'Best Step':>{best_step_width}}  "
        f"{'Best Value':>{best_value_width}}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        latest_value_str = f"{row['latest_value']:.4f}"
        best_value_str = f"{row['best_value']:.4f}"
        print(
            f"{row['metric']:<{metric_width}}  "
            f"{row['latest_step']:>{latest_step_width}}  "
            f"{latest_value_str:>{latest_value_width}}  "
            f"{row['best_step']:>{best_step_width}}  "
            f"{best_value_str:>{best_value_width}}"
        )


def list_available_metrics(series: Dict[str, List[Tuple[int, float]]]) -> None:
    metrics = sorted(series)
    print("Available metrics ({} total):".format(len(metrics)))
    for metric in metrics:
        print(f"  - {metric}")


def build_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        try:
            from tensorboardX import SummaryWriter
        except ModuleNotFoundError as exc:
            raise ImportError(
                "TensorBoard export requires either PyTorch ('torch') or 'tensorboardX'."
            ) from exc
    return SummaryWriter(str(log_dir))


def export_to_tensorboard(entries: Sequence[ProgressEntry], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = build_summary_writer(output_dir)

    try:
        for entry in entries:
            for key, value in entry.metrics.items():
                writer.add_scalar(key, value, entry.step)
        writer.flush()
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        type=Path,
        help="Training run directory containing a 'progress.csv' file.",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="Optional override for the progress CSV file. Defaults to log_dir/progress.csv.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        help="Comma-separated list of metrics to summarise. Defaults to a curated subset.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available metric keys and exit.",
    )
    parser.add_argument(
        "--export-tensorboard",
        action="store_true",
        help="Export all metrics to TensorBoard event files.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Destination directory for TensorBoard logs. Defaults to <log_dir>/tensorboard.",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Optional path to save the summary table as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress_file = args.progress_file or args.log_dir / "progress.csv"

    if not progress_file.exists():
        raise FileNotFoundError(f"Progress file not found: {progress_file}")

    entries, series = load_progress(progress_file)

    if args.list_metrics:
        list_available_metrics(series)
        return

    metric_names: Sequence[str]
    if args.metrics:
        metric_names = tuple(name.strip() for name in args.metrics.split(",") if name.strip())
    else:
        metric_names = DEFAULT_METRICS

    summary = build_summary(series, metric_names)
    print_summary(summary)

    if args.dump_json:
        payload = {
            "metrics": summary,
            "total_entries": len(entries),
            "progress_file": str(progress_file),
        }
        args.dump_json.write_text(json.dumps(payload, indent=2))
        print(f"Summary written to {args.dump_json}")

    if args.export_tensorboard:
        tb_dir = args.tensorboard_dir or args.log_dir / "tensorboard"
        export_to_tensorboard(entries, tb_dir)
        print(f"Exported {len(entries)} steps to TensorBoard at {tb_dir}")


if __name__ == "__main__":
    main()

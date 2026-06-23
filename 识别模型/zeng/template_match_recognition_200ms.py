#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""200 ms streaming-style template matching recognition.

This script keeps the same template matching logic as template_match_recognition.py,
but applies it independently in 200 ms time windows.  For each window it writes
the original PDW rows with one extra LABEL column containing the predicted label.
Truth labels are used only for final statistics when a truth file is provided.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import template_match_recognition as rec


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
UNKNOWN_LABEL = rec.UNKNOWN_LABEL


def resolve_relative_to_project(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def resolve_existing_input_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, SCRIPT_DIR / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_column_case_insensitive(columns: List[str], wanted: str) -> str:
    wanted_lower = wanted.lower()
    for col in columns:
        if str(col).lower() == wanted_lower:
            return str(col)
    raise ValueError(f"Column '{wanted}' not found. Available columns: {columns}")


def read_streaming_window(path: Path, batch_id_col: str, max_rows: int = 0) -> Tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(path, sep=r"\s+", engine="python")
    batch_col = find_column_case_insensitive([str(c) for c in data.columns], batch_id_col)
    pdw = data.iloc[:, :8].copy()
    pdw.columns = ["TOA(s)", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6", "Param7"]
    batch_ids = pd.to_numeric(data[batch_col], errors="raise").to_numpy(dtype=np.int64)
    if max_rows > 0:
        pdw = pdw.iloc[:max_rows].reset_index(drop=True)
        batch_ids = batch_ids[:max_rows]
    return pdw, batch_ids


def parse_label_float_map(text: str) -> Dict[int, float]:
    return rec.parse_label_float_map(text)


def resolve_input_paths(args: argparse.Namespace, sample: str) -> Tuple[Path, Path, Path | None, Path]:
    if args.pdw_file and args.sort_file:
        pdw_file = resolve_relative_to_project(args.pdw_file)
        sort_file = resolve_relative_to_project(args.sort_file)
        truth_file = resolve_relative_to_project(args.truth_file) if args.truth_file else None
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        return pdw_file, sort_file, truth_file, output_dir

    sample_dir = "Sample_1" if sample == "sample1" else "Sample_2"
    pdw_value = args.sample1_pdw_file if sample == "sample1" else args.sample2_pdw_file
    sort_value = args.sample1_sort_file if sample == "sample1" else args.sample2_sort_file
    truth_value = args.sample1_truth_file if sample == "sample1" else args.sample2_truth_file

    pdw_file = resolve_relative_to_project(pdw_value) if pdw_value else Path(args.root) / "Test_Data" / sample_dir / "Merge_PDW_Data.txt"
    sort_file = resolve_relative_to_project(sort_value) if sort_value else resolve_relative_to_project(Path(args.sort_root) / sample / f"{sample}_sort.txt")
    truth_file = resolve_relative_to_project(truth_value) if truth_value else Path(args.root) / "Test_Data" / sample_dir / "Sorted_PDW.txt"

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    return pdw_file, sort_file, truth_file, output_root / sample


def window_bounds(toa: np.ndarray, window_seconds: float) -> List[Tuple[int, float, float, np.ndarray]]:
    finite = np.isfinite(toa)
    if not bool(np.any(finite)):
        return []
    bucket_ids = np.floor(toa[finite] / window_seconds).astype(np.int64)
    finite_rows = np.flatnonzero(finite)
    windows: List[Tuple[int, float, float, np.ndarray]] = []
    for window_id, bucket_id in enumerate(np.unique(bucket_ids)):
        rows = finite_rows[bucket_ids == bucket_id]
        left = float(bucket_id) * window_seconds
        right = left + window_seconds
        windows.append((int(window_id), float(left), float(right), rows))
    return windows


def match_window(
    pdw_window: pd.DataFrame,
    sigidx_window: np.ndarray,
    templates: pd.DataFrame,
    metadata: Dict[str, object],
    args: argparse.Namespace,
    threshold_scale: float,
    label_scales: Dict[int, float],
    min_margin: float,
    class_floor_scale: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    labels = np.full(len(pdw_window), UNKNOWN_LABEL, dtype=np.int64)
    batches = rec.build_recognition_batches(
        pdw_window,
        sigidx_window,
        min_batch_pulses=int(args.min_batch_pulses),
        gap_multiplier=float(args.pri_gap_multiplier),
        gap_quantile=float(args.pri_gap_quantile),
    )
    if len(batches) == 0:
        return labels, pd.DataFrame()

    pred_batches = rec.match_batches(
        batches,
        templates,
        metadata,
        threshold_scale=threshold_scale,
        label_threshold_scales=label_scales,
        class_threshold_floor_scale=class_floor_scale,
        min_margin=min_margin,
        matching_mode=str(args.matching_mode),
        class_ratio_margin=float(args.class_ratio_margin),
        enable_label2_rescue=bool(args.enable_label2_rescue),
        label2_rescue_ratio=float(args.label2_rescue_ratio),
        label2_feature_padding=float(args.label2_feature_padding),
        secondary_reject_ratio_caps=parse_label_float_map(args.secondary_reject_ratio_caps),
        enable_topk_label_rescue=bool(args.enable_topk_label_rescue),
        topk_rescue_label=int(args.topk_rescue_label),
        topk_size=int(args.topk_size),
        topk_min_votes=int(args.topk_min_votes),
        topk_max_ratio=float(args.topk_max_ratio),
        topk_feature_padding=float(args.topk_feature_padding),
        enable_class_ratio_label_rescue=bool(args.enable_class_ratio_label_rescue),
        class_ratio_rescue_label=int(args.class_ratio_rescue_label),
        class_ratio_rescue_max_ratio=float(args.class_ratio_rescue_max_ratio),
        class_ratio_rescue_max_delta=float(args.class_ratio_rescue_max_delta),
        class_ratio_rescue_feature_padding=float(args.class_ratio_rescue_feature_padding),
    )
    labels = rec.labels_from_batches(sigidx_window, pred_batches)
    return labels, pred_batches


def safe_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def compute_pulse_metrics(pred_labels: np.ndarray, truth: pd.DataFrame | None) -> Dict[str, object]:
    if truth is None:
        return {}
    truth_labels = truth["LABEL"].to_numpy(dtype=np.int64)
    if len(truth_labels) != len(pred_labels):
        raise ValueError(f"Truth/prediction row mismatch: {len(truth_labels)} vs {len(pred_labels)}")

    known = truth_labels != UNKNOWN_LABEL
    unknown = truth_labels == UNKNOWN_LABEL
    metrics: Dict[str, object] = {
        "pulse_acc": float(np.mean(pred_labels == truth_labels)),
        "known_pulse_acc": float(np.mean(pred_labels[known] == truth_labels[known])) if bool(np.any(known)) else None,
        "unknown_reject_rate": float(np.mean(pred_labels[unknown] == UNKNOWN_LABEL)) if bool(np.any(unknown)) else None,
    }
    per_label = {}
    for label in sorted(int(v) for v in np.unique(truth_labels)):
        mask = truth_labels == label
        per_label[str(label)] = {
            "num_pulses": int(np.sum(mask)),
            "acc": float(np.mean(pred_labels[mask] == truth_labels[mask])),
            "reject_as_unknown_rate": float(np.mean(pred_labels[mask] == UNKNOWN_LABEL)),
        }
    metrics["per_true_label"] = per_label
    return metrics


def run_streaming_dir(args: argparse.Namespace) -> Dict[str, object]:
    streaming_dir = resolve_existing_input_path(args.streaming_dir)
    require_exists(streaming_dir, "streaming input directory")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = None
    truth_file = resolve_relative_to_project(args.truth_file) if args.truth_file else None
    if truth_file is not None and truth_file.exists():
        truth = rec.read_truth(truth_file, int(args.max_pulses))

    library_path = Path(args.template_library)
    if not library_path.is_absolute():
        library_path = PROJECT_ROOT / library_path
    templates, metadata = rec.load_template_library(library_path)
    threshold_scale, label_scales, min_margin, class_floor_scale = rec.resolve_matching_parameters(args, library_path)

    files = sorted(streaming_dir.glob(str(args.streaming_glob)))
    if not files:
        raise FileNotFoundError(f"No files matched {args.streaming_glob!r} in {streaming_dir}")

    all_labels_parts = []
    combined_parts = []
    batch_tables = []
    window_rows = []
    row_offset = 0
    max_rows_remaining = int(args.max_pulses)

    for window_id, path in enumerate(files):
        if max_rows_remaining == 0 and int(args.max_pulses) > 0:
            break
        read_limit = max_rows_remaining if int(args.max_pulses) > 0 else 0
        pdw_window, batch_ids = read_streaming_window(path, str(args.batch_id_col), read_limit)
        if len(pdw_window) == 0:
            continue
        if int(args.max_pulses) > 0:
            max_rows_remaining -= len(pdw_window)

        labels, window_batches = match_window(
            pdw_window,
            batch_ids,
            templates,
            metadata,
            args,
            threshold_scale,
            label_scales,
            min_margin,
            class_floor_scale,
        )

        out = pdw_window.copy()
        out["LABEL"] = labels.astype(np.int64)
        output_path = output_dir / path.name
        out.to_csv(output_path, sep=" ", index=False)

        if len(window_batches) > 0:
            window_batches = window_batches.copy()
            window_batches["window_id"] = int(window_id)
            window_batches["source_file"] = str(path)
            window_batches["output_file"] = str(output_path)
            batch_tables.append(window_batches)

        all_labels_parts.append(labels)
        combined_parts.append(out)
        window_acc = None
        if truth is not None and row_offset + len(labels) <= len(truth):
            truth_labels = truth.iloc[row_offset : row_offset + len(labels)]["LABEL"].to_numpy(dtype=np.int64)
            window_acc = float(np.mean(labels == truth_labels))
        window_rows.append(
            {
                "window_id": int(window_id),
                "source_file": str(path),
                "output_file": str(output_path),
                "num_pulses": int(len(labels)),
                "num_batches": int(len(window_batches)),
                "num_unknown_pulses_pred": int(np.sum(labels == UNKNOWN_LABEL)),
                "pulse_acc": safe_float(window_acc) if window_acc is not None else None,
            }
        )
        row_offset += len(labels)

    all_labels = np.concatenate(all_labels_parts) if all_labels_parts else np.array([], dtype=np.int64)
    truth_for_metrics = truth.iloc[: len(all_labels)].reset_index(drop=True) if truth is not None else None
    combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
    batch_df = pd.concat(batch_tables, ignore_index=True) if batch_tables else pd.DataFrame()

    combined_path = output_dir / "streaming_200ms_template_match_all_pdw_with_label.txt"
    window_summary_path = output_dir / "streaming_200ms_window_summary.csv"
    batch_path = output_dir / "streaming_200ms_template_match_batches.csv"
    summary_path = output_dir / "streaming_200ms_template_match_summary.json"
    combined.to_csv(combined_path, sep=" ", index=False)
    pd.DataFrame(window_rows).to_csv(window_summary_path, index=False, encoding="utf-8-sig")
    if len(batch_df) > 0:
        batch_df.to_csv(batch_path, index=False, encoding="utf-8-sig")

    metrics = compute_pulse_metrics(all_labels, truth_for_metrics)
    summary = {
        "mode": "streaming_txt_files_template_match",
        "streaming_dir": str(streaming_dir),
        "streaming_glob": str(args.streaming_glob),
        "batch_id_col": str(args.batch_id_col),
        "truth_file": str(truth_file) if truth_file is not None else "",
        "template_library": str(library_path),
        "num_files": int(len(window_rows)),
        "num_pulses": int(len(all_labels)),
        "num_batches": int(len(batch_df)),
        "pred_label_counts": {str(k): int(v) for k, v in pd.Series(all_labels).value_counts().sort_index().items()},
        "metrics": metrics,
        "threshold_scale": float(threshold_scale),
        "label_threshold_scales": {str(k): float(v) for k, v in label_scales.items()},
        "class_threshold_floor_scale": float(class_floor_scale),
        "min_margin": float(min_margin),
        "matching_mode": str(args.matching_mode),
        "output_dir": str(output_dir),
        "combined_file": str(combined_path),
        "window_summary_file": str(window_summary_path),
        "batch_file": str(batch_path) if len(batch_df) > 0 else "",
        "summary_file": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    acc_text = ""
    if metrics:
        acc_text = (
            f", pulse_acc={metrics['pulse_acc']:.4f}, "
            f"known_acc={metrics['known_pulse_acc']:.4f}, "
            f"unknown_reject={metrics['unknown_reject_rate']:.4f}"
        )
    print(f"[done] streaming_dir: files={len(window_rows)}, batches={len(batch_df)}{acc_text}")
    print(f"[done] output_dir: {output_dir}")
    print(f"[done] combined: {combined_path}")
    return summary


def run_one(sample: str, args: argparse.Namespace) -> Dict[str, object]:
    pdw_file, sort_file, truth_file, output_dir = resolve_input_paths(args, sample)
    windows_dir = output_dir / "windows_200ms"
    windows_dir.mkdir(parents=True, exist_ok=True)
    require_exists(pdw_file, "PDW file")
    require_exists(sort_file, "sort file")

    pdw = rec.read_pdw(pdw_file, int(args.max_pulses))
    pred_sigidx = rec.read_sort_sigidx(sort_file, int(args.max_pulses))
    if len(pdw) != len(pred_sigidx):
        raise ValueError(f"PDW/sort row mismatch: {len(pdw)} vs {len(pred_sigidx)}")

    truth = None
    if truth_file is not None and truth_file.exists():
        truth = rec.read_truth(truth_file, int(args.max_pulses))
        if len(truth) != len(pred_sigidx):
            raise ValueError(f"Truth/sort row mismatch: {len(truth)} vs {len(pred_sigidx)}")

    library_path = Path(args.template_library)
    if not library_path.is_absolute():
        library_path = PROJECT_ROOT / library_path
    templates, metadata = rec.load_template_library(library_path)
    threshold_scale, label_scales, min_margin, class_floor_scale = rec.resolve_matching_parameters(args, library_path)

    toa = pdw["TOA(s)"].to_numpy(dtype=np.float64)
    all_labels = np.full(len(pdw), UNKNOWN_LABEL, dtype=np.int64)
    batch_tables = []
    window_rows = []
    combined_parts = []

    for window_id, start_s, end_s, rows in window_bounds(toa, float(args.window_seconds)):
        window_pdw = pdw.iloc[rows].reset_index(drop=True)
        window_sigidx = pred_sigidx[rows]
        window_labels, window_batches = match_window(
            window_pdw,
            window_sigidx,
            templates,
            metadata,
            args,
            threshold_scale,
            label_scales,
            min_margin,
            class_floor_scale,
        )
        all_labels[rows] = window_labels

        window_out = window_pdw.copy()
        window_out["LABEL"] = window_labels.astype(np.int64)
        window_name = f"{sample}_window_{window_id:06d}_{start_s:.3f}_{end_s:.3f}s.txt"
        window_path = windows_dir / window_name
        window_out.to_csv(window_path, sep=" ", index=False)
        combined_parts.append(window_out)

        if len(window_batches) > 0:
            window_batches = window_batches.copy()
            window_batches["window_id"] = int(window_id)
            window_batches["window_start_s"] = float(start_s)
            window_batches["window_end_s"] = float(end_s)
            batch_tables.append(window_batches)

        window_acc = None
        if truth is not None:
            truth_labels = truth.iloc[rows]["LABEL"].to_numpy(dtype=np.int64)
            window_acc = float(np.mean(window_labels == truth_labels))
        window_rows.append(
            {
                "window_id": int(window_id),
                "window_start_s": float(start_s),
                "window_end_s": float(end_s),
                "num_pulses": int(len(rows)),
                "num_batches": int(len(window_batches)),
                "num_unknown_pulses_pred": int(np.sum(window_labels == UNKNOWN_LABEL)),
                "window_file": str(window_path),
                "pulse_acc": safe_float(window_acc) if window_acc is not None else None,
            }
        )

    combined = pd.concat(combined_parts, ignore_index=True) if combined_parts else pd.DataFrame()
    batch_df = pd.concat(batch_tables, ignore_index=True) if batch_tables else pd.DataFrame()

    combined_path = output_dir / f"{sample}_200ms_template_match_all_pdw_with_label.txt"
    window_summary_path = output_dir / f"{sample}_200ms_window_summary.csv"
    batch_path = output_dir / f"{sample}_200ms_template_match_batches.csv"
    summary_path = output_dir / f"{sample}_200ms_template_match_summary.json"

    combined.to_csv(combined_path, sep=" ", index=False)
    pd.DataFrame(window_rows).to_csv(window_summary_path, index=False, encoding="utf-8-sig")
    if len(batch_df) > 0:
        batch_df.to_csv(batch_path, index=False, encoding="utf-8-sig")

    metrics = compute_pulse_metrics(all_labels, truth)
    summary = {
        "sample": sample,
        "mode": "200ms_window_streaming_template_match",
        "window_seconds": float(args.window_seconds),
        "pdw_file": str(pdw_file),
        "sort_file": str(sort_file),
        "truth_file": str(truth_file) if truth_file is not None else "",
        "template_library": str(library_path),
        "num_pulses": int(len(pdw)),
        "num_windows": int(len(window_rows)),
        "num_batches": int(len(batch_df)),
        "pred_label_counts": {str(k): int(v) for k, v in pd.Series(all_labels).value_counts().sort_index().items()},
        "metrics": metrics,
        "threshold_scale": float(threshold_scale),
        "label_threshold_scales": {str(k): float(v) for k, v in label_scales.items()},
        "class_threshold_floor_scale": float(class_floor_scale),
        "min_margin": float(min_margin),
        "matching_mode": str(args.matching_mode),
        "windows_dir": str(windows_dir),
        "combined_file": str(combined_path),
        "window_summary_file": str(window_summary_path),
        "batch_file": str(batch_path) if len(batch_df) > 0 else "",
        "summary_file": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    acc_text = ""
    if metrics:
        acc_text = (
            f", pulse_acc={metrics['pulse_acc']:.4f}, "
            f"known_acc={metrics['known_pulse_acc']:.4f}, "
            f"unknown_reject={metrics['unknown_reject_rate']:.4f}"
        )
    print(f"[done] {sample}: windows={len(window_rows)}, batches={len(batch_df)}{acc_text}")
    print(f"[done] windows_dir: {windows_dir}")
    print(f"[done] combined: {combined_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template matching recognition split into 200 ms windows.")
    parser.add_argument("--root", type=Path, default=Path("E:/分选/Data"))
    parser.add_argument("--sample", choices=["sample1", "sample2", "all"], default="all")
    parser.add_argument("--sort_root", type=Path, default=Path("outputs_best_front_tracklet_graph"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs_template_match_recognition_200ms"))
    parser.add_argument("--streaming_dir", type=str, default="")
    parser.add_argument("--streaming_glob", type=str, default="*.txt")
    parser.add_argument("--batch_id_col", type=str, default="OurPredID")
    parser.add_argument("--pdw_file", type=str, default="")
    parser.add_argument("--sort_file", type=str, default="")
    parser.add_argument("--truth_file", type=str, default="")
    parser.add_argument("--sample1_pdw_file", type=str, default="E:/分选/Data/Test_Data/Sample_1/Merge_PDW_Data.txt")
    parser.add_argument("--sample1_sort_file", type=str, default="outputs_best_front_tracklet_graph/sample1/sample1_sort.txt")
    parser.add_argument("--sample1_truth_file", type=str, default="E:/分选/Data/Test_Data/Sample_1/Sorted_PDW.txt")
    parser.add_argument("--sample2_pdw_file", type=str, default="E:/分选/Data/Test_Data/Sample_2/Merge_PDW_Data.txt")
    parser.add_argument("--sample2_sort_file", type=str, default="outputs_best_front_tracklet_graph/sample2/sample2_sort.txt")
    parser.add_argument("--sample2_truth_file", type=str, default="E:/分选/Data/Test_Data/Sample_2/Sorted_PDW.txt")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_template_match_recognition_200ms/custom"))
    parser.add_argument("--template_library", type=Path, default=Path("outputs_expanded_template_library/template_library.json"))
    parser.add_argument("--max_pulses", type=int, default=0)
    parser.add_argument("--window_seconds", type=float, default=0.2)
    parser.add_argument("--min_batch_pulses", type=int, default=20)
    parser.add_argument("--pri_gap_multiplier", type=float, default=5.0)
    parser.add_argument("--pri_gap_quantile", type=float, default=0.90)
    parser.add_argument("--threshold_scale", type=float, default=0.5)
    parser.add_argument("--label_threshold_scales", type=str, default="1:0.5,2:0.5,3:0.5,4:0.5")
    parser.add_argument("--class_threshold_floor_scale", type=float, default=0.4)
    parser.add_argument("--min_margin", type=float, default=0.0)
    parser.add_argument("--matching_mode", choices=["nearest", "nearest_with_rescue", "class_min_ratio"], default="nearest")
    parser.add_argument("--class_ratio_margin", type=float, default=0.0)
    parser.add_argument("--enable_label2_rescue", action="store_true")
    parser.add_argument("--label2_rescue_ratio", type=float, default=1.0)
    parser.add_argument("--label2_feature_padding", type=float, default=0.5)
    parser.add_argument("--secondary_reject_ratio_caps", type=str, default="")
    parser.add_argument("--enable_topk_label_rescue", action="store_true")
    parser.add_argument("--topk_rescue_label", type=int, default=2)
    parser.add_argument("--topk_size", type=int, default=8)
    parser.add_argument("--topk_min_votes", type=int, default=2)
    parser.add_argument("--topk_max_ratio", type=float, default=2.0)
    parser.add_argument("--topk_feature_padding", type=float, default=1.0)
    parser.add_argument("--enable_class_ratio_label_rescue", action="store_true")
    parser.add_argument("--class_ratio_rescue_label", type=int, default=2)
    parser.add_argument("--class_ratio_rescue_max_ratio", type=float, default=2.0)
    parser.add_argument("--class_ratio_rescue_max_delta", type=float, default=0.5)
    parser.add_argument("--class_ratio_rescue_feature_padding", type=float, default=1.0)
    parser.add_argument("--tuning_file", type=str, default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if str(args.streaming_dir).strip():
        run_streaming_dir(args)
        return
    samples = ["sample1", "sample2"] if args.sample == "all" else [args.sample]
    summaries = [run_one(sample, args) for sample in samples]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    all_summary_path = output_root / "template_match_200ms_all_summary.json"
    all_summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] all_summary: {all_summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-level template matching recognition on sorted SigIdx outputs.

Input is the same kind of sorted output used by kmeans_sort_recognition.py:
test PDW rows plus a sort file containing SigIdx. The script builds one
feature vector per predicted SigIdx batch, matches it to the nearest training
template, rejects far batches as unknown label 99, and writes pulse-level LABEL
output plus batch-level diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


UNKNOWN_LABEL = 99


def read_pdw(path: Path, max_rows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df = df.iloc[:, :8].copy()
    df.columns = ["TOA(s)", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6", "Param7"]
    if max_rows > 0:
        df = df.iloc[:max_rows].reset_index(drop=True)
    return df


def read_sort_sigidx(path: Path, max_rows: int = 0) -> np.ndarray:
    data = pd.read_csv(path, sep=r"\s+", engine="python", usecols=["SigIdx"])
    sigidx = pd.to_numeric(data["SigIdx"], errors="raise").to_numpy(dtype=np.int64)
    if max_rows > 0:
        sigidx = sigidx[:max_rows]
    return sigidx


def robust_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0, "iqr": 0.0}
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def robust_pri_us(toa_s: np.ndarray, gap_multiplier: float, gap_quantile: float) -> np.ndarray:
    toa = np.sort(np.asarray(toa_s, dtype=np.float64))
    dtoa = np.diff(toa) * 1e6
    dtoa = dtoa[np.isfinite(dtoa) & (dtoa > 0.0)]
    if len(dtoa) < 3:
        return dtoa
    median = float(np.median(dtoa))
    cap = max(float(gap_multiplier) * max(median, 1e-9), float(np.quantile(dtoa, gap_quantile)))
    return dtoa[dtoa <= cap]


def extract_batch_features(
    sub: pd.DataFrame,
    pred_sigidx: int,
    gap_multiplier: float,
    gap_quantile: float,
) -> Dict[str, float | int]:
    toa = sub["TOA(s)"].to_numpy(dtype=np.float64)
    duration = float(max(np.max(toa) - np.min(toa), 0.0))
    pulse_rate = float(len(sub) / max(duration, 1e-9))
    pri = robust_pri_us(toa, gap_multiplier, gap_quantile)
    pri_stats = robust_stats(pri)

    row: Dict[str, float | int] = {
        "pred_sigidx": int(pred_sigidx),
        "num_pulses": int(len(sub)),
        "log_num_pulses": float(math.log1p(len(sub))),
        "duration_s": duration,
        "log_duration": float(math.log1p(duration)),
        "pulse_rate": pulse_rate,
        "log_pulse_rate": float(math.log1p(pulse_rate)),
        "pri_median_us": pri_stats["median"],
        "pri_iqr_us": pri_stats["iqr"],
        "pri_cv": float(pri_stats["std"] / max(pri_stats["mean"], 1e-9)) if len(pri) else 0.0,
    }
    for param, prefix in [("Param1", "p1"), ("Param2", "p2"), ("Param3", "p3"), ("Param4", "p4")]:
        stats = robust_stats(sub[param].to_numpy(dtype=np.float64))
        for name, value in stats.items():
            row[f"{prefix}_{name}"] = value
    return row


def build_recognition_batches(
    pdw: pd.DataFrame,
    pred_sigidx: np.ndarray,
    min_batch_pulses: int,
    gap_multiplier: float,
    gap_quantile: float,
) -> pd.DataFrame:
    work = pdw.copy()
    work["PredSigIdx"] = pred_sigidx.astype(np.int64)
    rows = []
    for sigidx, sub in work[work["PredSigIdx"] > 0].groupby("PredSigIdx", sort=True):
        if len(sub) < int(min_batch_pulses):
            continue
        rows.append(extract_batch_features(sub, int(sigidx), gap_multiplier, gap_quantile))
    return pd.DataFrame(rows)


def load_template_library(path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data["metadata"]
    templates = pd.DataFrame(data["templates"])
    if len(templates) == 0:
        raise ValueError(f"Template library is empty: {path}")
    return templates, metadata


def weighted_scaled_matrix(
    table: pd.DataFrame,
    features: List[str],
    centers: Dict[str, float],
    scales: Dict[str, float],
    weights: Dict[str, float],
) -> np.ndarray:
    cols = []
    for name in features:
        values = pd.to_numeric(table[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        center = float(centers[name])
        scale = max(float(scales[name]), 1e-9)
        weight = max(float(weights.get(name, 1.0)), 0.0)
        cols.append(((values.fillna(center).to_numpy(dtype=np.float64) - center) / scale) * math.sqrt(weight))
    return np.vstack(cols).T.astype(np.float64)


def match_batches(
    batches: pd.DataFrame,
    templates: pd.DataFrame,
    metadata: Dict[str, object],
    threshold_scale: float,
    min_margin: float,
) -> pd.DataFrame:
    features = [str(v) for v in metadata["features"] if str(v) in batches.columns and str(v) in templates.columns]
    if not features:
        raise ValueError("No shared features between batches and template library.")

    centers = {str(k): float(v) for k, v in metadata["feature_centers"].items()}
    scales = {str(k): float(v) for k, v in metadata["feature_scales"].items()}
    weights = {str(k): float(v) for k, v in metadata["feature_weights"].items()}

    batch_x = weighted_scaled_matrix(batches, features, centers, scales, weights)
    template_x = weighted_scaled_matrix(templates, features, centers, scales, weights)
    distances = np.linalg.norm(batch_x[:, None, :] - template_x[None, :, :], axis=2)

    order = np.argsort(distances, axis=1)
    nearest_idx = order[:, 0]
    second_idx = order[:, 1] if distances.shape[1] > 1 else order[:, 0]
    nearest_dist = distances[np.arange(len(batches)), nearest_idx]
    second_dist = distances[np.arange(len(batches)), second_idx]
    margin = second_dist - nearest_dist

    nearest = templates.iloc[nearest_idx].reset_index(drop=True)
    thresholds = pd.to_numeric(nearest["local_distance_threshold"], errors="coerce").fillna(np.inf).to_numpy(dtype=np.float64)
    thresholds = thresholds * float(threshold_scale)
    accepted = (nearest_dist <= thresholds) & (margin >= float(min_margin))

    out = batches.copy().reset_index(drop=True)
    out["nearest_template_id"] = nearest["template_id"].astype(str).to_numpy()
    out["nearest_template_label"] = nearest["label"].astype(np.int64).to_numpy()
    out["template_distance"] = nearest_dist.astype(np.float64)
    out["second_template_distance"] = second_dist.astype(np.float64)
    out["template_margin"] = margin.astype(np.float64)
    out["template_distance_threshold"] = thresholds.astype(np.float64)
    out["nearest_template_ambiguous"] = nearest.get("ambiguous_template", False).astype(bool).to_numpy()
    out["ood_reject"] = ~accepted
    out["batch_pred_label"] = np.where(out["ood_reject"], UNKNOWN_LABEL, out["nearest_template_label"]).astype(np.int64)
    out["batch_max_prob"] = np.clip(1.0 - out["template_distance"] / np.maximum(out["template_distance_threshold"], 1e-9), 0.0, 1.0)
    out["is_batch_confident"] = ~out["ood_reject"]
    return out


def labels_from_batches(pred_sigidx: np.ndarray, batch_df: pd.DataFrame) -> np.ndarray:
    lookup = {
        int(row.pred_sigidx): int(row.batch_pred_label)
        for row in batch_df[["pred_sigidx", "batch_pred_label"]].itertuples(index=False)
    }
    labels = np.full(len(pred_sigidx), UNKNOWN_LABEL, dtype=np.int64)
    for sigidx, label in lookup.items():
        labels[pred_sigidx == sigidx] = label
    return labels


def default_paths(root: Path, sample: str, sort_root: Path) -> Tuple[Path, Path]:
    sample_dir = "Sample_1" if sample == "sample1" else "Sample_2"
    pdw_file = root / "edata" / "Test_Data" / sample_dir / "Merge_PDW_Data.txt"
    sort_file = sort_root / sample / f"{sample}_sort.txt"
    return pdw_file, sort_file


def resolve_input_paths(args: argparse.Namespace, sample: str) -> Tuple[Path, Path, Path]:
    if args.pdw_file and args.sort_file:
        return Path(args.pdw_file), Path(args.sort_file), Path(args.output_dir)

    root = Path(args.root).resolve()
    sort_root = Path(args.sort_root)
    if not sort_root.is_absolute():
        sort_root = root / sort_root
    pdw_file, sort_file = default_paths(root, sample, sort_root)
    output_dir = Path(args.output_root) / sample
    return pdw_file, sort_file, output_dir


def run_one(sample: str, args: argparse.Namespace) -> Dict[str, object]:
    pdw_file, sort_file, output_dir = resolve_input_paths(args, sample)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdw = read_pdw(pdw_file, int(args.max_pulses))
    pred_sigidx = read_sort_sigidx(sort_file, int(args.max_pulses))
    if len(pdw) != len(pred_sigidx):
        raise ValueError(f"PDW/sort row mismatch: {len(pdw)} vs {len(pred_sigidx)}")

    templates, metadata = load_template_library(Path(args.template_library))
    batches = build_recognition_batches(
        pdw,
        pred_sigidx,
        min_batch_pulses=int(args.min_batch_pulses),
        gap_multiplier=float(args.pri_gap_multiplier),
        gap_quantile=float(args.pri_gap_quantile),
    )
    pred_batches = match_batches(
        batches,
        templates,
        metadata,
        threshold_scale=float(args.threshold_scale),
        min_margin=float(args.min_margin),
    )
    labels = labels_from_batches(pred_sigidx, pred_batches)

    final_df = pd.DataFrame(
        {
            "TOA(s)": pdw["TOA(s)"].to_numpy(dtype=np.float64),
            "SigIdx": pred_sigidx.astype(np.int64),
            "LABEL": labels.astype(np.int64),
        }
    )
    batch_path = output_dir / f"{sample}_template_match_batches.csv"
    final_path = output_dir / f"{sample}_template_match_final.txt"
    summary_path = output_dir / f"{sample}_template_match_summary.json"
    pred_batches.to_csv(batch_path, index=False, encoding="utf-8-sig")
    final_df.to_csv(final_path, sep=" ", index=False)

    summary = {
        "sample": sample,
        "pdw_file": str(pdw_file),
        "sort_file": str(sort_file),
        "template_library": str(args.template_library),
        "num_batches": int(len(pred_batches)),
        "num_unknown_batches": int((pred_batches["batch_pred_label"] == UNKNOWN_LABEL).sum()),
        "pred_label_counts": {str(k): int(v) for k, v in pred_batches["batch_pred_label"].value_counts().sort_index().items()},
        "threshold_scale": float(args.threshold_scale),
        "min_margin": float(args.min_margin),
        "output_file": str(final_path),
        "batch_file": str(batch_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[done] {sample}: batches={summary['num_batches']}, "
        f"unknown={summary['num_unknown_batches']}, output={final_path}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template matching recognition for sorted SigIdx outputs.")
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--sample", choices=["sample1", "sample2", "all"], default="all")
    parser.add_argument("--sort_root", type=Path, default=Path("outputs_performance_first"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs_template_match_recognition"))
    parser.add_argument("--pdw_file", type=str, default="")
    parser.add_argument("--sort_file", type=str, default="")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_template_match_recognition/custom"))
    parser.add_argument("--template_library", type=Path, default=Path("outputs_template_matching_library/template_library.json"))
    parser.add_argument("--max_pulses", type=int, default=0)
    parser.add_argument("--min_batch_pulses", type=int, default=20)
    parser.add_argument("--pri_gap_multiplier", type=float, default=5.0)
    parser.add_argument("--pri_gap_quantile", type=float, default=0.90)
    parser.add_argument("--threshold_scale", type=float, default=1.0)
    parser.add_argument("--min_margin", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = ["sample1", "sample2"] if args.sample == "all" else [args.sample]
    summaries = [run_one(sample, args) for sample in samples]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "template_match_all_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

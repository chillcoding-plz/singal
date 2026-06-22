#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-level template matching recognition on sorted SigIdx outputs.

Input is the same kind of sorted output used by kmeans_sort_recognition.py:
test PDW rows plus a sort file containing SigIdx.  The script builds one
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
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR


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


def read_truth(path: Path, max_rows: int = 0) -> pd.DataFrame:
    truth = pd.read_csv(path, sep=r"\s+", engine="python", usecols=["SigIdx", "LABEL"])
    if max_rows > 0:
        truth = truth.iloc[:max_rows].reset_index(drop=True)
    return truth


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


def parse_label_scales(text: str) -> Dict[int, float]:
    scales: Dict[int, float] = {}
    if not str(text).strip():
        return scales
    for item in str(text).split(","):
        if not item.strip():
            continue
        label, scale = item.split(":", 1)
        scales[int(label.strip())] = float(scale.strip())
    return scales


def parse_label_float_map(text: str) -> Dict[int, float]:
    return parse_label_scales(text)


def load_tuning_file(path: Path) -> Tuple[float | None, Dict[int, float], float | None, float | None]:
    if not path.exists():
        return None, {}, None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    threshold_scale = data.get("threshold_scale")
    label_scales = {int(k): float(v) for k, v in data.get("label_threshold_scales", {}).items()}
    min_margin = data.get("min_margin")
    class_floor_scale = data.get("class_threshold_floor_scale")
    return (
        float(threshold_scale) if threshold_scale is not None else None,
        label_scales,
        float(min_margin) if min_margin is not None else None,
        float(class_floor_scale) if class_floor_scale is not None else None,
    )


def resolve_matching_parameters(args: argparse.Namespace, template_library: Path) -> Tuple[float, Dict[int, float], float, float]:
    label_scales = parse_label_scales(args.label_threshold_scales)
    threshold_scale = None if math.isnan(float(args.threshold_scale)) else float(args.threshold_scale)
    min_margin = None if math.isnan(float(args.min_margin)) else float(args.min_margin)
    class_floor_scale = None if math.isnan(float(args.class_threshold_floor_scale)) else float(args.class_threshold_floor_scale)

    tuning_file = Path(args.tuning_file) if str(args.tuning_file).strip() else template_library.with_name("tuned_match_parameters.json")
    if not tuning_file.is_absolute():
        tuning_file = PROJECT_ROOT / tuning_file
    tuned_scale, tuned_label_scales, tuned_margin, tuned_floor_scale = load_tuning_file(tuning_file)

    if threshold_scale is None:
        threshold_scale = tuned_scale if tuned_scale is not None else 1.0
    if not label_scales:
        label_scales = tuned_label_scales
    if min_margin is None:
        min_margin = tuned_margin if tuned_margin is not None else 0.0
    if class_floor_scale is None:
        class_floor_scale = tuned_floor_scale if tuned_floor_scale is not None else 0.0
    return threshold_scale, label_scales, min_margin, class_floor_scale


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


def adjusted_template_thresholds(
    templates: pd.DataFrame,
    metadata: Dict[str, object],
    threshold_scale: float,
    label_threshold_scales: Dict[int, float],
    class_threshold_floor_scale: float,
) -> np.ndarray:
    labels = templates["label"].astype(np.int64).to_numpy()
    local = pd.to_numeric(templates["local_distance_threshold"], errors="coerce").fillna(np.inf).to_numpy(dtype=np.float64)
    label_scale = np.array([float(label_threshold_scales.get(int(label), 1.0)) for label in labels], dtype=np.float64)
    thresholds = local * float(threshold_scale) * label_scale
    class_thresholds = {int(k): float(v) for k, v in metadata.get("class_thresholds", {}).items()}
    class_floor = np.array(
        [float(class_thresholds.get(int(label), 0.0)) * float(class_threshold_floor_scale) for label in labels],
        dtype=np.float64,
    )
    return np.maximum(thresholds, class_floor)


def feature_in_class_range(
    row: pd.Series,
    metadata: Dict[str, object],
    label: int,
    feature_names: List[str],
    padding_scale: float,
) -> bool:
    class_summary = metadata.get("class_summary", {})
    if not isinstance(class_summary, dict):
        return True
    label_summary = class_summary.get(str(label), {})
    if not isinstance(label_summary, dict):
        return True
    feature_summary = label_summary.get("features", {})
    if not isinstance(feature_summary, dict):
        return True
    global_scales = {str(k): float(v) for k, v in metadata.get("feature_scales", {}).items()}
    for name in feature_names:
        stats = feature_summary.get(name, {})
        if not isinstance(stats, dict) or name not in row:
            continue
        value = float(row[name])
        pad = float(padding_scale) * max(float(global_scales.get(name, 1.0)), 1e-9)
        lo = float(stats.get("min", -np.inf)) - pad
        hi = float(stats.get("max", np.inf)) + pad
        if value < lo or value > hi:
            return False
    return True


def match_batches(
    batches: pd.DataFrame,
    templates: pd.DataFrame,
    metadata: Dict[str, object],
    threshold_scale: float,
    label_threshold_scales: Dict[int, float],
    class_threshold_floor_scale: float,
    min_margin: float,
    matching_mode: str = "nearest",
    class_ratio_margin: float = 0.0,
    enable_label2_rescue: bool = False,
    label2_rescue_ratio: float = 1.0,
    label2_feature_padding: float = 0.5,
    secondary_reject_ratio_caps: Dict[int, float] | None = None,
    enable_topk_label_rescue: bool = False,
    topk_rescue_label: int = 2,
    topk_size: int = 8,
    topk_min_votes: int = 2,
    topk_max_ratio: float = 2.0,
    topk_feature_padding: float = 1.0,
    enable_class_ratio_label_rescue: bool = False,
    class_ratio_rescue_label: int = 2,
    class_ratio_rescue_max_ratio: float = 2.0,
    class_ratio_rescue_max_delta: float = 0.5,
    class_ratio_rescue_feature_padding: float = 1.0,
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
    template_thresholds = adjusted_template_thresholds(
        templates,
        metadata,
        threshold_scale=threshold_scale,
        label_threshold_scales=label_threshold_scales,
        class_threshold_floor_scale=class_threshold_floor_scale,
    )

    order = np.argsort(distances, axis=1)
    nearest_idx = order[:, 0]
    second_idx = order[:, 1] if distances.shape[1] > 1 else order[:, 0]
    nearest_dist = distances[np.arange(len(batches)), nearest_idx]
    second_dist = distances[np.arange(len(batches)), second_idx]
    margin = second_dist - nearest_dist

    nearest = templates.iloc[nearest_idx].reset_index(drop=True)
    thresholds = template_thresholds[nearest_idx]
    nearest_labels = nearest["label"].astype(np.int64).to_numpy()
    accepted = (nearest_dist <= thresholds) & (margin >= float(min_margin))

    template_labels = templates["label"].astype(np.int64).to_numpy()
    class_labels = sorted(int(v) for v in np.unique(template_labels))
    class_best_idx = np.full((len(batches), len(class_labels)), -1, dtype=np.int64)
    class_best_dist = np.full((len(batches), len(class_labels)), np.inf, dtype=np.float64)
    class_best_threshold = np.full((len(batches), len(class_labels)), np.inf, dtype=np.float64)
    for col, label in enumerate(class_labels):
        idx = np.flatnonzero(template_labels == label)
        if len(idx) == 0:
            continue
        label_dists = distances[:, idx]
        local_order = np.argmin(label_dists, axis=1)
        best_idx = idx[local_order]
        class_best_idx[:, col] = best_idx
        class_best_dist[:, col] = distances[np.arange(len(batches)), best_idx]
        class_best_threshold[:, col] = template_thresholds[best_idx]
    class_best_ratio = class_best_dist / np.maximum(class_best_threshold, 1e-9)

    if matching_mode == "class_min_ratio":
        ratio_order = np.argsort(class_best_ratio, axis=1)
        best_class_col = ratio_order[:, 0]
        second_class_col = ratio_order[:, 1] if class_best_ratio.shape[1] > 1 else ratio_order[:, 0]
        chosen_idx = class_best_idx[np.arange(len(batches)), best_class_col]
        chosen = templates.iloc[chosen_idx].reset_index(drop=True)
        chosen_dist = class_best_dist[np.arange(len(batches)), best_class_col]
        chosen_threshold = class_best_threshold[np.arange(len(batches)), best_class_col]
        chosen_ratio = class_best_ratio[np.arange(len(batches)), best_class_col]
        second_ratio = class_best_ratio[np.arange(len(batches)), second_class_col]
        ratio_margin = second_ratio - chosen_ratio
        nearest = chosen
        nearest_idx = chosen_idx
        nearest_dist = chosen_dist
        thresholds = chosen_threshold
        nearest_labels = chosen["label"].astype(np.int64).to_numpy()
        accepted = (chosen_ratio <= 1.0) & (ratio_margin >= float(class_ratio_margin))
        margin = ratio_margin

    rescue_applied = np.zeros(len(batches), dtype=bool)
    if matching_mode in {"nearest", "nearest_with_rescue"} and enable_label2_rescue and 2 in class_labels:
        label2_col = class_labels.index(2)
        label2_ratio = class_best_ratio[:, label2_col]
        label2_idx = class_best_idx[:, label2_col]
        label2_dist = class_best_dist[:, label2_col]
        label2_threshold = class_best_threshold[:, label2_col]
        rescue_features = ["p1_median", "p2_median", "p4_median"]
        for i in range(len(batches)):
            if nearest_labels[i] == 2 and accepted[i]:
                continue
            if label2_idx[i] < 0 or label2_ratio[i] > float(label2_rescue_ratio):
                continue
            if not feature_in_class_range(batches.iloc[i], metadata, 2, rescue_features, float(label2_feature_padding)):
                continue
            nearest_idx[i] = label2_idx[i]
            nearest_dist[i] = label2_dist[i]
            thresholds[i] = label2_threshold[i]
            nearest_labels[i] = 2
            accepted[i] = True
            rescue_applied[i] = True
        nearest = templates.iloc[nearest_idx].reset_index(drop=True)

    topk_rescue_applied = np.zeros(len(batches), dtype=bool)
    if enable_topk_label_rescue and int(topk_rescue_label) in class_labels:
        rescue_label = int(topk_rescue_label)
        rescue_col = class_labels.index(rescue_label)
        k = max(1, min(int(topk_size), distances.shape[1]))
        topk_idx = order[:, :k]
        topk_labels = template_labels[topk_idx]
        rescue_features = ["p1_median", "p2_median", "p4_median"]
        for i in range(len(batches)):
            votes = int(np.sum(topk_labels[i] == rescue_label))
            if votes < int(topk_min_votes):
                continue
            if class_best_ratio[i, rescue_col] > float(topk_max_ratio):
                continue
            if not feature_in_class_range(batches.iloc[i], metadata, rescue_label, rescue_features, float(topk_feature_padding)):
                continue
            rescue_idx = class_best_idx[i, rescue_col]
            if rescue_idx < 0:
                continue
            nearest_idx[i] = rescue_idx
            nearest_dist[i] = class_best_dist[i, rescue_col]
            thresholds[i] = class_best_threshold[i, rescue_col]
            nearest_labels[i] = rescue_label
            accepted[i] = True
            topk_rescue_applied[i] = True
        nearest = templates.iloc[nearest_idx].reset_index(drop=True)

    class_ratio_rescue_applied = np.zeros(len(batches), dtype=bool)
    if enable_class_ratio_label_rescue and int(class_ratio_rescue_label) in class_labels:
        rescue_label = int(class_ratio_rescue_label)
        rescue_col = class_labels.index(rescue_label)
        best_ratio = np.min(class_best_ratio, axis=1)
        rescue_features = ["p1_median", "p2_median", "p4_median"]
        for i in range(len(batches)):
            rescue_ratio = float(class_best_ratio[i, rescue_col])
            if rescue_ratio > float(class_ratio_rescue_max_ratio):
                continue
            if rescue_ratio - float(best_ratio[i]) > float(class_ratio_rescue_max_delta):
                continue
            if not feature_in_class_range(
                batches.iloc[i],
                metadata,
                rescue_label,
                rescue_features,
                float(class_ratio_rescue_feature_padding),
            ):
                continue
            rescue_idx = class_best_idx[i, rescue_col]
            if rescue_idx < 0:
                continue
            nearest_idx[i] = rescue_idx
            nearest_dist[i] = class_best_dist[i, rescue_col]
            thresholds[i] = class_best_threshold[i, rescue_col]
            nearest_labels[i] = rescue_label
            accepted[i] = True
            class_ratio_rescue_applied[i] = True
        nearest = templates.iloc[nearest_idx].reset_index(drop=True)

    secondary_reject_applied = np.zeros(len(batches), dtype=bool)
    ratio = nearest_dist / np.maximum(thresholds, 1e-9)
    for label, cap in (secondary_reject_ratio_caps or {}).items():
        mask = accepted & (nearest_labels == int(label)) & (ratio > float(cap))
        if np.any(mask):
            accepted[mask] = False
            secondary_reject_applied[mask] = True

    out = batches.copy().reset_index(drop=True)
    out["nearest_template_id"] = nearest["template_id"].astype(str).to_numpy()
    out["nearest_template_label"] = nearest_labels
    out["template_distance"] = nearest_dist.astype(np.float64)
    out["second_template_distance"] = second_dist.astype(np.float64)
    out["template_margin"] = margin.astype(np.float64)
    out["template_distance_threshold"] = thresholds.astype(np.float64)
    out["template_distance_ratio"] = out["template_distance"] / np.maximum(out["template_distance_threshold"], 1e-9)
    out["nearest_template_ambiguous"] = nearest.get("ambiguous_template", False).astype(bool).to_numpy()
    for col, label in enumerate(class_labels):
        out[f"class{label}_distance"] = class_best_dist[:, col].astype(np.float64)
        out[f"class{label}_distance_threshold"] = class_best_threshold[:, col].astype(np.float64)
        out[f"class{label}_distance_ratio"] = class_best_ratio[:, col].astype(np.float64)
    out["matching_mode"] = str(matching_mode)
    out["label2_rescue_applied"] = rescue_applied
    out["topk_rescue_applied"] = topk_rescue_applied
    out["class_ratio_rescue_applied"] = class_ratio_rescue_applied
    out["secondary_reject_applied"] = secondary_reject_applied
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


def majority(values: np.ndarray, default: int = UNKNOWN_LABEL) -> int:
    values = np.asarray(values, dtype=np.int64)
    if len(values) == 0:
        return int(default)
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[np.argmax(counts)])


def add_truth_to_batches(batch_df: pd.DataFrame, truth: pd.DataFrame, pred_sigidx: np.ndarray) -> pd.DataFrame:
    work = truth.copy()
    work["PredSigIdx"] = pred_sigidx.astype(np.int64)
    rows = []
    for pred_id, sub in work[work["PredSigIdx"] > 0].groupby("PredSigIdx", sort=True):
        labels = sub["LABEL"].to_numpy(dtype=np.int64)
        maj = majority(labels)
        maj_count = int(np.sum(labels == maj))
        rows.append(
            {
                "pred_sigidx": int(pred_id),
                "true_majority_label": int(maj),
                "true_majority_count": maj_count,
                "true_batch_purity": float(maj_count / max(len(labels), 1)),
            }
        )
    truth_batch = pd.DataFrame(rows)
    return batch_df.merge(truth_batch, on="pred_sigidx", how="left")


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else float("nan")


def compute_recognition_metrics(batch_df: pd.DataFrame, pulse_labels: np.ndarray, truth: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {}
    if "true_majority_label" in batch_df.columns and len(batch_df):
        valid = batch_df["true_majority_label"].notna()
        known = valid & (batch_df["true_majority_label"].astype(np.int64) != UNKNOWN_LABEL)
        unknown = valid & (batch_df["true_majority_label"].astype(np.int64) == UNKNOWN_LABEL)
        out["batch_acc"] = float((batch_df.loc[valid, "batch_pred_label"].astype(np.int64) == batch_df.loc[valid, "true_majority_label"].astype(np.int64)).mean()) if bool(valid.any()) else float("nan")
        out["known_batch_acc"] = float((batch_df.loc[known, "batch_pred_label"].astype(np.int64) == batch_df.loc[known, "true_majority_label"].astype(np.int64)).mean()) if bool(known.any()) else float("nan")
        out["unknown_reject_rate"] = float((batch_df.loc[unknown, "batch_pred_label"].astype(np.int64) == UNKNOWN_LABEL).mean()) if bool(unknown.any()) else float("nan")
        out["false_accept_rate"] = float((batch_df.loc[unknown, "batch_pred_label"].astype(np.int64) != UNKNOWN_LABEL).mean()) if bool(unknown.any()) else float("nan")
        out["num_known_batches"] = int(known.sum())
        out["num_unknown_batches_truth"] = int(unknown.sum())
        ratio = (
            pd.to_numeric(batch_df["template_distance"], errors="coerce")
            / pd.to_numeric(batch_df["template_distance_threshold"], errors="coerce").replace(0.0, np.nan)
        )
        work = batch_df.copy()
        work["template_distance_ratio"] = ratio

        per_class: Dict[str, Dict[str, float | int]] = {}
        for label in sorted(work.loc[known, "true_majority_label"].astype(np.int64).unique().tolist()):
            mask = known & (work["true_majority_label"].astype(np.int64) == int(label))
            sub = work.loc[mask]
            if len(sub) == 0:
                continue
            true_label = sub["true_majority_label"].astype(np.int64)
            pred_label = sub["batch_pred_label"].astype(np.int64)
            nearest_label = sub["nearest_template_label"].astype(np.int64)
            per_class[str(label)] = {
                "num_batches": int(len(sub)),
                "acc": float((pred_label == true_label).mean()),
                "reject_as_unknown_rate": float((pred_label == UNKNOWN_LABEL).mean()),
                "nearest_label_hit_rate": float((nearest_label == true_label).mean()),
                "ratio_p50": _quantile(sub["template_distance_ratio"].to_numpy(dtype=np.float64), 0.50),
                "ratio_p75": _quantile(sub["template_distance_ratio"].to_numpy(dtype=np.float64), 0.75),
                "margin_p50": _quantile(pd.to_numeric(sub["template_margin"], errors="coerce").to_numpy(dtype=np.float64), 0.50),
            }
        out["per_class_known"] = per_class

        unknown_sub = work.loc[unknown]
        unknown_total = int(len(unknown_sub))
        false_accept_by_label: Dict[str, Dict[str, float | int]] = {}
        nearest_unknown_by_label: Dict[str, Dict[str, float | int]] = {}
        if unknown_total:
            false_accept = unknown_sub[unknown_sub["batch_pred_label"].astype(np.int64) != UNKNOWN_LABEL]
            for label, sub in false_accept.groupby("batch_pred_label", sort=True):
                false_accept_by_label[str(int(label))] = {
                    "num_batches": int(len(sub)),
                    "rate": float(len(sub) / unknown_total),
                }
            for label, sub in unknown_sub.groupby("nearest_template_label", sort=True):
                nearest_unknown_by_label[str(int(label))] = {
                    "num_batches": int(len(sub)),
                    "rate": float(len(sub) / unknown_total),
                    "ratio_p25": _quantile(sub["template_distance_ratio"].to_numpy(dtype=np.float64), 0.25),
                    "ratio_p50": _quantile(sub["template_distance_ratio"].to_numpy(dtype=np.float64), 0.50),
                    "ratio_min": float(np.nanmin(sub["template_distance_ratio"].to_numpy(dtype=np.float64))),
                }
        out["unknown_false_accept_by_pred_label"] = false_accept_by_label
        out["unknown_nearest_label"] = nearest_unknown_by_label

    truth_labels = truth["LABEL"].to_numpy(dtype=np.int64)
    if len(truth_labels) == len(pulse_labels):
        out["pulse_label_acc"] = float(np.mean(pulse_labels.astype(np.int64) == truth_labels))
        known_pulse = truth_labels != UNKNOWN_LABEL
        unknown_pulse = truth_labels == UNKNOWN_LABEL
        out["known_pulse_acc"] = float(np.mean(pulse_labels[known_pulse].astype(np.int64) == truth_labels[known_pulse])) if bool(np.any(known_pulse)) else float("nan")
        out["unknown_pulse_reject_rate"] = float(np.mean(pulse_labels[unknown_pulse].astype(np.int64) == UNKNOWN_LABEL)) if bool(np.any(unknown_pulse)) else float("nan")
    return out


def print_class_diagnostics(metrics: Dict[str, object]) -> None:
    per_class = metrics.get("per_class_known", {})
    if isinstance(per_class, dict) and per_class:
        print("  per true known label:")
        print("    label batches acc    reject99 nearest_hit ratio_p50 ratio_p75 margin_p50")
        for label, row in sorted(per_class.items(), key=lambda item: int(item[0])):
            if not isinstance(row, dict):
                continue
            print(
                f"    {int(label):>5} "
                f"{int(row.get('num_batches', 0)):>7} "
                f"{float(row.get('acc', float('nan'))):>6.4f} "
                f"{float(row.get('reject_as_unknown_rate', float('nan'))):>8.4f} "
                f"{float(row.get('nearest_label_hit_rate', float('nan'))):>11.4f} "
                f"{float(row.get('ratio_p50', float('nan'))):>9.3f} "
                f"{float(row.get('ratio_p75', float('nan'))):>9.3f} "
                f"{float(row.get('margin_p50', float('nan'))):>10.3f}"
            )

    false_accept = metrics.get("unknown_false_accept_by_pred_label", {})
    if isinstance(false_accept, dict):
        print("  true unknown accepted as known:")
        if false_accept:
            print("    label batches rate")
            for label, row in sorted(false_accept.items(), key=lambda item: int(item[0])):
                if not isinstance(row, dict):
                    continue
                print(
                    f"    {int(label):>5} "
                    f"{int(row.get('num_batches', 0)):>7} "
                    f"{float(row.get('rate', float('nan'))):>6.4f}"
                )
        else:
            print("    none")


def default_paths(data_root: Path, sample: str, sort_root: Path) -> Tuple[Path, Path, Path]:
    sample_dir = "Sample_1" if sample == "sample1" else "Sample_2"
    pdw_file = data_root / "Test_Data" / sample_dir / "Merge_PDW_Data.txt"
    truth_file = data_root / "Test_Data" / sample_dir / "Sorted_PDW.txt"
    sort_file = sort_root / sample / f"{sample}_sort.txt"
    return pdw_file, sort_file, truth_file


def resolve_relative_to_project(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def sample_override_paths(args: argparse.Namespace, sample: str) -> Tuple[Path | None, Path | None, Path | None]:
    if sample == "sample1":
        pdw_file = str(getattr(args, "sample1_pdw_file", "")).strip()
        sort_file = str(getattr(args, "sample1_sort_file", "")).strip()
        truth_file = str(getattr(args, "sample1_truth_file", "")).strip()
    else:
        pdw_file = str(getattr(args, "sample2_pdw_file", "")).strip()
        sort_file = str(getattr(args, "sample2_sort_file", "")).strip()
        truth_file = str(getattr(args, "sample2_truth_file", "")).strip()
    return (
        resolve_relative_to_project(pdw_file) if pdw_file else None,
        resolve_relative_to_project(sort_file) if sort_file else None,
        resolve_relative_to_project(truth_file) if truth_file else None,
    )


def resolve_input_paths(args: argparse.Namespace, sample: str) -> Tuple[Path, Path, Path | None, Path]:
    if args.pdw_file and args.sort_file:
        pdw_file = resolve_relative_to_project(args.pdw_file)
        sort_file = resolve_relative_to_project(args.sort_file)
        truth_file = resolve_relative_to_project(args.truth_file) if args.truth_file else None
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        return pdw_file, sort_file, truth_file, output_dir

    override_pdw, override_sort, override_truth = sample_override_paths(args, sample)
    if override_pdw is not None and override_sort is not None:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        return override_pdw, override_sort, override_truth, output_root / sample

    data_root = Path(args.root).resolve()
    sort_root = Path(args.sort_root)
    if not sort_root.is_absolute():
        sort_root = PROJECT_ROOT / sort_root
    pdw_file, sort_file, truth_file = default_paths(data_root, sample, sort_root)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_dir = output_root / sample
    return pdw_file, sort_file, truth_file, output_dir


def require_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_one(sample: str, args: argparse.Namespace) -> Dict[str, object]:
    pdw_file, sort_file, truth_file, output_dir = resolve_input_paths(args, sample)
    output_dir.mkdir(parents=True, exist_ok=True)
    require_exists(pdw_file, "PDW file")
    require_exists(sort_file, "sort file")

    pdw = read_pdw(pdw_file, int(args.max_pulses))
    pred_sigidx = read_sort_sigidx(sort_file, int(args.max_pulses))
    if len(pdw) != len(pred_sigidx):
        raise ValueError(f"PDW/sort row mismatch: {len(pdw)} vs {len(pred_sigidx)}")

    library_path = Path(args.template_library)
    if not library_path.is_absolute():
        library_path = PROJECT_ROOT / library_path
    templates, metadata = load_template_library(library_path)
    threshold_scale, label_scales, min_margin, class_floor_scale = resolve_matching_parameters(args, library_path)
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
    labels = labels_from_batches(pred_sigidx, pred_batches)
    metrics: Dict[str, object] = {}
    if truth_file is not None and truth_file.exists():
        truth = read_truth(truth_file, int(args.max_pulses))
        if len(truth) != len(pred_sigidx):
            raise ValueError(f"Truth/sort row mismatch: {len(truth)} vs {len(pred_sigidx)}")
        pred_batches = add_truth_to_batches(pred_batches, truth, pred_sigidx)
        metrics = compute_recognition_metrics(pred_batches, labels, truth)

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
        "truth_file": str(truth_file) if truth_file is not None else "",
        "template_library": str(library_path),
        "num_batches": int(len(pred_batches)),
        "num_unknown_batches": int((pred_batches["batch_pred_label"] == UNKNOWN_LABEL).sum()),
        "pred_label_counts": {str(k): int(v) for k, v in pred_batches["batch_pred_label"].value_counts().sort_index().items()},
        "metrics": metrics,
        "threshold_scale": float(threshold_scale),
        "label_threshold_scales": {str(k): float(v) for k, v in label_scales.items()},
        "class_threshold_floor_scale": float(class_floor_scale),
        "min_margin": float(min_margin),
        "matching_mode": str(args.matching_mode),
        "class_ratio_margin": float(args.class_ratio_margin),
        "enable_label2_rescue": bool(args.enable_label2_rescue),
        "label2_rescue_ratio": float(args.label2_rescue_ratio),
        "label2_feature_padding": float(args.label2_feature_padding),
        "secondary_reject_ratio_caps": {
            str(k): float(v) for k, v in parse_label_float_map(args.secondary_reject_ratio_caps).items()
        },
        "enable_topk_label_rescue": bool(args.enable_topk_label_rescue),
        "topk_rescue_label": int(args.topk_rescue_label),
        "topk_size": int(args.topk_size),
        "topk_min_votes": int(args.topk_min_votes),
        "topk_max_ratio": float(args.topk_max_ratio),
        "topk_feature_padding": float(args.topk_feature_padding),
        "enable_class_ratio_label_rescue": bool(args.enable_class_ratio_label_rescue),
        "class_ratio_rescue_label": int(args.class_ratio_rescue_label),
        "class_ratio_rescue_max_ratio": float(args.class_ratio_rescue_max_ratio),
        "class_ratio_rescue_max_delta": float(args.class_ratio_rescue_max_delta),
        "class_ratio_rescue_feature_padding": float(args.class_ratio_rescue_feature_padding),
        "output_file": str(final_path),
        "batch_file": str(batch_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if metrics:
        print(
            f"[done] {sample}: batches={summary['num_batches']}, unknown={summary['num_unknown_batches']}, "
            f"batch_acc={float(metrics.get('batch_acc', float('nan'))):.4f}, "
            f"known_batch_acc={float(metrics.get('known_batch_acc', float('nan'))):.4f}, "
            f"unknown_reject={float(metrics.get('unknown_reject_rate', float('nan'))):.4f}"
        )
        print_class_diagnostics(metrics)
    else:
        print(f"[done] {sample}: batches={summary['num_batches']}, unknown={summary['num_unknown_batches']}, output={final_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Template matching recognition for sorted SigIdx outputs.")
    parser.add_argument("--root", type=Path, default=Path("edata"))
    parser.add_argument("--sample", choices=["sample1", "sample2", "all"], default="all")
    parser.add_argument("--sort_root", type=Path, default=Path("outputs_performance_first"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs_template_match_recognition"))
    parser.add_argument("--pdw_file", type=str, default="")
    parser.add_argument("--sort_file", type=str, default="")
    parser.add_argument("--truth_file", type=str, default="")
    parser.add_argument("--sample1_pdw_file", type=str, default="edata/Test_Data/Sample_1/Merge_PDW_Data.txt")
    parser.add_argument("--sample1_sort_file", type=str, default="outputs_best_front_tracklet_graph/sample1/sample1_sort.txt")
    parser.add_argument("--sample1_truth_file", type=str, default="edata/Test_Data/Sample_1/Sorted_PDW.txt")
    parser.add_argument("--sample2_pdw_file", type=str, default="edata/Test_Data/Sample_2/Merge_PDW_Data.txt")
    parser.add_argument("--sample2_sort_file", type=str, default="outputs_best_front_tracklet_graph/sample2/sample2_sort.txt")
    parser.add_argument("--sample2_truth_file", type=str, default="edata/Test_Data/Sample_2/Sorted_PDW.txt")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_template_match_recognition/custom"))
    parser.add_argument("--template_library", type=Path, default=Path("outputs_expanded_template_library/template_library.json"))
    parser.add_argument("--max_pulses", type=int, default=0)
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

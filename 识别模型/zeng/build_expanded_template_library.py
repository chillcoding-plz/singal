#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an expanded template library from Class_*.txt files.

Compared with the baseline template builder, this script adds:
1. time-window templates
2. Param2 state templates
3. PRI state templates aggregated from time-window states
4. class-internal cluster prototypes aggregated from time-window templates
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


DEFAULT_FEATURES = [
    "p1_median",
    "p2_median",
    "p4_median",
    "pri_median_us",
]

DEFAULT_CLUSTER_FEATURES = [
    "p1_median",
    "p2_median",
    "p4_median",
    "pri_median_us",
    "p1_iqr",
    "p2_iqr",
    "p4_iqr",
    "pri_iqr_us",
]

DEFAULT_WEIGHTS = {
    "p1_median": 1.5,
    "p2_median": 1.5,
    "p4_median": 1.5,
    "pri_median_us": 2.0,
    "p1_iqr": 0.5,
    "p2_iqr": 0.5,
    "p4_iqr": 0.5,
    "pri_iqr_us": 1.0,
    "pri_cv": 0.8,
    "log_num_pulses": 0.3,
    "log_duration": 0.3,
    "log_pulse_rate": 0.5,
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def parse_feature_list(text: str) -> List[str]:
    return [item.strip() for item in str(text).replace("|", ",").split(",") if item.strip()]


def parse_weight_overrides(text: str) -> Dict[str, float]:
    overrides: Dict[str, float] = {}
    if not str(text).strip():
        return overrides
    for item in str(text).split(","):
        if not item.strip():
            continue
        name, value = item.split(":", 1)
        overrides[name.strip()] = float(value.strip())
    return overrides


def parse_bins(text: str) -> List[Tuple[int, float, float]]:
    bins = []
    for idx, item in enumerate(str(text).split(",")):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        bins.append((idx, float(left), float(right)))
    return bins


def class_label_from_path(path: Path) -> int:
    stem = path.stem.lower().replace("class_", "").replace("class", "")
    return int(stem)


def read_pdw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    df = df.iloc[:, :8].copy()
    df.columns = ["TOA(s)", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6", "Param7"]
    return df


def robust_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "q25": 0.0, "q75": 0.0, "iqr": 0.0}
    q25, median, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
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
    quantile_cap = float(np.quantile(dtoa, gap_quantile))
    cap = max(float(gap_multiplier) * max(median, 1e-9), quantile_cap)
    return dtoa[dtoa <= cap]


def time_window_ids(toa_s: np.ndarray, window_seconds: float) -> np.ndarray:
    toa = np.asarray(toa_s, dtype=np.float64)
    start = float(np.min(toa))
    return np.floor((toa - start) / float(window_seconds)).astype(np.int64)


def extract_template_features(
    sub: pd.DataFrame,
    label: int,
    template_id: str,
    source_file: str,
    source_kind: str,
    source_detail: str,
    gap_multiplier: float,
    gap_quantile: float,
) -> Dict[str, float | int | str]:
    toa = sub["TOA(s)"].to_numpy(dtype=np.float64)
    duration = float(max(np.max(toa) - np.min(toa), 0.0))
    pri = robust_pri_us(toa, gap_multiplier, gap_quantile)
    pri_stats = robust_stats(pri)
    row: Dict[str, float | int | str] = {
        "template_id": template_id,
        "label": int(label),
        "source_file": source_file,
        "source_kind": source_kind,
        "source_detail": source_detail,
        "start_toa_s": float(np.min(toa)),
        "end_toa_s": float(np.max(toa)),
        "num_pulses": int(len(sub)),
        "log_num_pulses": float(math.log1p(len(sub))),
        "duration_s": duration,
        "log_duration": float(math.log1p(duration)),
        "pulse_rate": float(len(sub) / max(duration, 1e-9)),
        "log_pulse_rate": float(math.log1p(len(sub) / max(duration, 1e-9))),
        "pri_median_us": pri_stats["median"],
        "pri_iqr_us": pri_stats["iqr"],
        "pri_cv": float(pri_stats["std"] / max(pri_stats["mean"], 1e-9)) if len(pri) else 0.0,
    }
    for param, prefix in [("Param1", "p1"), ("Param2", "p2"), ("Param3", "p3"), ("Param4", "p4")]:
        stats = robust_stats(sub[param].to_numpy(dtype=np.float64))
        for name, value in stats.items():
            row[f"{prefix}_{name}"] = value
    return row


def aggregate_template_rows(
    sub_templates: pd.DataFrame,
    label: int,
    template_id: str,
    source_file: str,
    source_kind: str,
    source_detail: str,
) -> Dict[str, float | int | str]:
    numeric_cols = [c for c in sub_templates.columns if c not in {"template_id", "label", "source_file", "source_kind", "source_detail"}]
    row: Dict[str, float | int | str] = {
        "template_id": template_id,
        "label": int(label),
        "source_file": source_file,
        "source_kind": source_kind,
        "source_detail": source_detail,
        "start_toa_s": float(pd.to_numeric(sub_templates["start_toa_s"], errors="coerce").min()),
        "end_toa_s": float(pd.to_numeric(sub_templates["end_toa_s"], errors="coerce").max()),
        "num_pulses": int(pd.to_numeric(sub_templates["num_pulses"], errors="coerce").sum()),
    }
    sum_cols = {"num_pulses"}
    min_cols = {"start_toa_s"}
    max_cols = {"end_toa_s"}
    for col in numeric_cols:
        if col in sum_cols or col in min_cols or col in max_cols:
            continue
        values = pd.to_numeric(sub_templates[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        row[col] = float(values.median()) if len(values) else 0.0
    return row


def build_time_window_templates_from_df(
    df: pd.DataFrame,
    label: int,
    args: argparse.Namespace,
    source_file: str,
    source_kind: str = "time_window",
) -> pd.DataFrame:
    window_ids = time_window_ids(df["TOA(s)"].to_numpy(dtype=np.float64), float(args.window_seconds))
    rows = []
    for window_id in sorted(np.unique(window_ids)):
        sub = df[window_ids == window_id]
        if len(sub) < int(args.min_pulses):
            continue
        template_id = f"class{label}_w{int(window_id):04d}"
        rows.append(
            extract_template_features(
                sub,
                label=label,
                template_id=template_id,
                source_file=source_file,
                source_kind=source_kind,
                source_detail=f"window_id={int(window_id)}",
                gap_multiplier=float(args.pri_gap_multiplier),
                gap_quantile=float(args.pri_gap_quantile),
            )
        )
    return pd.DataFrame(rows)


def build_param2_templates_from_df(
    df: pd.DataFrame,
    label: int,
    args: argparse.Namespace,
    source_file: str,
) -> pd.DataFrame:
    rows = []
    for mode_id, low, high in parse_bins(args.param2_mode_bins):
        sub = df[(df["Param2"] >= low) & (df["Param2"] < high)]
        if len(sub) < int(args.min_mode_pulses):
            continue
        template_id = f"class{label}_p2m{mode_id:02d}"
        rows.append(
            extract_template_features(
                sub,
                label=label,
                template_id=template_id,
                source_file=source_file,
                source_kind="param2_mode",
                source_detail=f"{low:.3f}<={high:.3f}",
                gap_multiplier=float(args.pri_gap_multiplier),
                gap_quantile=float(args.pri_gap_quantile),
            )
        )
    return pd.DataFrame(rows)


def build_pri_templates_from_windows(
    window_templates: pd.DataFrame,
    label: int,
    args: argparse.Namespace,
    source_file: str,
) -> pd.DataFrame:
    if len(window_templates) == 0:
        return pd.DataFrame()
    rows = []
    values = pd.to_numeric(window_templates["pri_median_us"], errors="coerce")
    for mode_id, low, high in parse_bins(args.pri_mode_bins):
        sub = window_templates[(values >= low) & (values < high)]
        if len(sub) < int(args.min_group_templates):
            continue
        template_id = f"class{label}_prim{mode_id:02d}"
        rows.append(
            aggregate_template_rows(
                sub,
                label=label,
                template_id=template_id,
                source_file=source_file,
                source_kind="pri_mode",
                source_detail=f"{low:.3f}<={high:.3f}",
            )
        )
    return pd.DataFrame(rows)


def robust_center_scale(templates: pd.DataFrame, features: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    centers: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    for name in features:
        values = pd.to_numeric(templates[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) == 0:
            centers[name] = 0.0
            scales[name] = 1.0
            continue
        q25, median, q75 = np.percentile(values.to_numpy(dtype=np.float64), [25, 50, 75])
        centers[name] = float(median)
        scales[name] = float(max(q75 - q25, 1e-6))
    return centers, scales


def scaled_matrix(
    table: pd.DataFrame,
    features: List[str],
    centers: Dict[str, float],
    scales: Dict[str, float],
    weights: Dict[str, float],
) -> np.ndarray:
    cols = []
    for name in features:
        values = pd.to_numeric(table[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        normalized = (values.fillna(centers[name]).to_numpy(dtype=np.float64) - centers[name]) / scales[name]
        cols.append(normalized * math.sqrt(max(float(weights.get(name, 1.0)), 0.0)))
    return np.vstack(cols).T.astype(np.float64)


def build_cluster_templates_from_windows(
    window_templates: pd.DataFrame,
    label: int,
    args: argparse.Namespace,
    source_file: str,
) -> pd.DataFrame:
    if len(window_templates) < max(int(args.min_group_templates), 3) or not args.enable_cluster_templates:
        return pd.DataFrame()
    features = [name for name in parse_feature_list(args.cluster_template_features) if name in window_templates.columns]
    if not features:
        return pd.DataFrame()
    weights = {name: float(DEFAULT_WEIGHTS.get(name, 1.0)) for name in features}
    centers, scales = robust_center_scale(window_templates, features)
    x = scaled_matrix(window_templates, features, centers, scales, weights)

    max_k = min(int(args.max_cluster_templates_per_class), max(1, len(window_templates) // max(int(args.min_cluster_members), 1)))
    if max_k < 2:
        return pd.DataFrame()
    n_clusters = max_k
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=int(args.seed))
    cluster_ids = km.fit_predict(x)
    rows = []
    for cluster_id in range(n_clusters):
        sub = window_templates[cluster_ids == cluster_id]
        if len(sub) < int(args.min_cluster_members):
            continue
        template_id = f"class{label}_cl{cluster_id:02d}"
        rows.append(
            aggregate_template_rows(
                sub,
                label=label,
                template_id=template_id,
                source_file=source_file,
                source_kind="cluster_proto",
                source_detail=f"members={len(sub)}",
            )
        )
    return pd.DataFrame(rows)


def build_templates_for_label_df(
    df: pd.DataFrame,
    label: int,
    args: argparse.Namespace,
    source_file: str,
) -> pd.DataFrame:
    tables = []
    windows = build_time_window_templates_from_df(df, label, args, source_file)
    if len(windows):
        tables.append(windows)
        pri_modes = build_pri_templates_from_windows(windows, label, args, source_file)
        if len(pri_modes):
            tables.append(pri_modes)
        clusters = build_cluster_templates_from_windows(windows, label, args, source_file)
        if len(clusters):
            tables.append(clusters)
    p2_modes = build_param2_templates_from_df(df, label, args, source_file)
    if len(p2_modes):
        tables.append(p2_modes)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def build_templates_for_class(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    return build_templates_for_label_df(read_pdw(path), class_label_from_path(path), args, str(path))


def class_thresholds(
    templates: pd.DataFrame,
    features: List[str],
    centers: Dict[str, float],
    scales: Dict[str, float],
    weights: Dict[str, float],
    quantile: float,
    iqr_multiplier: float,
    min_threshold: float,
) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    x_all = scaled_matrix(templates, features, centers, scales, weights)
    labels = templates["label"].to_numpy(dtype=np.int64)
    for label in sorted(np.unique(labels)):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            thresholds[int(label)] = float(min_threshold)
            continue
        x = x_all[idx]
        distances = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        distances[distances <= 1e-12] = np.inf
        nearest = np.min(distances, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if len(nearest) == 0:
            thresholds[int(label)] = float(min_threshold)
            continue
        q = float(np.quantile(nearest, quantile))
        spread = float(np.percentile(nearest, 75) - np.percentile(nearest, 25)) if len(nearest) >= 4 else 0.0
        thresholds[int(label)] = max(float(min_threshold), q + float(iqr_multiplier) * spread)
    return thresholds


def add_template_local_thresholds(
    templates: pd.DataFrame,
    features: List[str],
    centers: Dict[str, float],
    scales: Dict[str, float],
    weights: Dict[str, float],
    radius_multiplier: float,
    min_threshold: float,
) -> pd.DataFrame:
    out = templates.copy()
    x_all = scaled_matrix(out, features, centers, scales, weights)
    labels = out["label"].to_numpy(dtype=np.int64)
    same_nn = np.full(len(out), np.inf, dtype=np.float64)
    other_nn = np.full(len(out), np.inf, dtype=np.float64)
    for i in range(len(out)):
        distances = np.linalg.norm(x_all - x_all[i], axis=1)
        same_mask = (labels == labels[i]) & (np.arange(len(out)) != i)
        other_mask = labels != labels[i]
        if np.any(same_mask):
            same_nn[i] = float(np.min(distances[same_mask]))
        if np.any(other_mask):
            other_nn[i] = float(np.min(distances[other_mask]))
    finite_same = same_nn[np.isfinite(same_nn)]
    fallback = float(np.median(finite_same)) if len(finite_same) else float(min_threshold)
    same_nn = np.where(np.isfinite(same_nn), same_nn, fallback)
    raw_local_threshold = np.maximum(float(min_threshold), same_nn * float(radius_multiplier))
    other_cap = np.where(np.isfinite(other_nn), np.maximum(float(min_threshold), other_nn * 0.8), np.inf)
    local_threshold = np.minimum(raw_local_threshold, other_cap)
    out["same_class_nn_distance"] = same_nn
    out["nearest_other_class_distance"] = other_nn
    out["raw_local_distance_threshold"] = raw_local_threshold
    out["local_distance_threshold"] = local_threshold
    out["nearest_margin"] = other_nn - same_nn
    out["ambiguous_template"] = out["nearest_margin"] <= 0.0
    return out


def class_summary(templates: pd.DataFrame, features: List[str]) -> Dict[int, Dict[str, object]]:
    out: Dict[int, Dict[str, object]] = {}
    for label, sub in templates.groupby("label", sort=True):
        kind_counts = {str(k): int(v) for k, v in sub["source_kind"].value_counts().sort_index().items()}
        feature_summary = {}
        for name in features:
            values = pd.to_numeric(sub[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values):
                feature_summary[name] = {
                    "median": float(values.median()),
                    "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
        out[int(label)] = {
            "num_templates": int(len(sub)),
            "num_pulses_total": int(sub["num_pulses"].sum()),
            "template_source_counts": kind_counts,
            "features": feature_summary,
        }
    return out


def build_library(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_dir = Path(args.train_dir)
    class_files = sorted(train_dir.glob("Class_*.txt"))
    if not class_files:
        raise FileNotFoundError(f"No Class_*.txt files found in {train_dir}")
    tables = []
    for path in class_files:
        table = build_templates_for_class(path, args)
        if len(table):
            tables.append(table)
    if not tables:
        raise ValueError("No valid templates were built.")
    templates = pd.concat(tables, ignore_index=True)
    features = [name for name in parse_feature_list(args.features) if name in templates.columns]
    if not features:
        raise ValueError("No requested features are present in templates.")
    weights = {name: float(DEFAULT_WEIGHTS.get(name, 1.0)) for name in features}
    weights.update(parse_weight_overrides(args.feature_weights))
    centers, scales = robust_center_scale(templates, features)
    thresholds = class_thresholds(
        templates,
        features,
        centers,
        scales,
        weights,
        quantile=float(args.threshold_quantile),
        iqr_multiplier=float(args.threshold_iqr_multiplier),
        min_threshold=float(args.min_threshold),
    )
    templates = add_template_local_thresholds(
        templates,
        features,
        centers,
        scales,
        weights,
        radius_multiplier=float(args.local_threshold_multiplier),
        min_threshold=float(args.min_threshold),
    )
    metadata = {
        "method": "expanded_multi_template_matching_library",
        "train_dir": str(train_dir),
        "window_seconds": float(args.window_seconds),
        "min_pulses": int(args.min_pulses),
        "min_mode_pulses": int(args.min_mode_pulses),
        "min_group_templates": int(args.min_group_templates),
        "pri_gap_multiplier": float(args.pri_gap_multiplier),
        "pri_gap_quantile": float(args.pri_gap_quantile),
        "features": features,
        "feature_weights": weights,
        "feature_centers": centers,
        "feature_scales": scales,
        "class_thresholds": thresholds,
        "template_sources": {
            "time_window": True,
            "param2_mode_bins": args.param2_mode_bins,
            "pri_mode_bins": args.pri_mode_bins,
            "cluster_templates": bool(args.enable_cluster_templates),
            "cluster_template_features": parse_feature_list(args.cluster_template_features),
        },
        "threshold_rule": {
            "recommended_mode": "nearest_template_local_threshold",
            "nearest_template_distance_quantile": float(args.threshold_quantile),
            "iqr_multiplier": float(args.threshold_iqr_multiplier),
            "min_threshold": float(args.min_threshold),
            "local_threshold_multiplier": float(args.local_threshold_multiplier),
        },
        "class_summary": class_summary(templates, features),
    }
    return templates, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an expanded template library from Class_*.txt files.")
    parser.add_argument("--train_dir", type=Path, default=Path("E:/分选/Data/Train_Data"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_expanded_template_library"))
    parser.add_argument("--window_seconds", type=float, default=0.2)
    parser.add_argument("--min_pulses", type=int, default=200)
    parser.add_argument("--min_mode_pulses", type=int, default=100)
    parser.add_argument("--min_group_templates", type=int, default=2)
    parser.add_argument("--param2_mode_bins", type=str, default="0:1.2,1.2:1.8,1.8:3,3:5,5:10,10:30")
    parser.add_argument("--pri_mode_bins", type=str, default="0:6,6:15,15:40,40:120,120:400,400:4000")
    parser.add_argument("--enable_cluster_templates", action="store_true")
    parser.add_argument("--cluster_template_features", type=str, default=",".join(DEFAULT_CLUSTER_FEATURES))
    parser.add_argument("--max_cluster_templates_per_class", type=int, default=3)
    parser.add_argument("--min_cluster_members", type=int, default=2)
    parser.add_argument("--pri_gap_multiplier", type=float, default=5.0)
    parser.add_argument("--pri_gap_quantile", type=float, default=0.90)
    parser.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--feature_weights", type=str, default="")
    parser.add_argument("--threshold_quantile", type=float, default=0.95)
    parser.add_argument("--threshold_iqr_multiplier", type=float, default=1.5)
    parser.add_argument("--min_threshold", type=float, default=0.25)
    parser.add_argument("--local_threshold_multiplier", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    templates, metadata = build_library(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "template_library.csv"
    json_path = args.output_dir / "template_library.json"
    templates.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps({"metadata": json_safe(metadata), "templates": json_safe(templates.to_dict(orient="records"))}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] templates: {len(templates)}")
    print(f"[done] labels: {sorted(templates['label'].astype(int).unique().tolist())}")
    print(f"[done] csv: {csv_path}")
    print(f"[done] json: {json_path}")


if __name__ == "__main__":
    main()

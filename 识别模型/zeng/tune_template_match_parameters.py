#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tune template match thresholds using train-only holdout windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import build_expanded_template_library as lib


UNKNOWN_LABEL = 99


def weighted_scaled_matrix(
    table: pd.DataFrame,
    features: List[str],
    centers: Dict[str, float],
    scales: Dict[str, float],
    weights: Dict[str, float],
) -> np.ndarray:
    return lib.scaled_matrix(table, features, centers, scales, weights)


def match_queries(
    queries: pd.DataFrame,
    templates: pd.DataFrame,
    features: List[str],
    centers: Dict[str, float],
    scales: Dict[str, float],
    weights: Dict[str, float],
) -> pd.DataFrame:
    qx = weighted_scaled_matrix(queries, features, centers, scales, weights)
    tx = weighted_scaled_matrix(templates, features, centers, scales, weights)
    d = np.linalg.norm(qx[:, None, :] - tx[None, :, :], axis=2)
    order = np.argsort(d, axis=1)
    nearest_idx = order[:, 0]
    second_idx = order[:, 1] if d.shape[1] > 1 else order[:, 0]
    nearest = templates.iloc[nearest_idx].reset_index(drop=True)
    out = queries.copy().reset_index(drop=True)
    out["nearest_template_label"] = nearest["label"].astype(np.int64).to_numpy()
    out["template_distance"] = d[np.arange(len(out)), nearest_idx]
    out["template_distance_threshold"] = pd.to_numeric(nearest["local_distance_threshold"], errors="coerce").fillna(np.inf).to_numpy(dtype=np.float64)
    out["template_margin"] = d[np.arange(len(out)), second_idx] - out["template_distance"].to_numpy(dtype=np.float64)
    return out


def augment_queries(
    queries: pd.DataFrame,
    features: List[str],
    scales: Dict[str, float],
    num_copies: int,
    noise_scale: float,
    seed: int,
) -> pd.DataFrame:
    if num_copies <= 0 or noise_scale <= 0.0 or len(queries) == 0:
        return queries
    rng = np.random.default_rng(seed)
    tables = [queries]
    for copy_id in range(num_copies):
        aug = queries.copy()
        for name in features:
            base_scale = max(float(scales.get(name, 1.0)), 1e-9)
            noise = rng.normal(0.0, base_scale * float(noise_scale), size=len(aug))
            aug[name] = pd.to_numeric(aug[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) + noise
        aug["source_kind"] = aug.get("source_kind", "holdout").astype(str) + f"_aug{copy_id + 1}"
        tables.append(aug)
    return pd.concat(tables, ignore_index=True)


def build_holdout_sets(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    train_dir = Path(args.train_dir)
    class_files = sorted(train_dir.glob("Class_*.txt"))
    tables = []
    query_tables = []
    for path in class_files:
        label = lib.class_label_from_path(path)
        df = lib.read_pdw(path)
        window_ids = lib.time_window_ids(df["TOA(s)"].to_numpy(dtype=np.float64), float(args.window_seconds))
        build_df = df[window_ids % 2 == 0].copy()
        holdout_df = df[window_ids % 2 == 1].copy()
        build_table = lib.build_templates_for_label_df(build_df, label, args, str(path))
        if len(build_table):
            tables.append(build_table)
        holdout_windows = pd.DataFrame()
        for size in [float(v) for v in str(args.holdout_window_sizes).split(",") if str(v).strip()]:
            query_args = argparse.Namespace(**vars(args))
            query_args.window_seconds = float(size)
            query_table = lib.build_time_window_templates_from_df(holdout_df, label, query_args, str(path), source_kind=f"holdout_window_{size:g}s")
            if len(query_table):
                query_table["true_label"] = int(label)
                query_tables.append(query_table)
                if abs(size - float(args.window_seconds)) < 1e-9:
                    holdout_windows = query_table.copy()
        holdout_p2 = lib.build_param2_templates_from_df(holdout_df, label, args, str(path))
        if len(holdout_p2):
            holdout_p2["true_label"] = int(label)
            query_tables.append(holdout_p2)
        if len(holdout_windows):
            holdout_pri = lib.build_pri_templates_from_windows(holdout_windows, label, args, str(path))
            if len(holdout_pri):
                holdout_pri["true_label"] = int(label)
                query_tables.append(holdout_pri)
    if not tables or not query_tables:
        raise ValueError("Holdout split produced empty templates or queries.")
    templates = pd.concat(tables, ignore_index=True)
    features = [name for name in lib.parse_feature_list(args.features) if name in templates.columns]
    weights = {name: float(lib.DEFAULT_WEIGHTS.get(name, 1.0)) for name in features}
    weights.update(lib.parse_weight_overrides(args.feature_weights))
    centers, scales = lib.robust_center_scale(templates, features)
    templates = lib.add_template_local_thresholds(
        templates,
        features,
        centers,
        scales,
        weights,
        radius_multiplier=float(args.local_threshold_multiplier),
        min_threshold=float(args.min_threshold),
    )
    queries = pd.concat(query_tables, ignore_index=True)
    return templates, queries, {"features": features, "weights": weights, "centers": centers, "scales": scales}


def evaluate_parameters(
    matched: pd.DataFrame,
    threshold_scale: float,
    label_threshold_scales: Dict[int, float],
    class_threshold_floor_scale: float,
    min_margin: float,
) -> Dict[str, object]:
    y = matched["true_label"].astype(np.int64).to_numpy()
    nearest_label = matched["nearest_template_label"].astype(np.int64).to_numpy()
    thresholds = matched["template_distance_threshold"].to_numpy(dtype=np.float64)
    distances = matched["template_distance"].to_numpy(dtype=np.float64)
    margins = matched["template_margin"].to_numpy(dtype=np.float64)
    class_thresholds = matched["class_threshold"].to_numpy(dtype=np.float64)
    mult = np.array([float(label_threshold_scales.get(int(v), 1.0)) for v in nearest_label], dtype=np.float64)
    applied = np.maximum(thresholds * float(threshold_scale) * mult, class_thresholds * float(class_threshold_floor_scale))
    accepted = (distances <= applied) & (margins >= float(min_margin))
    pred = np.where(accepted, nearest_label, UNKNOWN_LABEL)

    labels = sorted(np.unique(y).tolist())
    per_class_acc = {}
    for label in labels:
        mask = y == label
        per_class_acc[int(label)] = float(np.mean(pred[mask] == y[mask])) if bool(np.any(mask)) else float("nan")
    macro = float(np.mean(list(per_class_acc.values())))
    overall = float(np.mean(pred == y))
    accept_rate = float(np.mean(pred != UNKNOWN_LABEL))
    return {
        "macro_known_acc": macro,
        "overall_known_acc": overall,
        "accept_rate": accept_rate,
        "per_class_acc": per_class_acc,
        "threshold_scale": float(threshold_scale),
        "label_threshold_scales": {str(k): float(v) for k, v in label_threshold_scales.items()},
        "class_threshold_floor_scale": float(class_threshold_floor_scale),
        "min_margin": float(min_margin),
    }


def tune_parameters(args: argparse.Namespace) -> Dict[str, object]:
    templates, queries, ctx = build_holdout_sets(args)
    queries = augment_queries(
        queries,
        ctx["features"],
        ctx["scales"],
        num_copies=int(args.augment_query_copies),
        noise_scale=float(args.augment_noise_scale),
        seed=int(args.seed),
    )
    matched = match_queries(queries, templates, ctx["features"], ctx["centers"], ctx["scales"], ctx["weights"])
    matched["class_threshold"] = matched["nearest_template_label"].map({int(k): float(v) for k, v in lib.class_thresholds(
        templates,
        ctx["features"],
        ctx["centers"],
        ctx["scales"],
        ctx["weights"],
        quantile=0.95,
        iqr_multiplier=1.5,
        min_threshold=float(args.min_threshold),
    ).items()}).astype(float)
    candidates = [float(v) for v in str(args.scale_candidates).split(",") if str(v).strip()]
    label_candidates = [float(v) for v in str(args.label_scale_candidates).split(",") if str(v).strip()]
    floor_candidates = [float(v) for v in str(args.class_floor_scale_candidates).split(",") if str(v).strip()]
    labels = sorted(templates["label"].astype(int).unique().tolist())

    best = None
    for threshold_scale in candidates:
        if len(labels) != 4:
            raise ValueError("Current tuner expects four known labels.")
        for floor_scale in floor_candidates:
            for s1 in label_candidates:
                for s2 in label_candidates:
                    for s3 in label_candidates:
                        for s4 in label_candidates:
                            params = {labels[0]: s1, labels[1]: s2, labels[2]: s3, labels[3]: s4}
                            metrics = evaluate_parameters(matched, threshold_scale, params, floor_scale, float(args.min_margin))
                            key = (
                                metrics["macro_known_acc"],
                                metrics["overall_known_acc"],
                                metrics["accept_rate"],
                                -threshold_scale,
                                -floor_scale,
                            )
                            if best is None or key > best[0]:
                                best = (key, metrics)
    assert best is not None
    result = best[1]
    result["num_holdout_queries"] = int(len(queries))
    result["num_templates"] = int(len(templates))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune template match thresholds using train-only holdout windows.")
    parser.add_argument("--train_dir", type=Path, default=Path("E:/分选/Data/Train_Data"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_expanded_template_library"))
    parser.add_argument("--window_seconds", type=float, default=0.2)
    parser.add_argument("--min_pulses", type=int, default=200)
    parser.add_argument("--min_mode_pulses", type=int, default=100)
    parser.add_argument("--min_group_templates", type=int, default=2)
    parser.add_argument("--param2_mode_bins", type=str, default="0:1.2,1.2:1.8,1.8:3,3:5,5:10,10:30")
    parser.add_argument("--pri_mode_bins", type=str, default="0:6,6:15,15:40,40:120,120:400,400:4000")
    parser.add_argument("--enable_cluster_templates", action="store_true")
    parser.add_argument("--cluster_template_features", type=str, default=",".join(lib.DEFAULT_CLUSTER_FEATURES))
    parser.add_argument("--max_cluster_templates_per_class", type=int, default=3)
    parser.add_argument("--min_cluster_members", type=int, default=2)
    parser.add_argument("--pri_gap_multiplier", type=float, default=5.0)
    parser.add_argument("--pri_gap_quantile", type=float, default=0.90)
    parser.add_argument("--features", type=str, default=",".join(lib.DEFAULT_FEATURES))
    parser.add_argument("--feature_weights", type=str, default="")
    parser.add_argument("--min_threshold", type=float, default=0.25)
    parser.add_argument("--local_threshold_multiplier", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min_margin", type=float, default=0.0)
    parser.add_argument("--scale_candidates", type=str, default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--label_scale_candidates", type=str, default="0.5,0.75,1.0,1.5,2.0")
    parser.add_argument("--holdout_window_sizes", type=str, default="0.2,0.4,0.6")
    parser.add_argument("--augment_query_copies", type=int, default=2)
    parser.add_argument("--augment_noise_scale", type=float, default=0.15)
    parser.add_argument("--class_floor_scale_candidates", type=str, default="0.0,0.25,0.5,0.75,1.0")
    args = parser.parse_args()

    tuned = tune_parameters(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "tuned_match_parameters.json"
    out_path.write_text(json.dumps(tuned, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] tuning file: {out_path}")
    print(json.dumps(tuned, ensure_ascii=False))


if __name__ == "__main__":
    main()

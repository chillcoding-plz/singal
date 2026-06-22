#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-aware adaptive-distance HDBSCAN with TSR refinement.

The implementation keeps the existing window-tracklet HDBSCAN pipeline, but
adds two research-facing changes:

1. PA-HDBSCAN: build a physics-aware adaptive distance space before HDBSCAN.
   The scalable default is an adaptive embedding. For small windows an exact
   precomputed pairwise distance can be enabled.
2. TSR-HDBSCAN: refine the HDBSCAN output with physical batch reduction,
   noise rescue, and automatic BIC/Bayesian-GMM parameter-band splitting.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import dbscan_sort as dbscan
import hdbscan_conservative_merge_sort as conservative
from 识别 import reduce_sigidx as reducer

try:
    import hdbscan_sort as hdbsort
except ImportError:
    from 分选 import hdbscan_sort as hdbsort


def _ensure_parent(path: Path) -> None:
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)


def _robust_scale(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = raw.astype(np.float64, copy=False)
    median = np.median(raw, axis=0)
    q75 = np.percentile(raw, 75, axis=0)
    q25 = np.percentile(raw, 25, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return ((raw - median) / scale).astype(np.float32), median.astype(np.float32), scale.astype(np.float32)


def _rolling_median_std(values: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    window = max(1, int(window))
    series = pd.Series(values.astype(np.float64))
    median = series.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=np.float64)
    std = series.rolling(window=window, center=True, min_periods=2).std().fillna(0.0).to_numpy(dtype=np.float64)
    return median, std


def _reliability_from_std(local_std: np.ndarray, ref_std: float, strength: float, floor: float) -> np.ndarray:
    ref = max(float(ref_std), 1e-6)
    reliability = 1.0 / (1.0 + max(float(strength), 0.0) * np.maximum(local_std, 0.0) / ref)
    return np.clip(reliability, floor, 1.0).astype(np.float32)


def circular_abs_diff_deg(values: np.ndarray, ref: float) -> np.ndarray:
    return np.abs((values - ref + 180.0) % 360.0 - 180.0)


def _safe_json(value):
    return dbscan.json_safe(value)


# ---------------------------------------------------------------------------
# PA-HDBSCAN front end
# ---------------------------------------------------------------------------


def build_pa_features(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, object]]:
    """Build the adaptive embedding used as the default PA-HDBSCAN distance.

    The exact pairwise metric used by ``build_pa_pairwise_distance`` is a
    weighted L1 physics distance. The embedding here is its scalable surrogate:
    robust-scaled physical features are multiplied by local reliability weights
    and normalized by a local dispersion estimate.
    """
    def report_feature_progress(percent: float, text: str) -> None:
        print(f"[pa-features] {percent:.0f}% {text}", flush=True)

    report_feature_progress(0, "start")
    # 通过局部统计、可靠性建模和稳健缩放，将原始 PDW 参数映射为物理感知的自适应嵌入特征，用于提高后续 PA-HDBSCAN 聚类的鲁棒性与可分性。
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    dtoa_us = np.diff(toa, prepend=toa[0]) * 1e6
    dtoa_us = np.clip(dtoa_us, 0.0, None)
    log_dtoa = np.log1p(dtoa_us)
    local_log_dtoa, local_log_dtoa_std = _rolling_median_std(log_dtoa, args.pa_local_window)
    pri_us = np.expm1(local_log_dtoa)
    pri_us = np.maximum(pri_us, args.min_valid_pri_us)
    period_residual = np.zeros_like(dtoa_us, dtype=np.float64)
    valid_pri = pri_us > args.min_valid_pri_us
    period_residual[valid_pri] = np.abs(dtoa_us[valid_pri] - np.round(dtoa_us[valid_pri] / pri_us[valid_pri]) * pri_us[valid_pri])
    period_residual = np.log1p(np.clip(period_residual, 0.0, None))
    report_feature_progress(20, "timing")

    p1 = df["Param1"].to_numpy(dtype=np.float64)
    p2 = df["Param2"].to_numpy(dtype=np.float64)
    p4 = df["Param4"].to_numpy(dtype=np.float64)
    p5 = df["Param5"].to_numpy(dtype=np.float64)
    angle = np.deg2rad(np.mod(p5, 360.0))
    sin_p5 = np.sin(angle)
    cos_p5 = np.cos(angle)

    _, p1_std = _rolling_median_std(p1, args.pa_local_window)
    _, p2_std = _rolling_median_std(p2, args.pa_local_window)
    _, p4_std = _rolling_median_std(p4, args.pa_local_window)
    _, sin_std = _rolling_median_std(sin_p5, args.pa_local_window)
    _, cos_std = _rolling_median_std(cos_p5, args.pa_local_window)
    angle_std = np.sqrt(sin_std**2 + cos_std**2)
    report_feature_progress(40, "local-statistics")

    rel_floor = float(args.pa_reliability_floor)
    rel_p1 = _reliability_from_std(p1_std, np.median(p1_std[p1_std > 0]) if np.any(p1_std > 0) else 1.0, args.pa_adapt_strength, rel_floor)
    rel_p2 = _reliability_from_std(p2_std, np.median(p2_std[p2_std > 0]) if np.any(p2_std > 0) else 1.0, args.pa_adapt_strength, rel_floor)
    rel_p4 = _reliability_from_std(p4_std, np.median(p4_std[p4_std > 0]) if np.any(p4_std > 0) else 1.0, args.pa_adapt_strength, rel_floor)
    rel_p5 = _reliability_from_std(angle_std, np.median(angle_std[angle_std > 0]) if np.any(angle_std > 0) else 1.0, args.pa_adapt_strength, rel_floor)
    rel_dtoa = _reliability_from_std(
        local_log_dtoa_std,
        np.median(local_log_dtoa_std[local_log_dtoa_std > 0]) if np.any(local_log_dtoa_std > 0) else 1.0,
        args.pa_dtoa_adapt_strength,
        rel_floor,
    )
    missing_like = dtoa_us > np.maximum(pri_us * args.pa_missing_pulse_ratio, args.min_valid_pri_us)
    rel_dtoa = rel_dtoa.copy()
    rel_dtoa[missing_like] *= float(args.pa_missing_dtoa_weight)
    rel_dtoa = np.clip(rel_dtoa, rel_floor, 1.0)
    report_feature_progress(60, "reliability")

    raw_parts = [
        p1,
        p2,
        p4,
        sin_p5,
        cos_p5,
    ]
    feature_names = [
        "Param1",
        "Param2",
        "Param4",
        "sin_Param5",
        "cos_Param5",
    ]
    weights = [
        args.pa_w_p1 * rel_p1,
        args.pa_w_p2 * rel_p2,
        args.pa_w_p4 * rel_p4,
        args.pa_w_p5 * rel_p5,
        args.pa_w_p5 * rel_p5,
    ]

    if args.pa_profile in ("compact", "full"):
        raw_parts.extend([log_dtoa, local_log_dtoa, period_residual])
        feature_names.extend(["log_dtoa_us", "local_log_dtoa_us", "period_residual"])
        weights.extend(
            [
                args.pa_w_dtoa * rel_dtoa,
                args.pa_w_dtoa_local * rel_dtoa,
                args.pa_w_period * rel_dtoa,
            ]
        )

    if args.pa_profile == "full":
        p3 = df["Param3"].to_numpy(dtype=np.float64)
        p6 = df["Param6"].to_numpy(dtype=np.float64)
        p7 = df["Param7"].to_numpy(dtype=np.float64)
        raw_parts.extend([p3, p6, p7])
        feature_names.extend(["Param3", "Param6", "Param7"])
        weights.extend(
            [
                np.full(len(df), args.pa_w_aux, dtype=np.float32),
                np.full(len(df), args.pa_w_aux, dtype=np.float32),
                np.full(len(df), args.pa_w_aux, dtype=np.float32),
            ]
        )

    raw = np.column_stack(raw_parts)
    scaled, _, scale = _robust_scale(raw)
    report_feature_progress(80, "robust-scale")
    weight_matrix = np.column_stack(weights).astype(np.float32)
    if args.pa_disable_adaptive_weights:
        mean_weights = np.median(weight_matrix, axis=0)
        weight_matrix = np.broadcast_to(mean_weights, weight_matrix.shape).astype(np.float32)
    weight_matrix = np.clip(weight_matrix, args.pa_weight_floor, args.pa_weight_ceiling)

    if args.pa_profile in ("compact", "full"):
        local_noise = (
            (1.0 - rel_p1)
            + (1.0 - rel_p2)
            + (1.0 - rel_p4)
            + (1.0 - rel_p5)
            + (1.0 - rel_dtoa)
        ) / 5.0
    else:
        local_noise = ((1.0 - rel_p1) + (1.0 - rel_p2) + (1.0 - rel_p4) + (1.0 - rel_p5)) / 4.0
    local_scale = 1.0 + args.pa_local_scale_strength * np.maximum(local_noise, 0.0)
    local_scale = np.clip(local_scale, args.pa_local_scale_min, args.pa_local_scale_max).astype(np.float32)

    features = scaled * np.sqrt(weight_matrix) / np.sqrt(local_scale[:, None])
    features = features.astype(np.float32)
    report_feature_progress(100, "done")
    context = {
        "toa": toa,
        "p1": p1,
        "p2": p2,
        "p4": p4,
        "p5": p5,
        "pri_us": pri_us,
        "rel_p1": rel_p1,
        "rel_p2": rel_p2,
        "rel_p4": rel_p4,
        "rel_p5": rel_p5,
        "rel_dtoa": rel_dtoa,
        "local_scale": local_scale,
        "scaled": scaled,
        "robust_scale": scale,
    }
    info = {
        "pa_profile": args.pa_profile,
        "pa_distance_mode": args.pa_distance_mode,
        "pa_feature_dim": int(features.shape[1]),
        "pa_feature_names": feature_names,
        "pa_local_window": int(args.pa_local_window),
        "pa_adapt_strength": float(args.pa_adapt_strength),
        "pa_dtoa_adapt_strength": float(args.pa_dtoa_adapt_strength),
        "pa_local_scale_strength": float(args.pa_local_scale_strength),
        "pa_mean_local_scale": float(np.mean(local_scale)),
        "pa_mean_rel_dtoa": float(np.mean(rel_dtoa)),
        "pa_robust_scale": [float(v) for v in scale],
    }
    return features, context, info


def build_pa_pairwise_distance(context: Dict[str, np.ndarray], idx: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    p1 = context["p1"][idx]
    p2 = context["p2"][idx]
    p4 = context["p4"][idx]
    p5 = context["p5"][idx]
    toa = context["toa"][idx]
    pri = np.maximum(context["pri_us"][idx], args.min_valid_pri_us)

    d_p1 = np.abs(p1[:, None] - p1[None, :]) / max(args.tol_p1, 1e-12)
    d_p2 = np.abs(p2[:, None] - p2[None, :]) / max(args.tol_p2, 1e-12)
    d_p4 = np.abs(p4[:, None] - p4[None, :]) / max(args.tol_p4, 1e-12)
    d_p5 = np.abs((p5[:, None] - p5[None, :] + 180.0) % 360.0 - 180.0) / max(args.tol_p5_deg, 1e-12)

    dt_us = np.abs(toa[:, None] - toa[None, :]) * 1e6
    pri_pair = np.maximum(0.5 * (pri[:, None] + pri[None, :]), args.min_valid_pri_us)
    rem = np.mod(dt_us, pri_pair)
    d_period = np.minimum(rem, pri_pair - rem) / max(args.tol_phase_us, 1e-12)
    d_pri = np.abs(pri[:, None] - pri[None, :]) / max(args.tol_pri_us, 1e-12)

    def pair_rel(name: str) -> np.ndarray:
        rel = context[name][idx].astype(np.float32)
        return 0.5 * (rel[:, None] + rel[None, :])

    dist = (
        args.pa_w_p1 * pair_rel("rel_p1") * d_p1
        + args.pa_w_p2 * pair_rel("rel_p2") * d_p2
        + args.pa_w_p4 * pair_rel("rel_p4") * d_p4
        + args.pa_w_p5 * pair_rel("rel_p5") * d_p5
        + args.pa_w_dtoa * pair_rel("rel_dtoa") * d_pri
        + args.pa_w_period * pair_rel("rel_dtoa") * d_period
    )
    scale = np.sqrt(context["local_scale"][idx][:, None] * context["local_scale"][idx][None, :])
    dist = dist / np.maximum(scale, 1e-6)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float64)


def fit_pa_window_clusterer(
    window_features: np.ndarray,
    context: Dict[str, np.ndarray],
    idx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    mode = args.pa_distance_mode
    if mode == "auto":
        mode = "precomputed" if len(idx) <= args.pa_precomputed_max_points else "embedding"
    cluster_args = copy.copy(args)
    if mode == "precomputed":
        cluster_args.hdbscan_metric = "precomputed"
        labels, info = hdbsort.fit_window_clusterer(build_pa_pairwise_distance(context, idx, args), cluster_args)
    else:
        cluster_args.hdbscan_metric = "euclidean"
        labels, info = hdbsort.fit_window_clusterer(window_features, cluster_args)
    info["pa_distance_mode_used"] = mode
    return labels, info


def create_pa_hdbscan_tracklets(
    df: pd.DataFrame,
    features: np.ndarray,
    context: Dict[str, np.ndarray],
    window_groups: List[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[dbscan.Tracklet], pd.DataFrame]:
    n = len(df)
    tracklet_ids = np.full(n, -1, dtype=np.int64)
    tracklets: List[dbscan.Tracklet] = []
    report_rows = []

    params = df[dbscan.PDW_COLUMNS].to_numpy(dtype=np.float64)
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    t0 = float(np.min(toa))
    start_time = time.perf_counter()
    total_windows = len(window_groups)

    for pos, idx in enumerate(window_groups, start=1):
        window_id = int(np.floor((toa[idx[0]] - t0) / args.window_seconds))
        n_window = int(len(idx))
        if n_window < max(args.min_tracklet_size, args.hdbscan_min_cluster_size):
            report_rows.append(
                {
                    "window_id": window_id,
                    "num_pulses": n_window,
                    "backend": "",
                    "pa_distance_mode_used": "",
                    "num_clusters": 0,
                    "num_tracklets": 0,
                    "noise_pulses": n_window,
                    "small_cluster_pulses": 0,
                    "mean_local_scale": float(np.mean(context["local_scale"][idx])) if len(idx) else np.nan,
                    "mean_rel_dtoa": float(np.mean(context["rel_dtoa"][idx])) if len(idx) else np.nan,
                    "reason": "window_too_small",
                }
            )
            continue
        # 先聚类，去掉太小的簇，记录噪声数量；如果允许，就把靠近有效簇的噪声点救回来，然后重新更新有效簇列表，供后面生成 tracklet。
        labels, backend_info = fit_pa_window_clusterer(features[idx], context, idx, args)
        unique_labels = [int(v) for v in np.unique(labels) if int(v) >= 0]
        counts: Dict[int, int] = {label: int(np.sum(labels == label)) for label in unique_labels}
        valid_clusters = [label for label, count in counts.items() if count >= args.min_tracklet_size]

        small_cluster_pulses = int(sum(count for label, count in counts.items() if label not in valid_clusters)) # 统计小簇数量
        noise_before = int(np.sum(labels < 0))
        if args.assign_noise and valid_clusters and backend_info.get("pa_distance_mode_used") != "precomputed":
            labels = dbscan.assign_noise_to_clusters(labels, features[idx], valid_clusters, args.noise_assign_max_dist) # 噪声点重新分配到簇中
            unique_labels = [int(v) for v in np.unique(labels) if int(v) >= 0]
            counts = {label: int(np.sum(labels == label)) for label in unique_labels}
            valid_clusters = [label for label, count in counts.items() if count >= args.min_tracklet_size]

        made = 0
        for cluster_id in valid_clusters:
            pulse_idx = idx[labels == cluster_id]
            if len(pulse_idx) < args.min_tracklet_size:
                continue
            node_id = len(tracklets)
            tracklet_ids[pulse_idx] = node_id
            tracklets.append(dbscan.make_tracklet(node_id, window_id, pulse_idx, params, toa))
            made += 1

        report_rows.append(
            {
                "window_id": window_id,
                "num_pulses": n_window,
                "backend": backend_info.get("backend", ""),
                "pa_distance_mode_used": backend_info.get("pa_distance_mode_used", ""),
                "num_clusters": int(len(unique_labels)),
                "num_tracklets": int(made),
                "noise_pulses": int(np.sum(labels < 0)),
                "noise_pulses_before_assign": noise_before,
                "small_cluster_pulses": small_cluster_pulses,
                "mean_probability": backend_info.get("mean_probability", np.nan),
                "mean_local_scale": float(np.mean(context["local_scale"][idx])),
                "mean_rel_dtoa": float(np.mean(context["rel_dtoa"][idx])),
                "reason": "",
            }
        )

        should_report = args.progress_every > 0 and (pos % args.progress_every == 0 or pos == total_windows)
        if args.verbose or should_report:
            elapsed = time.perf_counter() - start_time
            windows_per_sec = pos / max(elapsed, 1e-9)
            eta = (total_windows - pos) / max(windows_per_sec, 1e-9)
            print(
                f"[pa-hdbscan] windows {pos}/{total_windows} "
                f"({pos / max(total_windows, 1) * 100:.1f}%), "
                f"tracklets={len(tracklets)}, mode={backend_info.get('pa_distance_mode_used', '')}, "
                f"elapsed={elapsed / 60:.1f}min, eta={eta / 60:.1f}min",
                flush=True,
            )

    return tracklet_ids, tracklets, pd.DataFrame(report_rows)


def _track_centers_with_time(pdw: pd.DataFrame, sigidx: np.ndarray) -> pd.DataFrame:
    work = pdw.copy()
    work["_sigidx"] = sigidx.astype(np.int64)
    work = work[work["_sigidx"] > 0]
    if len(work) == 0:
        return pd.DataFrame()

    grouped = work.groupby("_sigidx", sort=True)
    rows = []
    for sig, group in grouped:
        toa = group["TOA(s)"].to_numpy(dtype=np.float64)
        pri_us, pri_iqr_us = dbscan.estimate_pri_us(toa)
        rows.append(
            {
                "sigidx": int(sig),
                "count": int(len(group)),
                "start_toa": float(np.min(toa)),
                "end_toa": float(np.max(toa)),
                "Param1": float(np.median(group["Param1"].to_numpy(dtype=np.float64))),
                "Param2": float(np.median(group["Param2"].to_numpy(dtype=np.float64))),
                "Param3": float(np.median(group["Param3"].to_numpy(dtype=np.float64))),
                "Param4": float(np.median(group["Param4"].to_numpy(dtype=np.float64))),
                "Param5": dbscan.circular_mean_deg(group["Param5"].to_numpy(dtype=np.float64)),
                "Param6": float(np.median(group["Param6"].to_numpy(dtype=np.float64))),
                "Param7": float(np.median(group["Param7"].to_numpy(dtype=np.float64))),
                "pri_us": float(pri_us),
                "pri_iqr_us": float(pri_iqr_us),
            }
        )
    return pd.DataFrame(rows).set_index("sigidx", drop=True)


# ---------------------------------------------------------------------------
# TSR output refinement
# ---------------------------------------------------------------------------

def tsr_rescue_noise_by_physics(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Attach unassigned pulses to nearby reliable batches with hard gates.

    This stage intentionally leaves all positive SigIdx values unchanged. It
    only tries to recover pulses that HDBSCAN marked as noise/unassigned, using
    sizeable batches as anchors to avoid small-fragment drift.
    """
    out = sigidx.astype(np.int64).copy()
    zero_idx = np.flatnonzero(out <= 0)
    if len(zero_idx) == 0:
        return out, {
            "enabled": True,
            "candidate_noise_pulses": 0,
            "anchor_batches": 0,
            "rescued_pulses": 0,
        }

    centers = _track_centers_with_time(pdw, out)
    if len(centers) == 0:
        return out, {
            "enabled": True,
            "candidate_noise_pulses": int(len(zero_idx)),
            "anchor_batches": 0,
            "rescued_pulses": 0,
        }

    anchor_centers = centers[centers["count"].to_numpy(dtype=np.int64) >= args.tsr_rescue_anchor_min_count]
    if len(anchor_centers) == 0:
        return out, {
            "enabled": True,
            "candidate_noise_pulses": int(len(zero_idx)),
            "anchor_batches": 0,
            "rescued_pulses": 0,
        }

    anchor_ids = anchor_centers.index.to_numpy(dtype=np.int64)
    anchor_arr = anchor_centers[["Param1", "Param2", "Param4", "Param5"]].to_numpy(dtype=np.float64)
    values = pdw[["Param1", "Param2", "Param4", "Param5"]].to_numpy(dtype=np.float64)
    gate = float(args.tsr_rescue_hard_gate)
    thresh = float(args.tsr_rescue_merge_thresh)

    rescued = 0
    chunk_size = max(1, int(args.tsr_rescue_chunk_size))
    anchor_block = max(1, int(args.tsr_rescue_anchor_block))
    for start in range(0, len(zero_idx), chunk_size):
        idx = zero_idx[start : start + chunk_size]
        x = values[idx]
        best_dist = np.full(len(idx), np.inf, dtype=np.float64)
        best_id = np.zeros(len(idx), dtype=np.int64)

        for block_start in range(0, len(anchor_ids), anchor_block):
            anchors = anchor_arr[block_start : block_start + anchor_block]
            d_p1 = np.abs(x[:, 0, None] - anchors[None, :, 0]) / max(args.tsr_rescue_tol_p1, 1e-12)
            d_p2 = np.abs(x[:, 1, None] - anchors[None, :, 1]) / max(args.tsr_rescue_tol_p2, 1e-12)
            d_p4 = np.abs(x[:, 2, None] - anchors[None, :, 2]) / max(args.tsr_rescue_tol_p4, 1e-12)
            d_p5 = (
                np.abs((x[:, 3, None] - anchors[None, :, 3] + 180.0) % 360.0 - 180.0)
                / max(args.tsr_rescue_tol_p5_deg, 1e-12)
            )
            hard_ok = (d_p1 <= gate) & (d_p2 <= gate) & (d_p4 <= gate) & (d_p5 <= gate)
            dist = (
                args.tsr_rescue_w_p1 * d_p1
                + args.tsr_rescue_w_p2 * d_p2
                + args.tsr_rescue_w_p4 * d_p4
                + args.tsr_rescue_w_p5 * d_p5
            )
            dist[~hard_ok] = np.inf
            nearest_pos = np.argmin(dist, axis=1)
            nearest_dist = dist[np.arange(len(idx)), nearest_pos]
            take = nearest_dist < best_dist
            if np.any(take):
                best_dist[take] = nearest_dist[take]
                best_id[take] = anchor_ids[block_start + nearest_pos[take]]

        take = best_dist <= thresh
        if np.any(take):
            out[idx[take]] = best_id[take]
            rescued += int(np.sum(take))

    return out, {
        "enabled": True,
        "candidate_noise_pulses": int(len(zero_idx)),
        "anchor_batches": int(len(anchor_ids)),
        "rescued_pulses": int(rescued),
        "rescue_rate": float(rescued / max(len(zero_idx), 1)),
        "merge_thresh": float(thresh),
        "hard_gate": float(gate),
        "anchor_min_count": int(args.tsr_rescue_anchor_min_count),
    }


def _parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


# ---------------------------------------------------------------------------
# BGMM/BIC parameter-band split
# ---------------------------------------------------------------------------


def _parse_band_split_groups(args: argparse.Namespace) -> List[Tuple[List[int], int, str]]:
    groups_text = str(getattr(args, "tsr_band_split_groups", "") or "").strip()
    groups: List[Tuple[List[int], int, str]] = []
    if groups_text:
        for raw_spec in groups_text.split(";"):
            spec = raw_spec.strip()
            if not spec:
                continue
            parts = [part.strip() for part in spec.split(":")]
            selected_ids = _parse_int_list(parts[0])
            if not selected_ids:
                continue
            components = int(parts[1]) if len(parts) >= 2 and parts[1] else int(args.tsr_band_split_components)
            group_name = parts[2] if len(parts) >= 3 and parts[2] else str(args.tsr_band_split_group)
            groups.append((selected_ids, components, group_name))
        return groups

    selected_ids = _parse_int_list(args.tsr_band_split_sigidx)
    if selected_ids:
        groups.append((selected_ids, int(args.tsr_band_split_components), str(args.tsr_band_split_group)))
    return groups


def _merge_close_band_components(
    means: np.ndarray,
    stds: np.ndarray,
    counts: np.ndarray,
    min_count: int,
    min_fraction: float,
    merge_sep: float,
) -> Tuple[List[float], List[int]]:
    order = np.argsort(means)
    bands: List[Dict[str, float]] = []
    total = max(int(np.sum(counts)), 1)
    for component in order:
        component = int(component)
        count = int(counts[component])
        if count < min_count or count / total < min_fraction:
            continue
        mean = float(means[component])
        std = max(float(stds[component]), 1e-6)
        if bands:
            prev = bands[-1]
            sep = abs(mean - float(prev["mean"])) / max(std + float(prev["std"]), 1e-6)
            if sep < merge_sep:
                merged_count = int(prev["count"]) + count
                prev["mean"] = (float(prev["mean"]) * int(prev["count"]) + mean * count) / merged_count
                prev["std"] = (float(prev["std"]) * int(prev["count"]) + std * count) / merged_count
                prev["count"] = merged_count
                continue
        bands.append({"mean": mean, "std": std, "count": count})
    return [float(item["mean"]) for item in bands], [int(item["count"]) for item in bands]


def _count_matched_bands(left: Sequence[float], right: Sequence[float], tol: float) -> int:
    used = set()
    matches = 0
    for value in left:
        best_pos = None
        best_dist = float("inf")
        for pos, other in enumerate(right):
            if pos in used:
                continue
            dist = abs(float(value) - float(other))
            if dist <= tol and dist < best_dist:
                best_dist = dist
                best_pos = pos
        if best_pos is not None:
            used.add(best_pos)
            matches += 1
    return int(matches)


def _build_chunk_ids_from_toa(toa: np.ndarray, chunk_seconds: float) -> np.ndarray:
    toa = np.asarray(toa, dtype=np.float64)
    if len(toa) == 0:
        return np.zeros(0, dtype=np.int64)
    if chunk_seconds <= 0:
        return np.zeros(len(toa), dtype=np.int64)
    t0 = float(np.min(toa))
    return np.floor((toa - t0) / float(chunk_seconds)).astype(np.int64)


def _compute_group_temporal_overlap_stats(
    beat_sets: Dict[int, set[int]],
    member_ids: Sequence[int],
    strong_overlap: float,
) -> Dict[str, float]:
    valid_ids = [int(sid) for sid in member_ids if int(sid) in beat_sets and len(beat_sets[int(sid)]) > 0]
    if len(valid_ids) <= 1:
        unique_beats = len(beat_sets[valid_ids[0]]) if len(valid_ids) == 1 else 0
        return {
            "avg_pair_min_overlap": 1.0 if len(valid_ids) == 1 else 0.0,
            "avg_pair_jaccard": 1.0 if len(valid_ids) == 1 else 0.0,
            "strong_pair_fraction": 1.0 if len(valid_ids) == 1 else 0.0,
            "multi_source_beat_fraction": 0.0,
            "pair_count": 0.0,
            "unique_beats": float(unique_beats),
        }

    min_overlaps: List[float] = []
    jaccard_overlaps: List[float] = []
    strong_pairs = 0
    beat_hist: Dict[int, int] = {}
    for sid in valid_ids:
        for beat in beat_sets[int(sid)]:
            beat_hist[int(beat)] = beat_hist.get(int(beat), 0) + 1

    for left_pos, left_id in enumerate(valid_ids):
        left_beats = beat_sets[int(left_id)]
        for right_id in valid_ids[left_pos + 1 :]:
            right_beats = beat_sets[int(right_id)]
            inter = len(left_beats & right_beats)
            union = len(left_beats | right_beats)
            min_size = min(len(left_beats), len(right_beats))
            min_overlap = float(inter / max(min_size, 1))
            jaccard = float(inter / max(union, 1))
            min_overlaps.append(min_overlap)
            jaccard_overlaps.append(jaccard)
            if min_overlap >= float(strong_overlap):
                strong_pairs += 1

    pair_count = len(min_overlaps)
    multi_source_beats = sum(1 for count in beat_hist.values() if count >= 2)
    unique_beats = len(beat_hist)
    return {
        "avg_pair_min_overlap": float(np.mean(min_overlaps)) if pair_count else 0.0,
        "avg_pair_jaccard": float(np.mean(jaccard_overlaps)) if pair_count else 0.0,
        "strong_pair_fraction": float(strong_pairs / pair_count) if pair_count else 0.0,
        "multi_source_beat_fraction": float(multi_source_beats / max(unique_beats, 1)),
        "pair_count": float(pair_count),
        "unique_beats": float(unique_beats),
    }


def _cluster_group_band_count(mean_count_pairs: Sequence[Tuple[float, int]], tol: float, min_fraction: float) -> int:
    if not mean_count_pairs:
        return 0
    ordered = sorted((float(mean), int(count)) for mean, count in mean_count_pairs)
    clusters = [{"mean": ordered[0][0], "count": ordered[0][1]}]
    for value, count in ordered[1:]:
        prev = clusters[-1]
        if abs(value - float(prev["mean"])) > tol:
            clusters.append({"mean": value, "count": count})
        else:
            merged_count = int(prev["count"]) + count
            prev["mean"] = (float(prev["mean"]) * int(prev["count"]) + value * count) / max(merged_count, 1)
            prev["count"] = merged_count
    total = max(sum(int(item["count"]) for item in clusters), 1)
    clusters = [item for item in clusters if int(item["count"]) / total >= min_fraction]
    return int(len(clusters))


def _choose_auto_group_mode(
    components: int,
    dominant_fraction: float,
    weighted_component_dominance: float,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    """Choose whether an automatically discovered group should split `all` or only the `middle` bands.

    Empirically, some high-value mixed groups are composed of source-pure components:
    the previous logic rejected them outright, which blocked beneficial "split all"
    cases on Sample_2. We now reuse the same dominance statistic to decide *how* to
    split the group instead of whether the group is allowed to split at all.
    """
    if int(components) < 4:
        return "all", "fewer_than_four_components"
    if float(weighted_component_dominance) >= float(args.tsr_auto_max_component_dominance):
        return "all", "high_component_dominance_prefers_all"
    if float(dominant_fraction) < float(args.tsr_auto_middle_dominance):
        return "middle", "low_dominant_fraction_prefers_middle"
    return "all", "default_all"


def learn_tsr_band_split_groups(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[List[Tuple[List[int], int, str]], Dict[str, object], pd.DataFrame]:
    """Learn GMM band-split groups without truth labels or hand-picked SigIdx values.

    BIC-GMM first detects whether a batch is multi-modal. Bayesian GMM then
    confirms the effective bands, so the final split is selected by an
    unsupervised model instead of manually naming mixed SigIdx values.
    """
    if not getattr(args, "tsr_band_split_auto", False):
        return [], {"enabled": False}, pd.DataFrame()
    if args.tsr_band_split_feature not in pdw.columns:
        raise ValueError(f"Unknown band split feature: {args.tsr_band_split_feature}")

    try:
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
    except ImportError as exc:
        raise RuntimeError("Automatic band split requires scikit-learn.") from exc

    values = pdw[args.tsr_band_split_feature].to_numpy(dtype=np.float64)
    sigidx = sigidx.astype(np.int64, copy=False)
    ids, counts = np.unique(sigidx[sigidx > 0], return_counts=True)
    candidates = [
        (int(sid), int(count))
        for sid, count in zip(ids, counts)
        if int(count) >= args.tsr_auto_min_batch_count and int(count) <= args.tsr_auto_max_batch_count
    ]
    rng = np.random.default_rng(int(args.seed))
    rows = []

    def compute_group_source_mixing(member_ids: Sequence[int], components: int) -> Dict[str, float]:
        idx = np.flatnonzero(np.isin(sigidx, np.asarray(member_ids, dtype=np.int64)))
        if len(idx) == 0:
            return {"weighted_component_dominance": 1.0, "component_count": 0.0}
        if len(idx) > args.tsr_auto_sample_size:
            fit_idx = rng.choice(idx, size=int(args.tsr_auto_sample_size), replace=False)
        else:
            fit_idx = idx
        z = values[fit_idx].reshape(-1, 1)
        src = sigidx[fit_idx]
        if len(z) < max(args.tsr_band_split_min_count, components * 10):
            return {"weighted_component_dominance": 1.0, "component_count": 0.0}

        model = GaussianMixture(
            n_components=int(components),
            covariance_type="full",
            random_state=args.seed,
            reg_covar=args.tsr_band_split_reg_covar,
            n_init=args.tsr_band_split_n_init,
            max_iter=args.tsr_band_split_max_iter,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak.*")
            labels = model.fit_predict(z)

        weighted_dominance = 0.0
        total_weight = 0.0
        component_count = 0
        for component in range(int(components)):
            comp_mask = labels == int(component)
            comp_total = int(np.sum(comp_mask))
            if comp_total <= 0:
                continue
            component_count += 1
            max_fraction = 0.0
            for sid in member_ids:
                sid_count = int(np.sum(comp_mask & (src == int(sid))))
                if sid_count <= 0:
                    continue
                max_fraction = max(max_fraction, sid_count / max(comp_total, 1))
            weighted_dominance += float(max_fraction) * float(comp_total)
            total_weight += float(comp_total)

        return {
            "weighted_component_dominance": float(weighted_dominance / max(total_weight, 1.0)),
            "component_count": float(component_count),
        }

    for sid, total_count in candidates:
        idx = np.flatnonzero(sigidx == sid)
        if len(idx) > args.tsr_auto_sample_size:
            fit_idx = rng.choice(idx, size=int(args.tsr_auto_sample_size), replace=False)
        else:
            fit_idx = idx
        z = values[fit_idx].reshape(-1, 1)
        if len(z) < max(args.tsr_band_split_min_count, args.tsr_auto_min_components * 10):
            continue

        models = []
        for components in range(1, int(args.tsr_auto_max_components) + 1):
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                random_state=args.seed,
                reg_covar=args.tsr_band_split_reg_covar,
                n_init=args.tsr_band_split_n_init,
                max_iter=args.tsr_band_split_max_iter,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak.*")
                labels = model.fit_predict(z)
            means = model.means_.reshape(-1)
            stds = np.sqrt(model.covariances_.reshape(-1))
            component_counts = np.array([int(np.sum(labels == component)) for component in range(components)])
            models.append((components, float(model.bic(z)), means, stds, component_counts))
        base_bic = models[0][1]
        best = min(models, key=lambda item: item[1])
        best_k, best_bic, means, stds, component_counts = best
        bic_gain = float((base_bic - best_bic) / max(len(z), 1))
        sample_min_count = max(10, int(np.ceil(args.tsr_band_split_min_child_count * len(fit_idx) / max(total_count, 1))))
        bic_band_means, bic_band_counts = _merge_close_band_components(
            means=means,
            stds=stds,
            counts=component_counts,
            min_count=sample_min_count,
            min_fraction=args.tsr_auto_min_component_fraction,
            merge_sep=args.tsr_auto_merge_sep,
        )
        bic_pass = best_k >= 2 and bic_gain >= args.tsr_auto_bic_gain and len(bic_band_means) >= args.tsr_auto_min_components
        if not bic_pass:
            continue

        bgmm_band_means: List[float] = []
        bgmm_band_counts: List[int] = []
        bgmm_weights: List[float] = []
        bgmm_confirmed = False
        bgmm_components = 0
        bgmm_effective_bands = 0
        bgmm_model = BayesianGaussianMixture(
            n_components=int(args.tsr_auto_bgmm_max_components),
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=float(args.tsr_auto_bgmm_weight_concentration),
            random_state=args.seed,
            reg_covar=args.tsr_band_split_reg_covar,
            n_init=args.tsr_band_split_n_init,
            max_iter=args.tsr_auto_bgmm_max_iter,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak.*")
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            bgmm_labels = bgmm_model.fit_predict(z)
        bgmm_means = bgmm_model.means_.reshape(-1)
        bgmm_stds = np.sqrt(bgmm_model.covariances_.reshape(-1))
        bgmm_component_counts = np.array(
            [int(np.sum(bgmm_labels == component)) for component in range(int(args.tsr_auto_bgmm_max_components))]
        )
        bgmm_weights_arr = bgmm_model.weights_.reshape(-1)
        bgmm_component_counts = bgmm_component_counts.copy()
        bgmm_component_counts[bgmm_weights_arr < args.tsr_auto_bgmm_weight_threshold] = 0
        bgmm_components = int(np.sum(bgmm_component_counts > 0))
        bgmm_band_means, bgmm_band_counts = _merge_close_band_components(
            means=bgmm_means,
            stds=bgmm_stds,
            counts=bgmm_component_counts,
            min_count=sample_min_count,
            min_fraction=args.tsr_auto_min_component_fraction,
            merge_sep=args.tsr_auto_merge_sep,
        )
        bgmm_effective_bands = int(len(bgmm_band_means))
        bgmm_weights = [float(value) for value in bgmm_weights_arr]

        selected_means = bic_band_means
        selected_counts = bic_band_counts
        selection_model = "bic_gmm_fallback"
        if bgmm_effective_bands >= args.tsr_auto_min_components:
            required = min(2, len(bic_band_means), len(bgmm_band_means))
            bgmm_confirmed = _count_matched_bands(
                bic_band_means,
                bgmm_band_means,
                args.tsr_auto_group_tol,
            ) >= required
            if bgmm_confirmed:
                selected_means = bgmm_band_means
                selected_counts = bgmm_band_counts
                selection_model = "bayesian_gmm_confirmed"

        dominant_fraction = max(selected_counts) / max(sum(selected_counts), 1)
        rows.append(
            {
                "sigidx": int(sid),
                "batch_pulses": int(total_count),
                "best_components": int(best_k),
                "effective_bands": int(len(selected_means)),
                "bic_gain_per_pulse": float(bic_gain),
                "selection_model": selection_model,
                "bic_effective_bands": int(len(bic_band_means)),
                "bgmm_components": int(bgmm_components),
                "bgmm_effective_bands": int(bgmm_effective_bands),
                "bgmm_confirmed": bool(bgmm_confirmed),
                "dominant_fraction": float(dominant_fraction),
                "band_means": ",".join(f"{value:.6g}" for value in selected_means),
                "band_counts_sample": ",".join(str(int(value)) for value in selected_counts),
                "bic_band_means": ",".join(f"{value:.6g}" for value in bic_band_means),
                "bgmm_band_means": ",".join(f"{value:.6g}" for value in bgmm_band_means),
                "bgmm_weights": ",".join(f"{value:.6g}" for value in bgmm_weights),
            }
        )

    report = pd.DataFrame(rows)
    if len(report) == 0:
        return [], {"enabled": True, "learned_groups": 0, "reason": "no_multimodal_candidates"}, report

    candidate_bands = {
        int(row.sigidx): [float(value) for value in str(row.band_means).split(",") if value]
        for row in report.itertuples(index=False)
    }
    chunk_seconds = float(getattr(args, "tsr_auto_chunk_seconds", getattr(args, "sort_chunk_seconds", 0.2)))
    beat_ids = _build_chunk_ids_from_toa(pdw["TOA(s)"].to_numpy(dtype=np.float64), chunk_seconds)
    beat_sets = {
        int(sid): set(int(v) for v in beat_ids[sigidx == int(sid)].tolist())
        for sid in candidate_bands
    }
    candidate_ids = list(candidate_bands)
    parent = {sid: sid for sid in candidate_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i, left_id in enumerate(candidate_ids):
        for right_id in candidate_ids[i + 1 :]:
            required = min(2, len(candidate_bands[left_id]), len(candidate_bands[right_id]))
            if _count_matched_bands(candidate_bands[left_id], candidate_bands[right_id], args.tsr_auto_group_tol) >= required:
                union(left_id, right_id)

    grouped: Dict[int, List[int]] = {}
    for sid in candidate_ids:
        grouped.setdefault(find(sid), []).append(sid)

    report_by_id = report.set_index("sigidx").to_dict(orient="index")
    learned = []
    group_summaries = []
    rejected_temporal_gate = 0
    for member_ids in grouped.values():
        member_ids = sorted(int(v) for v in member_ids)
        if len(member_ids) > args.tsr_auto_max_source_ids:
            continue
        all_band_pairs: List[Tuple[float, int]] = []
        total_band_count = 0
        max_band_count = 0
        total_pulses = 0
        max_bic_gain = 0.0
        score = 0.0
        for sid in member_ids:
            info = report_by_id[sid]
            means_i = candidate_bands[sid]
            counts_i = [int(value) for value in str(info["band_counts_sample"]).split(",") if value]
            all_band_pairs.extend((float(mean), int(count)) for mean, count in zip(means_i, counts_i))
            total_band_count += int(sum(counts_i))
            max_band_count = max(max_band_count, max(counts_i) if counts_i else 0)
            total_pulses += int(info["batch_pulses"])
            bic_gain = float(info["bic_gain_per_pulse"])
            max_bic_gain = max(max_bic_gain, bic_gain)
            score += bic_gain * np.log1p(float(info["batch_pulses"]))
        if total_pulses < args.tsr_auto_min_group_pulses and max_bic_gain < args.tsr_auto_small_group_bic_gain:
            continue
        if total_pulses < args.tsr_auto_medium_group_pulses and max_bic_gain < args.tsr_auto_medium_group_bic_gain:
            continue
        components = _cluster_group_band_count(
            all_band_pairs,
            args.tsr_auto_group_tol,
            args.tsr_auto_min_global_band_fraction,
        )
        if components < args.tsr_auto_min_components:
            continue
        temporal_stats = _compute_group_temporal_overlap_stats(
            beat_sets=beat_sets,
            member_ids=member_ids,
            strong_overlap=float(args.tsr_auto_strong_pair_overlap),
        )
        if len(member_ids) >= 2 and temporal_stats["avg_pair_min_overlap"] < args.tsr_auto_min_pair_min_overlap:
            rejected_temporal_gate += 1
            continue
        if len(member_ids) >= 3 and temporal_stats["strong_pair_fraction"] < args.tsr_auto_min_strong_pair_fraction:
            rejected_temporal_gate += 1
            continue
        source_mixing = {
            "weighted_component_dominance": 1.0,
            "component_count": float(components),
        }
        if len(member_ids) >= 2:
            source_mixing = compute_group_source_mixing(
                member_ids=member_ids,
                components=int(min(components, args.tsr_auto_max_components)),
            )
        dominant_fraction = float(max_band_count / max(total_band_count, 1))
        group_name, group_mode_reason = _choose_auto_group_mode(
            components=int(components),
            dominant_fraction=dominant_fraction,
            weighted_component_dominance=float(source_mixing["weighted_component_dominance"]),
            args=args,
        )
        learned.append((member_ids, int(min(components, args.tsr_auto_max_components)), group_name, float(score), total_pulses))
        group_summaries.append(
            {
                "selected_sigidx": member_ids,
                "components": int(min(components, args.tsr_auto_max_components)),
                "group": group_name,
                "group_mode_reason": str(group_mode_reason),
                "dominant_fraction": dominant_fraction,
                "total_pulses": int(total_pulses),
                "score": float(score),
                "avg_pair_min_overlap": float(temporal_stats["avg_pair_min_overlap"]),
                "strong_pair_fraction": float(temporal_stats["strong_pair_fraction"]),
                "multi_source_beat_fraction": float(temporal_stats["multi_source_beat_fraction"]),
                "group_unique_beats": int(temporal_stats["unique_beats"]),
                "weighted_component_dominance": float(source_mixing["weighted_component_dominance"]),
            }
        )

    learned.sort(key=lambda item: item[3], reverse=True)
    if args.tsr_auto_max_groups > 0:
        learned = learned[: int(args.tsr_auto_max_groups)]
    groups = [(ids, components, group_name) for ids, components, group_name, _, _ in learned]
    summary = {
        "enabled": True,
        "learned_groups": int(len(groups)),
        "candidate_batches": int(len(report)),
        "groups": group_summaries[: int(args.tsr_auto_max_groups)] if args.tsr_auto_max_groups > 0 else group_summaries,
        "feature": str(args.tsr_band_split_feature),
        "auto_model": "bayesian_gmm_confirmed_with_bic_fallback",
        "bic_gain_threshold": float(args.tsr_auto_bic_gain),
        "temporal_gate": {
            "enabled": True,
            "chunk_seconds": float(chunk_seconds),
            "min_pair_min_overlap": float(args.tsr_auto_min_pair_min_overlap),
            "strong_pair_overlap": float(args.tsr_auto_strong_pair_overlap),
            "min_strong_pair_fraction": float(args.tsr_auto_min_strong_pair_fraction),
            "rejected_groups": int(rejected_temporal_gate),
        },
        "source_purity_gate": {
            "enabled": False,
            "max_component_dominance": float(args.tsr_auto_max_component_dominance),
            "rejected_groups": 0,
            "note": "weighted_component_dominance now selects all-vs-middle mode instead of rejecting candidate groups",
        },
    }
    return groups, summary, report


def _tsr_param_band_split_once(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    args: argparse.Namespace,
    selected_ids: List[int],
    components: int,
    group_name: str,
    group_index: int = 1,
) -> Tuple[np.ndarray, Dict[str, object], pd.DataFrame]:
    """Split selected mixed batches by a 1-D Gaussian mixture band model."""
    if not selected_ids:
        return sigidx.copy(), {"enabled": False}, pd.DataFrame()
    if args.tsr_band_split_feature not in pdw.columns:
        raise ValueError(f"Unknown band split feature: {args.tsr_band_split_feature}")
    if group_name not in {"middle", "lower", "upper", "all"}:
        raise ValueError(f"Unsupported band split group: {group_name}")

    try:
        from sklearn.mixture import GaussianMixture
    except ImportError as exc:
        raise RuntimeError("tsr_param_band_split requires scikit-learn.") from exc

    out = sigidx.astype(np.int64).copy()
    idx = np.flatnonzero(np.isin(out, selected_ids))
    if len(idx) < max(args.tsr_band_split_min_count, components * 10):
        return out, {
            "enabled": True,
            "group_index": int(group_index),
            "selected_sigidx": selected_ids,
            "split": False,
            "reason": "too_few_selected_pulses",
            "selected_pulses": int(len(idx)),
        }, pd.DataFrame()

    z = pdw[args.tsr_band_split_feature].to_numpy(dtype=np.float64)[idx].reshape(-1, 1)
    model = GaussianMixture(
        n_components=components,
        covariance_type="full",
        random_state=args.seed,
        reg_covar=args.tsr_band_split_reg_covar,
        n_init=args.tsr_band_split_n_init,
        max_iter=args.tsr_band_split_max_iter,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak.*")
        labels = model.fit_predict(z)
    means = model.means_.reshape(-1)
    order = np.argsort(means)
    ordered_means = means[order]
    ordered_stds = np.sqrt(model.covariances_.reshape(-1))[order]

    if group_name == "all":
        component_counts = np.array([int(np.sum(labels == int(component))) for component in range(components)])
        eligible_components = [
            int(component)
            for component in order
            if int(component_counts[int(component)]) >= args.tsr_band_split_min_child_count
        ]
        if len(eligible_components) < 2:
            return out, {
                "enabled": True,
                "group_index": int(group_index),
                "selected_sigidx": selected_ids,
                "split": False,
                "reason": "too_few_eligible_components",
                "feature": str(args.tsr_band_split_feature),
                "components": int(components),
                "group": str(group_name),
                "selected_pulses": int(len(idx)),
                "component_means": [float(v) for v in ordered_means],
                "component_stds": [float(v) for v in ordered_stds],
                "component_counts": [int(component_counts[int(v)]) for v in order],
            }, pd.DataFrame()

        next_id = int(np.max(out)) + 1 if np.any(out > 0) else 1
        largest_component = max(eligible_components, key=lambda component: int(component_counts[component]))
        component_to_sigidx: Dict[int, int] = {}
        for component in eligible_components:
            if args.tsr_band_split_keep_largest and component == largest_component:
                component_to_sigidx[int(component)] = int(selected_ids[0])
            else:
                component_to_sigidx[int(component)] = int(next_id)
                next_id += 1

        selected_sigidx = out[idx].copy()
        rows = []
        assigned_pulses = 0
        for component in eligible_components:
            comp_mask = labels == int(component)
            new_sigidx = int(component_to_sigidx[int(component)])
            out[idx[comp_mask]] = new_sigidx
            assigned_pulses += int(np.sum(comp_mask))
            for source_id in selected_ids:
                source_count = int(np.sum(comp_mask & (selected_sigidx == int(source_id))))
                if source_count <= 0:
                    continue
                rows.append(
                    {
                        "group_index": int(group_index),
                        "source_sigidx": int(source_id),
                        "component": int(component),
                        "new_sigidx": int(new_sigidx),
                        "component_mean": float(means[int(component)]),
                        "component_std": float(np.sqrt(model.covariances_.reshape(-1)[int(component)])),
                        "component_pulses": int(source_count),
                    }
                )

        summary = {
            "enabled": True,
            "group_index": int(group_index),
            "selected_sigidx": selected_ids,
            "split": True,
            "feature": str(args.tsr_band_split_feature),
            "components": int(components),
            "group": str(group_name),
            "selected_pulses": int(len(idx)),
            "assigned_pulses": int(assigned_pulses),
            "split_components": int(len(eligible_components)),
            "keep_largest": bool(args.tsr_band_split_keep_largest),
            "component_means": [float(v) for v in ordered_means],
            "component_stds": [float(v) for v in ordered_stds],
            "component_counts": [int(component_counts[int(v)]) for v in order],
        }
        return out.astype(np.int64), summary, pd.DataFrame(rows)

    if group_name == "middle":
        if len(order) < 3:
            selected_components = set(int(v) for v in order)
        else:
            selected_components = set(int(v) for v in order[1:-1])
    elif group_name == "lower":
        selected_components = {int(order[0])}
    elif group_name == "upper":
        selected_components = {int(order[-1])}
    else:
        raise ValueError(f"Unsupported --tsr_band_split_group: {group_name}")

    take = np.array([int(label) in selected_components for label in labels], dtype=bool)
    if int(np.sum(take)) < args.tsr_band_split_min_child_count or int(np.sum(~take)) < args.tsr_band_split_min_child_count:
        return out, {
            "enabled": True,
            "group_index": int(group_index),
            "selected_sigidx": selected_ids,
            "split": False,
            "reason": "child_too_small",
            "selected_pulses": int(len(idx)),
            "child_pulses": int(np.sum(take)),
            "rest_pulses": int(np.sum(~take)),
            "component_means": [float(v) for v in ordered_means],
            "component_stds": [float(v) for v in ordered_stds],
        }, pd.DataFrame()

    next_id = int(np.max(out)) + 1 if np.any(out > 0) else 1
    selected_sigidx = out[idx].copy()
    source_infos = []
    for source_id in selected_ids:
        source_mask = selected_sigidx == source_id
        child_mask = source_mask & take
        child_count = int(np.sum(child_mask))
        rest_count = int(np.sum(source_mask & ~take))
        if child_count < args.tsr_band_split_min_child_count:
            continue
        source_infos.append((int(source_id), source_mask, child_mask, child_count, rest_count))

    if not source_infos:
        return out, {
            "enabled": True,
            "group_index": int(group_index),
            "selected_sigidx": selected_ids,
            "split": False,
            "reason": "no_source_child_large_enough",
            "selected_pulses": int(len(idx)),
            "child_pulses": int(np.sum(take)),
            "rest_pulses": int(np.sum(~take)),
            "component_means": [float(v) for v in ordered_means],
            "component_stds": [float(v) for v in ordered_stds],
        }, pd.DataFrame()

    rows = []
    if args.tsr_band_split_shared_child:
        child_id = next_id
        child_union = np.zeros(len(idx), dtype=bool)
        rest_union = np.zeros(len(idx), dtype=bool)
        rest_id = int(source_infos[0][0])
        for source_id, source_mask, child_mask, child_count, rest_count in source_infos:
            child_union |= child_mask
            rest_union |= source_mask & ~take
            rows.append(
                {
                    "source_sigidx": int(source_id),
                    "group_index": int(group_index),
                    "new_sigidx": int(child_id),
                    "rest_sigidx": int(rest_id) if args.tsr_band_split_merge_rest else int(source_id),
                    "child_pulses": int(child_count),
                    "remaining_source_pulses": int(rest_count),
                }
            )
        if args.tsr_band_split_merge_rest:
            out[idx[rest_union]] = rest_id
        out[idx[child_union]] = child_id
        split_count = len(source_infos)
    else:
        rest_id = int(source_infos[0][0])
        for source_id, source_mask, child_mask, child_count, rest_count in source_infos:
            if args.tsr_band_split_merge_rest:
                out[idx[source_mask & ~take]] = rest_id
            out[idx[child_mask]] = next_id
            rows.append(
                {
                    "source_sigidx": int(source_id),
                    "group_index": int(group_index),
                    "new_sigidx": int(next_id),
                    "rest_sigidx": int(rest_id) if args.tsr_band_split_merge_rest else int(source_id),
                    "child_pulses": int(child_count),
                    "remaining_source_pulses": int(rest_count),
                }
            )
            next_id += 1
        split_count = len(source_infos)

    report = pd.DataFrame(rows)
    summary = {
        "enabled": True,
        "group_index": int(group_index),
        "selected_sigidx": selected_ids,
        "split": bool(split_count > 0),
        "feature": str(args.tsr_band_split_feature),
        "components": int(components),
        "group": str(group_name),
        "selected_pulses": int(len(idx)),
        "child_pulses": int(np.sum(take)),
        "rest_pulses": int(np.sum(~take)),
        "assigned_child_pulses": int(sum(info[3] for info in source_infos)),
        "split_source_batches": int(split_count),
        "shared_child": bool(args.tsr_band_split_shared_child),
        "merge_rest": bool(args.tsr_band_split_merge_rest),
        "component_means": [float(v) for v in ordered_means],
        "component_stds": [float(v) for v in ordered_stds],
    }
    return out.astype(np.int64), summary, report


def tsr_param_band_split(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object], pd.DataFrame]:
    groups = _parse_band_split_groups(args)
    auto_summary: Dict[str, object] = {"enabled": False}
    auto_report = pd.DataFrame()
    if not groups:
        groups, auto_summary, auto_report = learn_tsr_band_split_groups(pdw, sigidx, args)
    if not groups:
        return sigidx.copy(), {"enabled": False, "auto_learning": auto_summary}, auto_report

    out = sigidx.astype(np.int64).copy()
    summaries = []
    reports = []
    for group_index, (selected_ids, components, group_name) in enumerate(groups, start=1):
        out, summary, report = _tsr_param_band_split_once(
            pdw=pdw,
            sigidx=out,
            args=args,
            selected_ids=selected_ids,
            components=components,
            group_name=group_name,
            group_index=group_index,
        )
        summaries.append(summary)
        if len(report) > 0:
            reports.append(report)

    merged_report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
    if len(auto_report) > 0:
        auto_report = auto_report.assign(report_type="auto_candidate")
        if len(merged_report) > 0:
            merged_report = pd.concat([auto_report, merged_report.assign(report_type="split_assignment")], ignore_index=True, sort=False)
        else:
            merged_report = auto_report
    if len(summaries) == 1:
        single = summaries[0]
        single["auto_learning"] = auto_summary
        return out.astype(np.int64), single, merged_report

    summary = {
        "enabled": True,
        "split": bool(any(item.get("split", False) for item in summaries)),
        "num_groups": int(len(summaries)),
        "groups": summaries,
        "auto_learning": auto_summary,
    }
    return out.astype(np.int64), summary, merged_report


def apply_tsr_refinement(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Apply the production TSR refinement path.

    The retained path is deliberately compact:
    1. merge obvious fragments by physical consistency;
    2. attach only noise/unassigned pulses to reliable anchors;
    3. split mixed parameter bands with manual rules or automatic BIC/BGMM.
    """
    out = sigidx.astype(np.int64).copy()
    positive_before = out[out > 0]
    summary: Dict[str, object] = {
        "mode": "tsr",
        "positive_sigidx_before": int(len(np.unique(positive_before))) if len(positive_before) else 0,
        "noise_or_zero_before": int(np.sum(out <= 0)),
    }
    split_report = pd.DataFrame()
    merge_report = pd.DataFrame()

    hard_gates = (
        args.tsr_reduce_hard_gate,
        args.tsr_reduce_hard_gate,
        args.tsr_reduce_hard_gate,
        args.tsr_reduce_hard_gate,
    )
    out, reduce_summary, reduce_split_report = reducer.reduce_sigidx_fixed(
        pdw=pdw,
        sigidx=out,
        min_cluster_size=args.tsr_reduce_min_cluster_size,
        merge_thresh=args.tsr_reduce_merge_thresh,
        min_batch_fraction=args.tsr_reduce_min_batch_fraction,
        weights=(args.tsr_reduce_w_p1, args.tsr_reduce_w_p2, args.tsr_reduce_w_p4, args.tsr_reduce_w_p5),
        tolerances=(args.tsr_reduce_tol_p1, args.tsr_reduce_tol_p2, args.tsr_reduce_tol_p4, args.tsr_reduce_tol_p5_deg),
        hard_gates=hard_gates,
        dense_relabel_output=False,
        split_large_batches=False,
    )
    summary["tsr_post_reduce"] = reduce_summary
    if len(reduce_split_report) > 0:
        split_report = pd.concat([split_report, reduce_split_report.assign(stage="post_reduce")], ignore_index=True)

    out, rescue_summary = tsr_rescue_noise_by_physics(pdw, out, args)
    summary["tsr_rescue_noise"] = rescue_summary

    band_split_requested = bool(_parse_band_split_groups(args))
    if band_split_requested:
        out = dbscan.dense_relabel(out)
        summary["tsr_pre_band_dense_relabel"] = True
    else:
        summary["tsr_pre_band_dense_relabel"] = False

    out, band_summary, band_report = tsr_param_band_split(pdw, out, args)
    summary["tsr_param_band_split"] = band_summary
    if len(band_report) > 0:
        merge_report = pd.concat([merge_report, band_report.assign(stage="param_band_split")], ignore_index=True)

    out = dbscan.dense_relabel(out)
    positive_after = out[out > 0]
    summary["positive_sigidx_after"] = int(len(np.unique(positive_after))) if len(positive_after) else 0
    summary["noise_or_zero_after"] = int(np.sum(out <= 0))
    return out.astype(np.int64), summary, split_report, merge_report


def compute_and_save_sort_metrics(
    truth_file: Path,
    pred_sigidx: np.ndarray,
    metrics_dir: Path,
    args: argparse.Namespace,
    prefix: str = "pa_tsr_hdbscan",
) -> Dict[str, object]:
    from 识别.sort_metrics import compute_sort_metrics_by_beat

    truth = dbscan.read_truth_file(truth_file)
    if args.max_pulses > 0:
        truth = truth.iloc[: args.max_pulses].reset_index(drop=True)
    if len(truth) != len(pred_sigidx):
        raise ValueError(f"Truth rows ({len(truth)}) and predicted SigIdx rows ({len(pred_sigidx)}) do not match.")

    empty_batch_df = pd.DataFrame(columns=["pred_sigidx", "batch_pred_label", "batch_max_prob", "is_batch_confident"])
    sort_batch_df, sort_target_df, sort_target_beat_df, sort_beat_df, metrics = compute_sort_metrics_by_beat(
        toa=truth["TOA(s)"].to_numpy(dtype=np.float64),
        true_sigidx=truth["SigIdx"].to_numpy(dtype=np.int64),
        pred_sigidx=pred_sigidx.astype(np.int64),
        true_labels=truth["LABEL"].to_numpy(dtype=np.int64),
        purity_threshold=args.sort_purity_threshold,
        min_target_fraction=args.sort_min_target_fraction,
        mix_fail_min_pulses=args.sort_mix_fail_min_pulses,
        batch_df=empty_batch_df,
        chunk_seconds=args.sort_chunk_seconds,
    )
    metrics["recognition_acc_on_success"] = float("nan")
    metrics["recognition_note"] = f"N/A: {prefix} does not predict LABEL."

    fail_reason = sort_batch_df["fail_reason"].fillna("").astype(str) if len(sort_batch_df) else pd.Series(dtype=str)
    total_rows = int(len(fail_reason))
    failed_rows = int((fail_reason != "").sum()) if total_rows else 0
    metrics["fail_reason_counts"] = {
        "low_purity": int((fail_reason == "low_purity").sum()) if total_rows else 0,
        "low_target_fraction": int((fail_reason == "low_target_fraction").sum()) if total_rows else 0,
        "swallowed_other_target": int((fail_reason == "swallowed_other_target").sum()) if total_rows else 0,
    }
    metrics["fail_reason_rates_all"] = {
        key: float(value / total_rows) if total_rows else float("nan")
        for key, value in metrics["fail_reason_counts"].items()
    }
    metrics["fail_reason_rates_failed"] = {
        key: float(value / failed_rows) if failed_rows else float("nan")
        for key, value in metrics["fail_reason_counts"].items()
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    sort_batch_df.to_csv(metrics_dir / f"{prefix}_sort_batch_eval.csv", index=False, encoding="utf-8-sig")
    sort_target_df.to_csv(metrics_dir / f"{prefix}_sort_target_accuracy.csv", index=False, encoding="utf-8-sig")
    sort_target_beat_df.to_csv(metrics_dir / f"{prefix}_sort_target_beat_eval.csv", index=False, encoding="utf-8-sig")
    sort_beat_df.to_csv(metrics_dir / f"{prefix}_sort_beat_eval.csv", index=False, encoding="utf-8-sig")
    (metrics_dir / f"{prefix}_sort_metrics.json").write_text(
        json.dumps(_safe_json(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return _safe_json(metrics)


def _sigidx_summary(sigidx: np.ndarray) -> Dict[str, int]:
    positive = sigidx[sigidx > 0]
    return {
        "positive_sigidx_count": int(len(np.unique(positive))) if len(positive) else 0,
        "noise_or_unassigned_count": int(np.sum(sigidx <= 0)),
    }


def run_pa_tsr_sort(args: argparse.Namespace) -> Dict[str, object]:
    """Run the complete sorter and save predictions, diagnostics, and metrics."""
    df = dbscan.read_pdw(args.input_file)
    if args.max_pulses > 0:
        df = df.iloc[: args.max_pulses].reset_index(drop=True)
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    backend = hdbsort.resolve_backend(args)

    print(f"Input PDW: {args.input_file}")
    print(f"Pulses:    {len(df)}")
    print(f"HDBSCAN backend: {backend}")
    print(f"PA distance: {args.pa_distance_mode}")

    # Stage 1: encode PDW pulses into a physics-aware adaptive feature space.
    pa_features, pa_context, feature_info = build_pa_features(df, args)
    _, window_groups = dbscan.make_window_groups(toa, args.window_seconds)
    print(f"Windows:   {len(window_groups)} ({args.window_seconds:.3f}s)")
    print(f"Features:  {pa_features.shape[1]} PA dims ({args.pa_profile})")

    # Stage 2: cluster each short time window, then connect compatible tracklets.
    tracklet_ids, tracklets, window_report = create_pa_hdbscan_tracklets(df, pa_features, pa_context, window_groups, args)
    print(f"[pa-hdbscan] merging tracklets: {len(tracklets)} via {args.pa_initial_merge}")
    # 窗口间批次进行合并
    if args.pa_initial_merge == "conservative":
        roots, edge_report = conservative.merge_tracklets_conservative(tracklets, args)
    else:
        roots, edge_report = dbscan.merge_tracklets(tracklets, args)
    # 把标签重新整理成连续编号。
    raw_sigidx = dbscan.dense_relabel(dbscan.roots_to_sigidx(tracklet_ids, roots))

    _ensure_parent(args.raw_output_file)
    dbscan.write_sort_file(df, raw_sigidx, args.raw_output_file, emit_label99=args.emit_label99)

    # Stage 3: refine HDBSCAN output using physical consistency and BIC/BGMM band splitting.
    final_sigidx, tsr_summary, split_report, merge_report = apply_tsr_refinement(df, raw_sigidx, args)
    _ensure_parent(args.output_file)
    dbscan.write_sort_file(df, final_sigidx, args.output_file, emit_label99=args.emit_label99)

    _ensure_parent(args.window_report_csv)
    _ensure_parent(args.edge_report_csv)
    window_report.to_csv(args.window_report_csv, index=False, encoding="utf-8-sig")
    edge_report.to_csv(args.edge_report_csv, index=False, encoding="utf-8-sig")
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    if len(split_report) > 0:
        split_report.to_csv(args.metrics_dir / "pa_tsr_split_report.csv", index=False, encoding="utf-8-sig")
    if len(merge_report) > 0:
        merge_report.to_csv(args.metrics_dir / "pa_tsr_merge_report.csv", index=False, encoding="utf-8-sig")

    metrics = None
    if not args.skip_metrics:
        if args.truth_file.exists():
            metrics = compute_and_save_sort_metrics(args.truth_file, final_sigidx, args.metrics_dir, args, prefix="pa_tsr_hdbscan")
        else:
            metrics = {"skipped": True, "reason": f"truth file not found: {args.truth_file}"}

    summary = {
        "method": "PA-TSR-HDBSCAN",
        "input_file": str(args.input_file),
        "raw_output_file": str(args.raw_output_file),
        "final_output_file": str(args.output_file),
        "total_pulses": int(len(df)),
        "num_windows": int(len(window_groups)),
        "num_tracklets": int(len(tracklets)),
        "hdbscan_backend": backend,
        "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
        "hdbscan_min_samples": int(args.hdbscan_min_samples),
        "window_seconds": float(args.window_seconds),
        "pa_initial_merge": str(args.pa_initial_merge),
        "candidate_edges": int(edge_report.attrs.get("candidate_edges", len(edge_report))),
        "filtered_edges": int(edge_report.attrs.get("filtered_edges", len(edge_report))),
        "accepted_edges": int(len(edge_report)),
        "raw_sort_summary": _sigidx_summary(raw_sigidx),
        "final_sort_summary": _sigidx_summary(final_sigidx),
        "pa_features": feature_info,
        "tsr": tsr_summary,
        "metrics": metrics,
    }
    _ensure_parent(args.report_json)
    args.report_json.write_text(json.dumps(_safe_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return _safe_json(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end PA-TSR-HDBSCAN sorter: PDW -> HDBSCAN -> physical reduce/rescue -> GMM band split."
    )

    # Files and runtime switches.
    parser.add_argument("--input_file", "--test_data", type=Path, default=Path("./edata/Test_Data/Sample_2/Merge_PDW_Data.txt"))
    parser.add_argument("--truth_file", "--test_labels", type=Path, default=Path("./edata/Test_Data/Sample_2/Sorted_PDW.txt"))
    parser.add_argument("--output_file", "--test_sort_file", type=Path, default=Path("./Sorted_PDW_pred_pa_tsr_hdbscan.txt"))
    parser.add_argument("--raw_output_file", type=Path, default=Path("./outputs_pa_tsr/raw_pa_hdbscan.txt"))
    parser.add_argument("--report_json", type=Path, default=Path("./outputs_pa_tsr/pa_tsr_hdbscan_summary.json"))
    parser.add_argument("--metrics_dir", type=Path, default=Path("./outputs_pa_tsr"))
    parser.add_argument("--window_report_csv", type=Path, default=Path("./outputs_pa_tsr/pa_hdbscan_windows.csv"))
    parser.add_argument("--edge_report_csv", type=Path, default=Path("./outputs_pa_tsr/pa_hdbscan_edges.csv"))
    parser.add_argument("--max_pulses", type=int, default=0)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--emit_label99", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10)

    # Window-level HDBSCAN settings.
    parser.add_argument("--hdbscan_backend", choices=["auto", "external", "sklearn", "optics"], default="auto")
    parser.add_argument("--allow_optics_fallback", action="store_true")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=20)
    parser.add_argument("--hdbscan_min_samples", type=int, default=8)
    parser.add_argument("--window_seconds", type=float, default=0.1)
    parser.add_argument("--min_tracklet_size", type=int, default=5)
    parser.set_defaults(assign_noise=True)
    parser.add_argument("--assign_noise", dest="assign_noise", action="store_true")
    parser.add_argument("--no_assign_noise", dest="assign_noise", action="store_false")
    parser.add_argument("--noise_assign_max_dist", type=float, default=0.50)

    # Cross-window conservative tracklet merge settings.
    parser.add_argument("--merge_thresh", type=float, default=3.0)
    parser.add_argument("--max_window_gap", type=int, default=2)
    parser.add_argument("--tol_p1", type=float, default=150.0)
    parser.add_argument("--tol_p2", type=float, default=0.2)
    parser.add_argument("--tol_p4", type=float, default=10.0)
    parser.add_argument("--tol_p5_deg", type=float, default=20.0)
    parser.add_argument("--tol_pri_us", type=float, default=40.0)
    parser.add_argument("--tol_phase_us", type=float, default=40.0)
    parser.add_argument("--hard_gate_p1", type=float, default=1.5)
    parser.add_argument("--hard_gate_p2", type=float, default=1.5)
    parser.add_argument("--hard_gate_p4", type=float, default=1.5)
    parser.add_argument("--hard_gate_p5", type=float, default=1.5)
    parser.add_argument("--hard_gate_pri", type=float, default=2.0)
    parser.add_argument("--hard_gate_phase", type=float, default=2.0)
    parser.add_argument("--w_p1", type=float, default=1.4)
    parser.add_argument("--w_p2", type=float, default=1.0)
    parser.add_argument("--w_p4", type=float, default=0.8)
    parser.add_argument("--w_p5", type=float, default=1.2)
    parser.add_argument("--w_pri", type=float, default=1.2)
    parser.add_argument("--w_phase", type=float, default=1.2)

    # Physics-aware adaptive embedding weights.
    parser.add_argument("--pa_local_window", type=int, default=9)
    parser.add_argument("--pa_w_p1", type=float, default=1.0)
    parser.add_argument("--pa_w_p2", type=float, default=1.0)
    parser.add_argument("--pa_w_p4", type=float, default=1.0)
    parser.add_argument("--pa_w_p5", type=float, default=1.0)

    # TSR physical refinement: fragment reduction and noise rescue.
    parser.add_argument("--tsr_reduce_merge_thresh", type=float, default=0.2)
    parser.add_argument("--tsr_reduce_hard_gate", type=float, default=1.0)
    parser.add_argument("--tsr_rescue_anchor_min_count", type=int, default=500)
    parser.add_argument("--tsr_rescue_merge_thresh", type=float, default=0.75)
    parser.add_argument("--tsr_rescue_hard_gate", type=float, default=0.5)

    # TSR parameter-band split. Manual groups are optional; --tsr_band_split_auto is preferred.
    parser.add_argument("--tsr_band_split_sigidx", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--tsr_band_split_groups", type=str, default="", help="Example: '22,42,41:4:all;25:3:all'.")
    parser.add_argument("--tsr_band_split_auto", action="store_true", help="Learn GMM band-split groups from current SigIdx automatically.")
    parser.add_argument("--tsr_band_split_feature", type=str, default="Param1", help=argparse.SUPPRESS)
    parser.add_argument("--tsr_band_split_components", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--tsr_band_split_group", choices=["middle", "lower", "upper", "all"], default="middle", help=argparse.SUPPRESS)
    parser.add_argument("--tsr_band_split_min_count", type=int, default=1000, help=argparse.SUPPRESS)
    parser.add_argument("--tsr_band_split_min_child_count", type=int, default=100, help=argparse.SUPPRESS)
    parser.set_defaults(tsr_band_split_shared_child=True, tsr_band_split_merge_rest=True, tsr_band_split_keep_largest=True)
    parser.add_argument("--no_tsr_band_split_shared_child", dest="tsr_band_split_shared_child", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--no_tsr_band_split_merge_rest", dest="tsr_band_split_merge_rest", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--no_tsr_band_split_keep_largest", dest="tsr_band_split_keep_largest", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--tsr_auto_chunk_seconds", type=float, default=0.2, help=argparse.SUPPRESS)
    parser.add_argument("--tsr_auto_min_pair_min_overlap", type=float, default=0.50, help=argparse.SUPPRESS)
    parser.add_argument("--tsr_auto_strong_pair_overlap", type=float, default=0.60, help=argparse.SUPPRESS)
    parser.add_argument("--tsr_auto_min_strong_pair_fraction", type=float, default=0.50, help=argparse.SUPPRESS)
    parser.add_argument("--tsr_auto_max_component_dominance", type=float, default=0.80, help=argparse.SUPPRESS)
    parser.set_defaults(
        tsr_auto_min_batch_count=3000,
        tsr_auto_max_batch_count=250000,
        tsr_auto_max_components=5,
        tsr_auto_min_components=3,
        tsr_auto_bic_gain=2.5,
        tsr_auto_sample_size=12000,
        tsr_auto_min_component_fraction=0.005,
        tsr_auto_merge_sep=2.0,
        tsr_auto_group_tol=15.0,
        tsr_auto_middle_dominance=0.60,
        tsr_auto_max_groups=3,
        tsr_auto_max_source_ids=5,
        tsr_auto_min_group_pulses=20000,
        tsr_auto_small_group_bic_gain=5.0,
        tsr_auto_medium_group_pulses=100000,
        tsr_auto_medium_group_bic_gain=5.0,
        tsr_auto_min_global_band_fraction=0.02,
        tsr_auto_bgmm_max_components=6,
        tsr_auto_bgmm_weight_concentration=0.1,
        tsr_auto_bgmm_weight_threshold=0.005,
        tsr_auto_bgmm_max_iter=300,
        tsr_auto_chunk_seconds=0.2,
        tsr_auto_min_pair_min_overlap=0.50,
        tsr_auto_strong_pair_overlap=0.60,
        tsr_auto_min_strong_pair_fraction=0.50,
        tsr_auto_max_component_dominance=0.80,
    )

    # Evaluation settings.
    parser.add_argument("--sort_purity_threshold", type=float, default=0.90)
    parser.add_argument("--sort_min_target_fraction", type=float, default=0.10)
    parser.add_argument("--sort_mix_fail_min_pulses", type=int, default=150)
    parser.add_argument("--sort_chunk_seconds", type=float, default=0.2)

    # Hidden defaults consumed by the HDBSCAN/conservative/TSR helper functions.
    parser.set_defaults(
        hdbscan_metric="euclidean",
        hdbscan_alpha=1.0,
        hdbscan_cluster_selection_epsilon=0.0,
        hdbscan_cluster_selection_method="eom",
        hdbscan_allow_single_cluster=False,
        hdbscan_leaf_size=40,
        optics_max_eps=np.inf,
        optics_xi=0.05,
        optics_cluster_method="xi",
        min_valid_pri_us=1e-6,
        missing_pri_penalty=0.5,
        missing_phase_penalty=0.5,
        cons_mutual_nearest=True,
        cons_check_component=True,
        cons_component_span=2.8,
        cons_component_angle_span=2.8,
        cons_component_pri_span=2.5,
        pa_initial_merge="conservative",
        pa_profile="stable",
        pa_distance_mode="embedding",
        pa_precomputed_max_points=2500,
        pa_adapt_strength=0.0,
        pa_dtoa_adapt_strength=1.2,
        pa_reliability_floor=1.0,
        pa_missing_pulse_ratio=2.8,
        pa_missing_dtoa_weight=0.45,
        pa_weight_floor=0.05,
        pa_weight_ceiling=6.0,
        pa_local_scale_strength=0.0,
        pa_local_scale_min=0.75,
        pa_local_scale_max=2.5,
        pa_disable_adaptive_weights=False,
        pa_w_dtoa=0.15,
        pa_w_dtoa_local=0.08,
        pa_w_period=0.12,
        pa_w_aux=0.35,
        tsr_reduce_min_cluster_size=1,
        tsr_reduce_min_batch_fraction=0.0,
        tsr_reduce_w_p1=1.4,
        tsr_reduce_w_p2=1.0,
        tsr_reduce_w_p4=0.8,
        tsr_reduce_w_p5=1.2,
        tsr_reduce_tol_p1=150.0,
        tsr_reduce_tol_p2=0.2,
        tsr_reduce_tol_p4=10.0,
        tsr_reduce_tol_p5_deg=20.0,
        tsr_rescue_w_p1=1.4,
        tsr_rescue_w_p2=1.0,
        tsr_rescue_w_p4=0.8,
        tsr_rescue_w_p5=1.2,
        tsr_rescue_tol_p1=150.0,
        tsr_rescue_tol_p2=0.2,
        tsr_rescue_tol_p4=10.0,
        tsr_rescue_tol_p5_deg=20.0,
        tsr_rescue_chunk_size=30000,
        tsr_rescue_anchor_block=128,
        tsr_band_split_reg_covar=1e-6,
        tsr_band_split_n_init=5,
        tsr_band_split_max_iter=200,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pa_tsr_sort(args)
    print("Summary:")
    print(json.dumps(_safe_json(summary), indent=2, ensure_ascii=False))
    metrics = summary.get("metrics")
    if metrics and not metrics.get("skipped"):
        print("-" * 80)
        print("Direct sorting metrics, no XGBoost LABEL recognition:")
        print(f"Sort Acc    : {metrics['sample_sort_acc']:.4f} ({metrics['sample_sort_acc'] * 100:.2f}%)")
        print(f"MR          : {metrics['MR']:.4f} ({metrics['MR'] * 100:.2f}%)")
        print(f"MP          : {metrics['MP']:.4f} ({metrics['MP'] * 100:.2f}%)")
        print(f"MIOU        : {metrics['MIOU']:.4f} ({metrics['MIOU'] * 100:.2f}%)")
        print(f"Extra Batch : {metrics['sample_extra_batch_rate']:.4f} ({metrics['sample_extra_batch_rate'] * 100:.2f}%)")
        print(f"Wrong Batch : {metrics['sample_wrong_batch_rate']:.4f} ({metrics['sample_wrong_batch_rate'] * 100:.2f}%)")
        tracking = metrics.get("sample_signal_tracking_stability")
        if tracking is not None:
            print(f"Tracking    : {tracking:.4f} ({tracking * 100:.2f}%)")
    elif metrics:
        print(f"Metrics skipped: {metrics['reason']}")
    print(f"Saved final sort file: {args.output_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sliding-window tracklet sorter with PRI/DTOA consistency.

This is a replacement candidate for GATE.py and an upgrade over
tracklet_sort.py. The main change is in cross-window tracklet merging:
tracklets are merged only when their parameter centers are close AND their
PRI/DTOA rhythm is reasonably consistent.

Output format is aligned with XGBoost.py:
    TOA(s) SigIdx
one row per input PDW pulse.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


PDW_COLUMNS = [
    "TOA(s)",
    "Param1",
    "Param2",
    "Param3",
    "Param4",
    "Param5",
    "Param6",
    "Param7",
]


@dataclass
class Tracklet:
    node_id: int
    window_id: int
    count: int
    start_toa: float
    end_toa: float
    center_p1: float
    center_p2: float
    center_p3: float
    center_p4: float
    center_p5: float
    center_p6: float
    center_p7: float
    pri_us: float
    pri_iqr_us: float


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def read_pdw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    if df.shape[1] < len(PDW_COLUMNS):
        raise ValueError(f"PDW file has {df.shape[1]} columns, expected at least {len(PDW_COLUMNS)}: {path}")

    if list(df.columns[: len(PDW_COLUMNS)]) != PDW_COLUMNS:
        df = df.iloc[:, : len(PDW_COLUMNS)].copy()
        df.columns = PDW_COLUMNS
    else:
        df = df[PDW_COLUMNS].copy()

    for col in PDW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = int(df[PDW_COLUMNS].isna().any(axis=1).sum())
    if bad_rows:
        raise ValueError(f"PDW file contains {bad_rows} non-numeric rows: {path}")
    return df.reset_index(drop=True)


def circular_mean_deg(values: np.ndarray) -> float:
    radians = np.deg2rad(np.mod(values.astype(np.float64), 360.0))
    sin_mean = float(np.sin(radians).mean())
    cos_mean = float(np.cos(radians).mean())
    return float((np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0)


def circular_abs_diff_deg(values: np.ndarray, ref: float) -> np.ndarray:
    return np.abs((values - ref + 180.0) % 360.0 - 180.0)


def dense_relabel(sigidx: np.ndarray) -> np.ndarray:
    out = sigidx.copy()
    valid_ids = sorted(int(v) for v in np.unique(out) if int(v) > 0)
    mapping = {old: new for new, old in enumerate(valid_ids, start=1)}
    for old, new in mapping.items():
        out[out == old] = new
    return out


def robust_scaled_features(df: pd.DataFrame, feature_mode: str) -> np.ndarray:
    angle = np.deg2rad(np.mod(df["Param5"].to_numpy(dtype=np.float64), 360.0))
    if feature_mode == "merge":
        raw = np.column_stack(
            [
                df["Param1"].to_numpy(dtype=np.float64),
                df["Param2"].to_numpy(dtype=np.float64),
                df["Param4"].to_numpy(dtype=np.float64),
                np.sin(angle),
                np.cos(angle),
            ]
        )
    else:
        raw = np.column_stack(
            [
                df["Param1"].to_numpy(dtype=np.float64),
                df["Param2"].to_numpy(dtype=np.float64),
                df["Param3"].to_numpy(dtype=np.float64),
                df["Param4"].to_numpy(dtype=np.float64),
                np.sin(angle),
                np.cos(angle),
                df["Param6"].to_numpy(dtype=np.float64),
                df["Param7"].to_numpy(dtype=np.float64),
            ]
        )

    median = np.median(raw, axis=0)
    q75 = np.percentile(raw, 75, axis=0)
    q25 = np.percentile(raw, 25, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return ((raw - median) / scale).astype(np.float32)


def estimate_pri_us(toa_values: np.ndarray) -> Tuple[float, float]:
    if len(toa_values) < 3:
        return float("nan"), float("nan")
    sorted_toa = np.sort(toa_values.astype(np.float64))
    dtoa_us = np.diff(sorted_toa) * 1e6
    dtoa_us = dtoa_us[dtoa_us > 1e-9]
    if len(dtoa_us) < 2:
        return float("nan"), float("nan")
    pri = float(np.median(dtoa_us))
    iqr = float(np.percentile(dtoa_us, 75) - np.percentile(dtoa_us, 25))
    return pri, iqr


def make_window_groups(toa: np.ndarray, window_seconds: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    if window_seconds <= 0:
        raise ValueError("--window_seconds must be positive.")
    t0 = float(np.min(toa))
    window_ids = np.floor((toa.astype(np.float64) - t0) / window_seconds).astype(np.int64)
    order = np.argsort(window_ids, kind="mergesort")
    ordered_windows = window_ids[order]
    split_points = np.flatnonzero(np.diff(ordered_windows)) + 1
    groups = [part for part in np.split(order, split_points) if len(part) > 0]
    return window_ids, groups


def choose_window_k(n_pulses: int, args: argparse.Namespace) -> int:
    if args.window_k > 0:
        k = int(args.window_k)
    else:
        k = int(np.ceil(n_pulses / max(args.target_pulses_per_cluster, 1)))
        k = max(args.min_window_k, min(args.max_window_k, k))
    max_by_size = max(1, n_pulses // max(args.min_tracklet_size, 1))
    return max(1, min(k, max_by_size))


def create_tracklets(
    df: pd.DataFrame,
    features: np.ndarray,
    window_groups: List[np.ndarray],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, List[Tracklet], pd.DataFrame]:
    n = len(df)
    tracklet_ids = np.full(n, -1, dtype=np.int64)
    tracklets: List[Tracklet] = []
    report_rows = []

    params = df[PDW_COLUMNS].to_numpy(dtype=np.float64)
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    t0 = float(np.min(toa))

    for pos, idx in enumerate(window_groups, start=1):
        window_id = int(np.floor((toa[idx[0]] - t0) / args.window_seconds))
        n_window = len(idx)
        if n_window < args.min_tracklet_size:
            report_rows.append(
                {
                    "window_id": window_id,
                    "num_pulses": n_window,
                    "num_tracklets": 0,
                    "k": 0,
                    "reason": "window_too_small",
                }
            )
            continue

        k = choose_window_k(n_window, args)
        if k <= 1:
            labels = np.zeros(n_window, dtype=np.int64)
        else:
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=args.seed + window_id,
                batch_size=args.kmeans_batch_size,
                n_init=args.kmeans_n_init,
                max_iter=args.kmeans_max_iter,
            )
            labels = model.fit_predict(features[idx]).astype(np.int64)

        counts = np.bincount(labels, minlength=max(k, int(labels.max()) + 1))
        valid_clusters = [int(c) for c, count in enumerate(counts) if int(count) >= args.min_tracklet_size]

        if args.assign_small_clusters and valid_clusters:
            centers = np.vstack([features[idx[labels == c]].mean(axis=0) for c in valid_clusters]).astype(np.float32)
            for c, count in enumerate(counts):
                if int(count) >= args.min_tracklet_size:
                    continue
                small_pos = np.flatnonzero(labels == c)
                if len(small_pos) == 0:
                    continue
                small_center = features[idx[small_pos]].mean(axis=0)
                nearest = valid_clusters[int(np.argmin(np.linalg.norm(centers - small_center, axis=1)))]
                labels[small_pos] = nearest
            counts = np.bincount(labels, minlength=max(k, int(labels.max()) + 1))
            valid_clusters = [int(c) for c, count in enumerate(counts) if int(count) >= args.min_tracklet_size]

        made = 0
        for cluster_id in valid_clusters:
            pulse_idx = idx[labels == cluster_id]
            if len(pulse_idx) < args.min_tracklet_size:
                continue
            node_id = len(tracklets)
            tracklet_ids[pulse_idx] = node_id

            p = params[pulse_idx]
            pri_us, pri_iqr_us = estimate_pri_us(toa[pulse_idx])
            tracklets.append(
                Tracklet(
                    node_id=node_id,
                    window_id=window_id,
                    count=int(len(pulse_idx)),
                    start_toa=float(np.min(toa[pulse_idx])),
                    end_toa=float(np.max(toa[pulse_idx])),
                    center_p1=float(np.median(p[:, 1])),
                    center_p2=float(np.median(p[:, 2])),
                    center_p3=float(np.median(p[:, 3])),
                    center_p4=float(np.median(p[:, 4])),
                    center_p5=circular_mean_deg(p[:, 5]),
                    center_p6=float(np.median(p[:, 6])),
                    center_p7=float(np.median(p[:, 7])),
                    pri_us=pri_us,
                    pri_iqr_us=pri_iqr_us,
                )
            )
            made += 1

        report_rows.append(
            {
                "window_id": window_id,
                "num_pulses": n_window,
                "num_tracklets": made,
                "k": k,
                "reason": "",
            }
        )
        if args.verbose and (pos % 10 == 0 or pos == len(window_groups)):
            print(f"[tracklet-pri] windows {pos}/{len(window_groups)}, total_tracklets={len(tracklets)}")

    return tracklet_ids, tracklets, pd.DataFrame(report_rows)


def _pri_phase_distances(a: Tracklet, b: Tracklet, args: argparse.Namespace) -> Tuple[float, float]:
    pri_a = float(a.pri_us)
    pri_b = float(b.pri_us)
    has_pri = np.isfinite(pri_a) and np.isfinite(pri_b) and pri_a > args.min_valid_pri_us and pri_b > args.min_valid_pri_us
    if not has_pri:
        return float(args.missing_pri_penalty), float(args.missing_phase_penalty)

    pri_ref = 0.5 * (pri_a + pri_b)
    d_pri = abs(pri_a - pri_b) / max(args.tol_pri_us, 1e-12)

    gap_us = max((float(b.start_toa) - float(a.end_toa)) * 1e6, 0.0)
    k = max(1, int(round(gap_us / max(pri_ref, 1e-12))))
    phase_err_us = abs(gap_us - k * pri_ref)
    d_phase = phase_err_us / max(args.tol_phase_us, 1e-12)
    return float(d_pri), float(d_phase)


def merge_tracklets(tracklets: List[Tracklet], args: argparse.Namespace) -> Tuple[np.ndarray, pd.DataFrame]:
    if not tracklets:
        return np.zeros(0, dtype=np.int64), pd.DataFrame()

    uf = UnionFind(len(tracklets))
    by_window: Dict[int, List[int]] = {}
    for tr in tracklets:
        by_window.setdefault(tr.window_id, []).append(tr.node_id)

    p1 = np.array([tr.center_p1 for tr in tracklets], dtype=np.float64)
    p2 = np.array([tr.center_p2 for tr in tracklets], dtype=np.float64)
    p4 = np.array([tr.center_p4 for tr in tracklets], dtype=np.float64)
    p5 = np.array([tr.center_p5 for tr in tracklets], dtype=np.float64)

    edge_rows = []
    windows = sorted(by_window)
    for wi in windows:
        left_ids = by_window[wi]
        right_ids: List[int] = []
        for gap in range(1, args.max_window_gap + 1):
            right_ids.extend(by_window.get(wi + gap, []))
        if not right_ids:
            continue

        right = np.array(right_ids, dtype=np.int64)
        for a_id in left_ids:
            d_p1 = np.abs(p1[right] - p1[a_id]) / max(args.tol_p1, 1e-12)
            d_p2 = np.abs(p2[right] - p2[a_id]) / max(args.tol_p2, 1e-12)
            d_p4 = np.abs(p4[right] - p4[a_id]) / max(args.tol_p4, 1e-12)
            d_p5 = circular_abs_diff_deg(p5[right], p5[a_id]) / max(args.tol_p5_deg, 1e-12)
            param_hard_ok = (
                (d_p1 <= args.hard_gate_p1)
                & (d_p2 <= args.hard_gate_p2)
                & (d_p4 <= args.hard_gate_p4)
                & (d_p5 <= args.hard_gate_p5)
            )
            param_dist = args.w_p1 * d_p1 + args.w_p2 * d_p2 + args.w_p4 * d_p4 + args.w_p5 * d_p5

            candidate_pos = np.flatnonzero(param_hard_ok)
            for pos in candidate_pos:
                b_id = int(right[pos])
                d_pri, d_phase = _pri_phase_distances(tracklets[a_id], tracklets[b_id], args)
                if d_pri > args.hard_gate_pri or d_phase > args.hard_gate_phase:
                    continue
                total_dist = float(param_dist[pos] + args.w_pri * d_pri + args.w_phase * d_phase)
                if total_dist > args.merge_thresh:
                    continue
                uf.union(int(a_id), int(b_id))
                edge_rows.append(
                    {
                        "a": int(a_id),
                        "b": int(b_id),
                        "window_a": int(tracklets[a_id].window_id),
                        "window_b": int(tracklets[b_id].window_id),
                        "param_dist": float(param_dist[pos]),
                        "pri_dist": float(d_pri),
                        "phase_dist": float(d_phase),
                        "total_dist": total_dist,
                    }
                )

    roots = np.array([uf.find(i) for i in range(len(tracklets))], dtype=np.int64)
    return roots, pd.DataFrame(edge_rows)


def roots_to_sigidx(tracklet_ids: np.ndarray, roots: np.ndarray) -> np.ndarray:
    sigidx = np.zeros(len(tracklet_ids), dtype=np.int64)
    if len(roots) == 0:
        return sigidx
    unique_roots = sorted(int(v) for v in np.unique(roots))
    root_to_sigidx = {root: pos for pos, root in enumerate(unique_roots, start=1)}
    assigned = tracklet_ids >= 0
    sigidx[assigned] = np.array([root_to_sigidx[int(roots[int(tid)])] for tid in tracklet_ids[assigned]], dtype=np.int64)
    return dense_relabel(sigidx)


def write_sort_file(df: pd.DataFrame, sigidx: np.ndarray, output_file: Path, emit_label99: bool) -> None:
    out = pd.DataFrame(
        {
            "TOA(s)": df["TOA(s)"].map(lambda x: f"{float(x):.9f}"),
            "SigIdx": sigidx.astype(np.int64),
        }
    )
    if emit_label99:
        out["LABEL"] = 99
    out.to_csv(output_file, sep=" ", index=False)


def summarize(sigidx: np.ndarray) -> Dict[str, int]:
    positive = sigidx[sigidx > 0]
    return {
        "total_pulses": int(len(sigidx)),
        "positive_sigidx_count": int(len(np.unique(positive))) if len(positive) else 0,
        "noise_or_unassigned_count": int(np.sum(sigidx <= 0)),
    }


def read_truth_file(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path, sep=r"\s+", engine="python")
    required = ["TOA(s)", "SigIdx", "LABEL"]
    missing = [col for col in required if col not in truth.columns]
    if missing:
        raise ValueError(f"Truth file is missing columns {missing}: {path}")
    truth = truth[required].copy()
    for col in required:
        truth[col] = pd.to_numeric(truth[col], errors="coerce")
    bad_rows = int(truth.isna().any(axis=1).sum())
    if bad_rows:
        raise ValueError(f"Truth file contains {bad_rows} non-numeric rows: {path}")
    return truth.reset_index(drop=True)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def compute_and_save_sort_metrics(
    truth_file: Path,
    pred_sigidx: np.ndarray,
    metrics_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    from 识别.sort_metrics import compute_sort_metrics_by_beat

    truth = read_truth_file(truth_file)
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
    metrics["recognition_note"] = "N/A: tracklet_pri_sort.py does not predict LABEL. Run XGBoost.py to compute recognition accuracy."

    fail_reason = sort_batch_df["fail_reason"].fillna("").astype(str) if len(sort_batch_df) else pd.Series(dtype=str)
    total_rows = int(len(fail_reason))
    failed_rows = int((fail_reason != "").sum()) if total_rows else 0
    metrics["fail_reason_counts"] = {
        "low_purity": int((fail_reason == "low_purity").sum()) if total_rows else 0,
        "low_target_fraction": int((fail_reason == "low_target_fraction").sum()) if total_rows else 0,
        "swallowed_other_target": int((fail_reason == "swallowed_other_target").sum()) if total_rows else 0,
    }
    metrics["fail_reason_rates_all"] = {
        key: float(count / total_rows) if total_rows else float("nan")
        for key, count in metrics["fail_reason_counts"].items()
    }
    metrics["fail_reason_rates_failed"] = {
        key: float(count / failed_rows) if failed_rows else float("nan")
        for key, count in metrics["fail_reason_counts"].items()
    }

    metrics_dir.mkdir(parents=True, exist_ok=True)
    sort_batch_df.to_csv(metrics_dir / "tracklet_pri_sort_batch_eval.csv", index=False, encoding="utf-8-sig")
    sort_target_df.to_csv(metrics_dir / "tracklet_pri_sort_target_accuracy.csv", index=False, encoding="utf-8-sig")
    sort_target_beat_df.to_csv(metrics_dir / "tracklet_pri_sort_target_beat_eval.csv", index=False, encoding="utf-8-sig")
    sort_beat_df.to_csv(metrics_dir / "tracklet_pri_sort_beat_eval.csv", index=False, encoding="utf-8-sig")
    (metrics_dir / "tracklet_pri_sort_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return json_safe(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tracklet + PRI/DTOA graph PDW sorter.")
    parser.add_argument(
        "--input_file",
        "--test_data",
        dest="input_file",
        type=Path,
        default=Path("E:/\u5206\u9009/Data/Test_Data/Sample_1/Merge_PDW_Data.txt"),
        help="Input PDW file. Alias --test_data matches XGBoost.py.",
    )
    parser.add_argument(
        "--output_file",
        "--test_sort_file",
        dest="output_file",
        type=Path,
        default=Path("./Sorted_PDW_pred_tracklet_pri.txt"),
        help="Output sorting file. Alias --test_sort_file matches XGBoost.py.",
    )
    parser.add_argument(
        "--truth_file",
        "--test_labels",
        dest="truth_file",
        type=Path,
        default=Path("E:/\u5206\u9009/Data/Test_Data/Sample_1/Sorted_PDW.txt"),
        help="Truth Sorted_PDW.txt for direct sorting metrics. Alias --test_labels matches XGBoost.py.",
    )
    parser.add_argument("--emit_label99", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--metrics_dir", type=Path, default=Path("./outputs_xgboost_pdw"))
    parser.add_argument("--report_json", type=Path, default=Path("./outputs_xgboost_pdw/tracklet_pri_sort_summary.json"))
    parser.add_argument("--tracklet_report_csv", type=Path, default=Path("./outputs_xgboost_pdw/tracklet_pri_windows.csv"))
    parser.add_argument("--edge_report_csv", type=Path, default=Path("./outputs_xgboost_pdw/tracklet_pri_edges.csv"))

    parser.add_argument("--sort_purity_threshold", type=float, default=0.90)
    parser.add_argument("--sort_min_target_fraction", type=float, default=0.10)
    parser.add_argument("--sort_mix_fail_min_pulses", type=int, default=150)
    parser.add_argument("--sort_chunk_seconds", type=float, default=0.2)

    parser.add_argument("--window_seconds", type=float, default=0.2)
    parser.add_argument("--window_k", type=int, default=24, help="Fixed KMeans clusters per window. Use 0 for automatic K.")
    parser.add_argument("--min_window_k", type=int, default=8)
    parser.add_argument("--max_window_k", type=int, default=32)
    parser.add_argument("--target_pulses_per_cluster", type=int, default=3000)
    parser.add_argument("--min_tracklet_size", type=int, default=5)
    parser.set_defaults(assign_small_clusters=True)
    parser.add_argument("--assign_small_clusters", dest="assign_small_clusters", action="store_true")
    parser.add_argument("--no_assign_small_clusters", dest="assign_small_clusters", action="store_false")
    parser.add_argument("--feature_mode", choices=["all", "merge"], default="all")

    parser.add_argument("--max_window_gap", type=int, default=2)
    parser.add_argument("--merge_thresh", type=float, default=3.0)
    parser.add_argument("--w_p1", type=float, default=1.4)
    parser.add_argument("--w_p2", type=float, default=1.0)
    parser.add_argument("--w_p4", type=float, default=0.8)
    parser.add_argument("--w_p5", type=float, default=1.2)
    parser.add_argument("--w_pri", type=float, default=0.8)
    parser.add_argument("--w_phase", type=float, default=0.8)
    parser.add_argument("--tol_p1", type=float, default=150.0)
    parser.add_argument("--tol_p2", type=float, default=0.2)
    parser.add_argument("--tol_p4", type=float, default=10.0)
    parser.add_argument("--tol_p5_deg", type=float, default=20.0)
    parser.add_argument("--tol_pri_us", type=float, default=50.0)
    parser.add_argument("--tol_phase_us", type=float, default=50.0)
    parser.add_argument("--hard_gate_p1", type=float, default=1.5)
    parser.add_argument("--hard_gate_p2", type=float, default=1.5)
    parser.add_argument("--hard_gate_p4", type=float, default=1.5)
    parser.add_argument("--hard_gate_p5", type=float, default=1.5)
    parser.add_argument("--hard_gate_pri", type=float, default=3.0)
    parser.add_argument("--hard_gate_phase", type=float, default=3.0)
    parser.add_argument("--min_valid_pri_us", type=float, default=1e-6)
    parser.add_argument("--missing_pri_penalty", type=float, default=0.5)
    parser.add_argument("--missing_phase_penalty", type=float, default=0.5)

    parser.add_argument("--kmeans_batch_size", type=int, default=8192)
    parser.add_argument("--kmeans_n_init", type=int, default=3)
    parser.add_argument("--kmeans_max_iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    df = read_pdw(args.input_file)
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    print(f"Input PDW: {args.input_file}")
    print(f"Pulses:    {len(df)}")

    features = robust_scaled_features(df, args.feature_mode)
    _, window_groups = make_window_groups(toa, args.window_seconds)
    print(f"Windows:   {len(window_groups)} ({args.window_seconds:.3f}s)")

    tracklet_ids, tracklets, window_report = create_tracklets(df, features, window_groups, args)
    roots, edge_report = merge_tracklets(tracklets, args)
    sigidx = roots_to_sigidx(tracklet_ids, roots)
    write_sort_file(df, sigidx, args.output_file, emit_label99=args.emit_label99)

    args.tracklet_report_csv.parent.mkdir(parents=True, exist_ok=True)
    window_report.to_csv(args.tracklet_report_csv, index=False, encoding="utf-8-sig")
    edge_report.to_csv(args.edge_report_csv, index=False, encoding="utf-8-sig")

    summary = summarize(sigidx)
    summary.update(
        {
            "input_file": str(args.input_file),
            "output_file": str(args.output_file),
            "num_windows": int(len(window_groups)),
            "num_tracklets": int(len(tracklets)),
            "graph_edges": int(len(edge_report)),
            "window_seconds": float(args.window_seconds),
            "window_k": int(args.window_k),
            "merge_thresh": float(args.merge_thresh),
            "w_pri": float(args.w_pri),
            "w_phase": float(args.w_phase),
            "tol_pri_us": float(args.tol_pri_us),
            "tol_phase_us": float(args.tol_phase_us),
        }
    )

    direct_metrics = None
    if not args.skip_metrics:
        if args.truth_file.exists():
            direct_metrics = compute_and_save_sort_metrics(
                truth_file=args.truth_file,
                pred_sigidx=sigidx,
                metrics_dir=args.metrics_dir,
                args=args,
            )
            summary["direct_sort_metrics"] = direct_metrics
        else:
            summary["direct_sort_metrics"] = {
                "skipped": True,
                "reason": f"truth file not found: {args.truth_file}",
            }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Summary:")
    print(json.dumps(json_safe(summary), indent=2, ensure_ascii=False))
    if direct_metrics is not None:
        print("-" * 80)
        print("Direct sorting metrics, no XGBoost LABEL recognition:")
        print(f"分选准确率 Sort Acc : {direct_metrics['sample_sort_acc']:.4f} ({direct_metrics['sample_sort_acc'] * 100:.2f}%)")
        print(f"增批率 Extra Batch  : {direct_metrics['sample_extra_batch_rate']:.4f} ({direct_metrics['sample_extra_batch_rate'] * 100:.2f}%)")
        print(f"错批率 Wrong Batch  : {direct_metrics['sample_wrong_batch_rate']:.4f} ({direct_metrics['sample_wrong_batch_rate'] * 100:.2f}%)")
        print("识别准确率 Recog Acc: N/A (tracklet_pri_sort.py does not predict LABEL)")
        counts = direct_metrics.get("fail_reason_counts", {})
        rates = direct_metrics.get("fail_reason_rates_all", {})
        print(
            f"low_purity          : {counts.get('low_purity', 0)} "
            f"({rates.get('low_purity', 0.0) * 100:.2f}%)"
        )
        print(
            f"low_target_fraction : {counts.get('low_target_fraction', 0)} "
            f"({rates.get('low_target_fraction', 0.0) * 100:.2f}%)"
        )
    print(f"Saved sort file: {args.output_file}")
    print(f"Use with XGBoost: python XGBoost.py --test_sort_file \"{args.output_file}\" --use_reduce_postprocess 0")


if __name__ == "__main__":
    main()

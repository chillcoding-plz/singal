#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DBSCAN based PDW sorter.

This is a standalone alternative to GATE.py / tracklet_sort.py.

Flow:
1. Split pulses into short TOA windows.
2. Run DBSCAN inside each window on robust-scaled PDW features.
3. Treat each DBSCAN cluster as a short tracklet.
4. Merge tracklets across nearby windows with the same parameter + PRI/phase
   graph logic used by tracklet_pri_sort.py.

Output format is aligned with XGBoost.py:
    TOA(s) SigIdx
one row per input PDW pulse.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from tracklet_pri_sort import (
    PDW_COLUMNS,
    Tracklet,
    circular_mean_deg,
    dense_relabel,
    estimate_pri_us,
    json_safe,
    make_window_groups,
    merge_tracklets,
    read_pdw,
    read_truth_file,
    robust_scaled_features,
    roots_to_sigidx,
    summarize,
    write_sort_file,
)


def estimate_window_eps(features: np.ndarray, args: argparse.Namespace, seed: int) -> float:
    """Estimate DBSCAN eps from the k-distance curve in one window."""
    if len(features) <= max(args.dbscan_min_samples, 1):
        return float(args.dbscan_eps)

    sample = features
    if len(features) > args.eps_sample_size:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(features), size=args.eps_sample_size, replace=False)
        sample = features[sample_idx]

    n_neighbors = max(2, min(args.dbscan_min_samples, len(sample)))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=args.dbscan_metric, n_jobs=args.n_jobs)
    nn.fit(sample)
    distances, _ = nn.kneighbors(sample, return_distance=True)
    kth_dist = distances[:, -1]
    eps = float(np.quantile(kth_dist, args.eps_quantile))
    eps = max(args.eps_min, min(args.eps_max, eps * args.eps_scale))
    return eps


def assign_noise_to_clusters(
    labels: np.ndarray,
    window_features: np.ndarray,
    valid_clusters: List[int],
    max_dist: float,
) -> np.ndarray:
    if not valid_clusters:
        return labels
    noise_pos = np.flatnonzero(labels < 0)
    if len(noise_pos) == 0:
        return labels

    centers = np.vstack([window_features[labels == c].mean(axis=0) for c in valid_clusters]).astype(np.float32)
    noise_features = window_features[noise_pos]
    dists = np.linalg.norm(noise_features[:, None, :] - centers[None, :, :], axis=2)
    nearest_pos = np.argmin(dists, axis=1)
    nearest_dist = dists[np.arange(len(noise_pos)), nearest_pos]
    take = nearest_dist <= max_dist
    if np.any(take):
        labels[noise_pos[take]] = np.array(valid_clusters, dtype=np.int64)[nearest_pos[take]]
    return labels


def make_tracklet(
    node_id: int,
    window_id: int,
    pulse_idx: np.ndarray,
    params: np.ndarray,
    toa: np.ndarray,
) -> Tracklet:
    p = params[pulse_idx]
    pri_us, pri_iqr_us = estimate_pri_us(toa[pulse_idx])
    return Tracklet(
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


def create_dbscan_tracklets(
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
    start_time = time.perf_counter()
    total_windows = len(window_groups)

    for pos, idx in enumerate(window_groups, start=1):
        window_id = int(np.floor((toa[idx[0]] - t0) / args.window_seconds))
        n_window = int(len(idx))
        if n_window < args.min_tracklet_size:
            report_rows.append(
                {
                    "window_id": window_id,
                    "num_pulses": n_window,
                    "eps": np.nan,
                    "num_clusters": 0,
                    "num_tracklets": 0,
                    "noise_pulses": n_window,
                    "small_cluster_pulses": 0,
                    "reason": "window_too_small",
                }
            )
            continue

        window_features = features[idx]
        eps = estimate_window_eps(window_features, args, seed=args.seed + window_id) if args.auto_eps else args.dbscan_eps
        model = DBSCAN(
            eps=eps,
            min_samples=args.dbscan_min_samples,
            metric=args.dbscan_metric,
            algorithm=args.dbscan_algorithm,
            leaf_size=args.dbscan_leaf_size,
            n_jobs=args.n_jobs,
        )
        labels = model.fit_predict(window_features).astype(np.int64)

        unique_labels = [int(v) for v in np.unique(labels) if int(v) >= 0]
        counts: Dict[int, int] = {label: int(np.sum(labels == label)) for label in unique_labels}
        valid_clusters = [label for label, count in counts.items() if count >= args.min_tracklet_size]

        small_cluster_pulses = int(sum(count for label, count in counts.items() if label not in valid_clusters))
        noise_before = int(np.sum(labels < 0))
        if args.assign_noise and valid_clusters:
            labels = assign_noise_to_clusters(labels, window_features, valid_clusters, args.noise_assign_max_dist)
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
            tracklets.append(make_tracklet(node_id, window_id, pulse_idx, params, toa))
            made += 1

        report_rows.append(
            {
                "window_id": window_id,
                "num_pulses": n_window,
                "eps": float(eps),
                "num_clusters": int(len(unique_labels)),
                "num_tracklets": int(made),
                "noise_pulses": int(np.sum(labels < 0)),
                "noise_pulses_before_assign": noise_before,
                "small_cluster_pulses": small_cluster_pulses,
                "reason": "",
            }
        )
        should_report = args.progress_every > 0 and (pos % args.progress_every == 0 or pos == total_windows)
        if args.verbose or should_report:
            elapsed = time.perf_counter() - start_time
            windows_per_sec = pos / max(elapsed, 1e-9)
            eta = (total_windows - pos) / max(windows_per_sec, 1e-9)
            print(
                f"[dbscan-sort] windows {pos}/{total_windows} "
                f"({pos / max(total_windows, 1) * 100:.1f}%), "
                f"tracklets={len(tracklets)}, elapsed={elapsed / 60:.1f}min, eta={eta / 60:.1f}min",
                flush=True,
            )

    return tracklet_ids, tracklets, pd.DataFrame(report_rows)


def compute_and_save_sort_metrics(
    truth_file: Path,
    pred_sigidx: np.ndarray,
    metrics_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    from 识别.sort_metrics import compute_sort_metrics_by_beat

    truth = read_truth_file(truth_file)
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
    metrics["recognition_note"] = "N/A: dbscan_sort.py does not predict LABEL. Run XGBoost.py to compute recognition accuracy."

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
    sort_batch_df.to_csv(metrics_dir / "dbscan_sort_batch_eval.csv", index=False, encoding="utf-8-sig")
    sort_target_df.to_csv(metrics_dir / "dbscan_sort_target_accuracy.csv", index=False, encoding="utf-8-sig")
    sort_target_beat_df.to_csv(metrics_dir / "dbscan_sort_target_beat_eval.csv", index=False, encoding="utf-8-sig")
    sort_beat_df.to_csv(metrics_dir / "dbscan_sort_beat_eval.csv", index=False, encoding="utf-8-sig")
    (metrics_dir / "dbscan_sort_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return json_safe(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DBSCAN window-tracklet PDW sorter.")
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
        default=Path("./Sorted_PDW_pred_dbscan.txt"),
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
    # python dbscan_sort.py `
    #   --test_data "E:\分选\Data\Test_Data\Sample_1\Merge_PDW_Data.txt" `
    #   --test_labels "E:\分选\Data\Test_Data\Sample_1\Sorted_PDW.txt" `
    #   --test_sort_file "Sorted_PDW_pred_dbscan_merge2.txt" `
    #   --window_seconds 0.1 `
    #   --feature_mode merge `
    #   --auto_eps `
    #   --eps_quantile 0.90 `
    #   --eps_scale 1.10 `
    #   --eps_max 1.50 `
    #   --dbscan_min_samples 8 `
    #   --min_tracklet_size 5 `
    #   --max_window_gap 3 `
    #   --merge_thresh 3.8 `
    #   --w_pri 1.2 `
    #   --w_phase 1.2 `
    #   --tol_pri_us 70 `
    #   --tol_phase_us 70 `
    #   --hard_gate_pri 3.0 `
    #   --hard_gate_phase 3.0 `
    #   --assign_noise True

    parser.add_argument("--emit_label99", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--metrics_dir", type=Path, default=Path("./outputs_xgboost_pdw"))
    parser.add_argument("--report_json", type=Path, default=Path("./outputs_xgboost_pdw/dbscan_sort_summary.json"))
    parser.add_argument("--window_report_csv", type=Path, default=Path("./outputs_xgboost_pdw/dbscan_windows.csv"))
    parser.add_argument("--edge_report_csv", type=Path, default=Path("./outputs_xgboost_pdw/dbscan_edges.csv"))

    parser.add_argument("--sort_purity_threshold", type=float, default=0.90)
    parser.add_argument("--sort_min_target_fraction", type=float, default=0.10)
    parser.add_argument("--sort_mix_fail_min_pulses", type=int, default=150)
    parser.add_argument("--sort_chunk_seconds", type=float, default=0.2)

    parser.add_argument("--window_seconds", type=float, default=0.1)
    parser.add_argument("--min_tracklet_size", type=int, default=5)
    parser.add_argument("--feature_mode", choices=["all", "merge"], default="merge")
    parser.add_argument("--max_pulses", type=int, default=0, help="Debug only. Use first N pulses when > 0.")

    parser.set_defaults(auto_eps=True)
    parser.add_argument("--auto_eps", dest="auto_eps", action="store_true")
    parser.add_argument("--no_auto_eps", dest="auto_eps", action="store_false")
    parser.add_argument("--dbscan_eps", type=float, default=0.65)
    parser.add_argument("--eps_quantile", type=float, default=0.85)
    parser.add_argument("--eps_scale", type=float, default=1.0)
    parser.add_argument("--eps_min", type=float, default=0.20)
    parser.add_argument("--eps_max", type=float, default=1.20)
    parser.add_argument("--eps_sample_size", type=int, default=10000)
    parser.add_argument("--dbscan_min_samples", type=int, default=8)
    parser.add_argument("--dbscan_metric", type=str, default="euclidean")
    parser.add_argument("--dbscan_algorithm", choices=["auto", "ball_tree", "kd_tree", "brute"], default="auto")
    parser.add_argument("--dbscan_leaf_size", type=int, default=40)
    parser.add_argument("--n_jobs", type=int, default=1)

    parser.set_defaults(assign_noise=True)
    parser.add_argument("--assign_noise", dest="assign_noise", action="store_true")
    parser.add_argument("--no_assign_noise", dest="assign_noise", action="store_false")
    parser.add_argument("--noise_assign_max_dist", type=float, default=0.50)

    parser.add_argument("--max_window_gap", type=int, default=2)
    parser.add_argument("--merge_thresh", type=float, default=3.0)
    parser.add_argument("--w_p1", type=float, default=1.4)
    parser.add_argument("--w_p2", type=float, default=1.0)
    parser.add_argument("--w_p4", type=float, default=0.8)
    parser.add_argument("--w_p5", type=float, default=1.2)
    parser.add_argument("--w_pri", type=float, default=1.2)
    parser.add_argument("--w_phase", type=float, default=1.2)
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
    parser.add_argument("--min_valid_pri_us", type=float, default=1e-6)
    parser.add_argument("--missing_pri_penalty", type=float, default=0.5)
    parser.add_argument("--missing_phase_penalty", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10, help="Print progress every N windows. Use 0 to disable.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    df = read_pdw(args.input_file)
    if args.max_pulses > 0:
        df = df.iloc[: args.max_pulses].reset_index(drop=True)
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    print(f"Input PDW: {args.input_file}")
    print(f"Pulses:    {len(df)}")

    features = robust_scaled_features(df, args.feature_mode)
    _, window_groups = make_window_groups(toa, args.window_seconds)
    print(f"Windows:   {len(window_groups)} ({args.window_seconds:.3f}s)")

    tracklet_ids, tracklets, window_report = create_dbscan_tracklets(df, features, window_groups, args)
    roots, edge_report = merge_tracklets(tracklets, args)
    sigidx = roots_to_sigidx(tracklet_ids, roots)
    sigidx = dense_relabel(sigidx)
    write_sort_file(df, sigidx, args.output_file, emit_label99=args.emit_label99)

    args.window_report_csv.parent.mkdir(parents=True, exist_ok=True)
    window_report.to_csv(args.window_report_csv, index=False, encoding="utf-8-sig")
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
            "max_pulses": int(args.max_pulses),
            "feature_mode": str(args.feature_mode),
            "auto_eps": bool(args.auto_eps),
            "dbscan_eps": float(args.dbscan_eps),
            "eps_quantile": float(args.eps_quantile),
            "eps_scale": float(args.eps_scale),
            "dbscan_min_samples": int(args.dbscan_min_samples),
            "assign_noise": bool(args.assign_noise),
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
        print(f"Sort Acc    : {direct_metrics['sample_sort_acc']:.4f} ({direct_metrics['sample_sort_acc'] * 100:.2f}%)")
        print(
            f"Extra Batch : {direct_metrics['sample_extra_batch_rate']:.4f} "
            f"({direct_metrics['sample_extra_batch_rate'] * 100:.2f}%)"
        )
        print(
            f"Wrong Batch : {direct_metrics['sample_wrong_batch_rate']:.4f} "
            f"({direct_metrics['sample_wrong_batch_rate'] * 100:.2f}%)"
        )
        print("Recog Acc   : N/A (dbscan_sort.py does not predict LABEL)")
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

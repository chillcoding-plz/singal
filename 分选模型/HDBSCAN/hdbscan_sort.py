#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HDBSCAN window-tracklet PDW sorter.

This is a standalone alternative to dbscan_sort.py. The original DBSCAN files
are not modified.

Flow:
1. Split pulses into TOA windows.
2. Run HDBSCAN inside each window on robust-scaled PDW features.
3. Treat each HDBSCAN cluster as a short tracklet.
4. Merge tracklets across nearby windows with the existing PRI/phase graph.
5. Compute the same sorting metrics when truth is provided.

The script supports two real HDBSCAN backends:
    - external package: pip/conda package named "hdbscan"
    - sklearn.cluster.HDBSCAN in newer scikit-learn versions

Current local sklearn may not provide HDBSCAN. In that case, install hdbscan or
use --allow_optics_fallback for an OPTICS-based approximation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import dbscan_sort as dbscan


def _has_external_hdbscan() -> bool:
    return importlib.util.find_spec("hdbscan") is not None


def _get_sklearn_hdbscan():
    try:
        from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

        return SklearnHDBSCAN
    except Exception:
        return None


def resolve_backend(args: argparse.Namespace) -> str:
    backend = args.hdbscan_backend
    if backend == "external":
        if not _has_external_hdbscan():
            raise RuntimeError("External package 'hdbscan' is not installed.")
        return "external"
    if backend == "sklearn":
        if _get_sklearn_hdbscan() is None:
            raise RuntimeError("This sklearn version does not provide sklearn.cluster.HDBSCAN.")
        return "sklearn"
    if backend == "optics":
        return "optics"

    if _has_external_hdbscan():
        return "external"
    if _get_sklearn_hdbscan() is not None:
        return "sklearn"
    if args.allow_optics_fallback:
        return "optics"
    raise RuntimeError(
        "No HDBSCAN backend is available. Install package 'hdbscan', upgrade scikit-learn, "
        "or pass --allow_optics_fallback to run an OPTICS approximation."
    )


def fit_window_clusterer(features: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, object]]:
    backend = resolve_backend(args)
    min_samples = None if args.hdbscan_min_samples <= 0 else int(args.hdbscan_min_samples)
    info: Dict[str, object] = {
        "backend": backend,
        "min_cluster_size": int(args.hdbscan_min_cluster_size),
        "min_samples": int(args.hdbscan_min_samples),
    }

    if backend == "external":
        import hdbscan

        model = hdbscan.HDBSCAN(
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=min_samples,
            metric=args.hdbscan_metric,
            alpha=args.hdbscan_alpha,
            cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
            cluster_selection_method=args.hdbscan_cluster_selection_method,
            allow_single_cluster=args.hdbscan_allow_single_cluster,
            core_dist_n_jobs=args.n_jobs,
        )
        labels = model.fit_predict(features).astype(np.int64)
        probs = getattr(model, "probabilities_", None)
        info["mean_probability"] = float(np.mean(probs)) if probs is not None and len(probs) else float("nan")
        info["num_persistent_clusters"] = int(len(getattr(model, "cluster_persistence_", [])))
        return labels, info

    if backend == "sklearn":
        SklearnHDBSCAN = _get_sklearn_hdbscan()
        model = SklearnHDBSCAN(
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=min_samples,
            metric=args.hdbscan_metric,
            alpha=args.hdbscan_alpha,
            cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
            allow_single_cluster=args.hdbscan_allow_single_cluster,
            leaf_size=args.hdbscan_leaf_size,
            n_jobs=args.n_jobs,
        )
        labels = model.fit_predict(features).astype(np.int64)
        probs = getattr(model, "probabilities_", None)
        info["mean_probability"] = float(np.mean(probs)) if probs is not None and len(probs) else float("nan")
        return labels, info

    from sklearn.cluster import OPTICS

    optics_min_samples = min_samples if min_samples is not None else max(2, args.hdbscan_min_cluster_size // 2)
    model = OPTICS(
        min_samples=optics_min_samples,
        min_cluster_size=args.hdbscan_min_cluster_size,
        max_eps=args.optics_max_eps,
        metric=args.hdbscan_metric,
        cluster_method=args.optics_cluster_method,
        xi=args.optics_xi,
        leaf_size=args.hdbscan_leaf_size,
        n_jobs=args.n_jobs,
    )
    labels = model.fit_predict(features).astype(np.int64)
    info["optics_warning"] = "OPTICS fallback is not real HDBSCAN."
    return labels, info


def create_hdbscan_tracklets(
    df: pd.DataFrame,
    features: np.ndarray,
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
                    "num_clusters": 0,
                    "num_tracklets": 0,
                    "noise_pulses": n_window,
                    "small_cluster_pulses": 0,
                    "reason": "window_too_small",
                }
            )
            continue

        window_features = features[idx]
        labels, backend_info = fit_window_clusterer(window_features, args)

        unique_labels = [int(v) for v in np.unique(labels) if int(v) >= 0]
        counts: Dict[int, int] = {label: int(np.sum(labels == label)) for label in unique_labels}
        valid_clusters = [label for label, count in counts.items() if count >= args.min_tracklet_size]

        small_cluster_pulses = int(sum(count for label, count in counts.items() if label not in valid_clusters))
        noise_before = int(np.sum(labels < 0))
        if args.assign_noise and valid_clusters:
            labels = dbscan.assign_noise_to_clusters(labels, window_features, valid_clusters, args.noise_assign_max_dist)
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
                "num_clusters": int(len(unique_labels)),
                "num_tracklets": int(made),
                "noise_pulses": int(np.sum(labels < 0)),
                "noise_pulses_before_assign": noise_before,
                "small_cluster_pulses": small_cluster_pulses,
                "mean_probability": backend_info.get("mean_probability", np.nan),
                "reason": "",
            }
        )

        should_report = args.progress_every > 0 and (pos % args.progress_every == 0 or pos == total_windows)
        if args.verbose or should_report:
            elapsed = time.perf_counter() - start_time
            windows_per_sec = pos / max(elapsed, 1e-9)
            eta = (total_windows - pos) / max(windows_per_sec, 1e-9)
            print(
                f"[hdbscan-sort] windows {pos}/{total_windows} "
                f"({pos / max(total_windows, 1) * 100:.1f}%), "
                f"tracklets={len(tracklets)}, backend={backend_info.get('backend', '')}, "
                f"elapsed={elapsed / 60:.1f}min, eta={eta / 60:.1f}min",
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
    metrics["recognition_note"] = "N/A: hdbscan_sort.py does not predict LABEL."

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
    sort_batch_df.to_csv(metrics_dir / "hdbscan_sort_batch_eval.csv", index=False, encoding="utf-8-sig")
    sort_target_df.to_csv(metrics_dir / "hdbscan_sort_target_accuracy.csv", index=False, encoding="utf-8-sig")
    sort_target_beat_df.to_csv(metrics_dir / "hdbscan_sort_target_beat_eval.csv", index=False, encoding="utf-8-sig")
    sort_beat_df.to_csv(metrics_dir / "hdbscan_sort_beat_eval.csv", index=False, encoding="utf-8-sig")
    (metrics_dir / "hdbscan_sort_metrics.json").write_text(
        json.dumps(dbscan.json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dbscan.json_safe(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = dbscan.build_parser()
    parser.description = "HDBSCAN window-tracklet PDW sorter."
    parser.set_defaults(
        output_file=Path("./Sorted_PDW_pred_hdbscan.txt"),
        report_json=Path("./outputs_xgboost_pdw/hdbscan_sort_summary.json"),
        window_report_csv=Path("./outputs_xgboost_pdw/hdbscan_windows.csv"),
        edge_report_csv=Path("./outputs_xgboost_pdw/hdbscan_edges.csv"),
        metrics_dir=Path("./outputs_xgboost_pdw"),
    )
    parser.add_argument("--hdbscan_backend", choices=["auto", "external", "sklearn", "optics"], default="auto")
    parser.add_argument("--allow_optics_fallback", action="store_true")
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=20)
    parser.add_argument("--hdbscan_min_samples", type=int, default=8, help="Use <=0 to let HDBSCAN choose min_samples.")
    parser.add_argument("--hdbscan_metric", type=str, default="euclidean")
    parser.add_argument("--hdbscan_alpha", type=float, default=1.0)
    parser.add_argument("--hdbscan_cluster_selection_epsilon", type=float, default=0.0)
    parser.add_argument("--hdbscan_cluster_selection_method", choices=["eom", "leaf"], default="eom")
    parser.add_argument("--hdbscan_allow_single_cluster", action="store_true")
    parser.add_argument("--hdbscan_leaf_size", type=int, default=40)
    parser.add_argument("--optics_max_eps", type=float, default=np.inf)
    parser.add_argument("--optics_xi", type=float, default=0.05)
    parser.add_argument("--optics_cluster_method", choices=["xi", "dbscan"], default="xi")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    df = dbscan.read_pdw(args.input_file)
    if args.max_pulses > 0:
        df = df.iloc[: args.max_pulses].reset_index(drop=True)
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    print(f"Input PDW: {args.input_file}")
    print(f"Pulses:    {len(df)}")
    print(f"HDBSCAN backend: {resolve_backend(args)}")

    features = dbscan.robust_scaled_features(df, args.feature_mode)
    _, window_groups = dbscan.make_window_groups(toa, args.window_seconds)
    print(f"Windows:   {len(window_groups)} ({args.window_seconds:.3f}s)")

    tracklet_ids, tracklets, window_report = create_hdbscan_tracklets(df, features, window_groups, args)
    roots, edge_report = dbscan.merge_tracklets(tracklets, args)
    sigidx = dbscan.roots_to_sigidx(tracklet_ids, roots)
    sigidx = dbscan.dense_relabel(sigidx)

    if args.output_file.parent and str(args.output_file.parent) not in ("", "."):
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
    dbscan.write_sort_file(df, sigidx, args.output_file, emit_label99=args.emit_label99)

    args.window_report_csv.parent.mkdir(parents=True, exist_ok=True)
    window_report.to_csv(args.window_report_csv, index=False, encoding="utf-8-sig")
    edge_report.to_csv(args.edge_report_csv, index=False, encoding="utf-8-sig")

    summary = dbscan.summarize(sigidx)
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
            "hdbscan_backend": resolve_backend(args),
            "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
            "hdbscan_min_samples": int(args.hdbscan_min_samples),
            "hdbscan_cluster_selection_epsilon": float(args.hdbscan_cluster_selection_epsilon),
            "hdbscan_cluster_selection_method": str(args.hdbscan_cluster_selection_method),
            "assign_noise": bool(args.assign_noise),
            "max_window_gap": int(args.max_window_gap),
            "merge_thresh": float(args.merge_thresh),
        }
    )

    direct_metrics = None
    if not args.skip_metrics:
        if args.truth_file.exists():
            direct_metrics = compute_and_save_sort_metrics(args.truth_file, sigidx, args.metrics_dir, args)
            summary["direct_sort_metrics"] = direct_metrics
        else:
            summary["direct_sort_metrics"] = {"skipped": True, "reason": f"truth file not found: {args.truth_file}"}

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(dbscan.json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print("Summary:")
    print(json.dumps(dbscan.json_safe(summary), indent=2, ensure_ascii=False))
    if direct_metrics is not None:
        print("-" * 80)
        print("Direct sorting metrics, no XGBoost LABEL recognition:")
        print(f"Sort Acc    : {direct_metrics['sample_sort_acc']:.4f} ({direct_metrics['sample_sort_acc'] * 100:.2f}%)")
        print(f"Extra Batch : {direct_metrics['sample_extra_batch_rate']:.4f} ({direct_metrics['sample_extra_batch_rate'] * 100:.2f}%)")
        print(f"Wrong Batch : {direct_metrics['sample_wrong_batch_rate']:.4f} ({direct_metrics['sample_wrong_batch_rate'] * 100:.2f}%)")
        print("Recog Acc   : N/A (hdbscan_sort.py does not predict LABEL)")
        counts = direct_metrics.get("fail_reason_counts", {})
        rates = direct_metrics.get("fail_reason_rates_all", {})
        print(f"low_purity          : {counts.get('low_purity', 0)} ({rates.get('low_purity', 0.0) * 100:.2f}%)")
        print(f"low_target_fraction : {counts.get('low_target_fraction', 0)} ({rates.get('low_target_fraction', 0.0) * 100:.2f}%)")
    print(f"Saved sort file: {args.output_file}")


if __name__ == "__main__":
    main()

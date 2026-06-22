#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HDBSCAN sorter with conservative cross-window merging.

The window-level HDBSCAN clusters are unchanged. The innovation is in the
tracklet graph: only mutual-nearest edges are accepted by default, and each
union must keep the resulting component physically compact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import dbscan_sort as dbscan
try:
    import hdbscan_sort as hdbsort
except ImportError:
    from 分选 import hdbscan_sort as hdbsort


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return int(x)

    def union_roots(self, ra: int, rb: int) -> int:
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return int(ra)


def circular_abs_diff_deg(values: np.ndarray, ref: float) -> np.ndarray:
    return np.abs((values - ref + 180.0) % 360.0 - 180.0)


def circular_mean_deg(values: np.ndarray) -> float:
    radians = np.deg2rad(np.mod(values.astype(np.float64), 360.0))
    return float((np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) + 360.0) % 360.0)


def pri_phase_distances(a: dbscan.Tracklet, b: dbscan.Tracklet, args: argparse.Namespace) -> Tuple[float, float]:
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


def component_is_consistent(member_ids: List[int], tracklets: List[dbscan.Tracklet], args: argparse.Namespace) -> bool:
    if len(member_ids) <= 2:
        return True
    trs = [tracklets[i] for i in member_ids]
    p1 = np.array([tr.center_p1 for tr in trs], dtype=np.float64)
    p2 = np.array([tr.center_p2 for tr in trs], dtype=np.float64)
    p4 = np.array([tr.center_p4 for tr in trs], dtype=np.float64)
    p5 = np.array([tr.center_p5 for tr in trs], dtype=np.float64)

    if (np.max(p1) - np.min(p1)) / max(args.tol_p1, 1e-12) > args.cons_component_span:
        return False
    if (np.max(p2) - np.min(p2)) / max(args.tol_p2, 1e-12) > args.cons_component_span:
        return False
    if (np.max(p4) - np.min(p4)) / max(args.tol_p4, 1e-12) > args.cons_component_span:
        return False
    p5_center = circular_mean_deg(p5)
    if np.max(circular_abs_diff_deg(p5, p5_center)) / max(args.tol_p5_deg, 1e-12) > args.cons_component_angle_span:
        return False

    pri = np.array([tr.pri_us for tr in trs], dtype=np.float64)
    pri = pri[np.isfinite(pri) & (pri > args.min_valid_pri_us)]
    if len(pri) >= 2 and (np.max(pri) - np.min(pri)) / max(args.tol_pri_us, 1e-12) > args.cons_component_pri_span:
        return False
    return True


def enumerate_candidate_edges(tracklets: List[dbscan.Tracklet], args: argparse.Namespace) -> list[dict]:
    by_window: Dict[int, List[int]] = {}
    for tr in tracklets:
        by_window.setdefault(tr.window_id, []).append(tr.node_id)

    p1 = np.array([tr.center_p1 for tr in tracklets], dtype=np.float64)
    p2 = np.array([tr.center_p2 for tr in tracklets], dtype=np.float64)
    p4 = np.array([tr.center_p4 for tr in tracklets], dtype=np.float64)
    p5 = np.array([tr.center_p5 for tr in tracklets], dtype=np.float64)

    candidates: list[dict] = []
    for wi in sorted(by_window):
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
            hard_ok = (
                (d_p1 <= args.hard_gate_p1)
                & (d_p2 <= args.hard_gate_p2)
                & (d_p4 <= args.hard_gate_p4)
                & (d_p5 <= args.hard_gate_p5)
            )
            param_dist = args.w_p1 * d_p1 + args.w_p2 * d_p2 + args.w_p4 * d_p4 + args.w_p5 * d_p5
            for pos in np.flatnonzero(hard_ok):
                b_id = int(right[pos])
                d_pri, d_phase = pri_phase_distances(tracklets[a_id], tracklets[b_id], args)
                if d_pri > args.hard_gate_pri or d_phase > args.hard_gate_phase:
                    continue
                total_dist = float(param_dist[pos] + args.w_pri * d_pri + args.w_phase * d_phase)
                if total_dist > args.merge_thresh:
                    continue
                candidates.append(
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
    return candidates


def merge_tracklets_conservative(tracklets: List[dbscan.Tracklet], args: argparse.Namespace) -> Tuple[np.ndarray, pd.DataFrame]:
    if not tracklets:
        return np.zeros(0, dtype=np.int64), pd.DataFrame()

    candidates = enumerate_candidate_edges(tracklets, args)
    best_out: Dict[int, dict] = {}
    best_in: Dict[int, dict] = {}
    for edge in candidates:
        a = int(edge["a"])
        b = int(edge["b"])
        if a not in best_out or edge["total_dist"] < best_out[a]["total_dist"]:
            best_out[a] = edge
        if b not in best_in or edge["total_dist"] < best_in[b]["total_dist"]:
            best_in[b] = edge

    if args.cons_mutual_nearest:
        filtered = [
            edge
            for edge in candidates
            if best_out.get(int(edge["a"])) is edge and best_in.get(int(edge["b"])) is edge
        ]
    else:
        filtered = list(candidates)
    filtered.sort(key=lambda item: item["total_dist"])

    uf = UnionFind(len(tracklets))
    members: Dict[int, List[int]] = {i: [i] for i in range(len(tracklets))}
    accepted = []
    rejected_component = 0
    skipped_cycle = 0

    for edge in filtered:
        a = int(edge["a"])
        b = int(edge["b"])
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            skipped_cycle += 1
            continue
        combined = members[ra] + members[rb]
        if args.cons_check_component and not component_is_consistent(combined, tracklets, args):
            rejected_component += 1
            continue
        new_root = uf.union_roots(ra, rb)
        old_root = rb if new_root == ra else ra
        members[new_root] = combined
        members.pop(old_root, None)
        edge = dict(edge)
        edge["accepted"] = True
        accepted.append(edge)

    roots = np.array([uf.find(i) for i in range(len(tracklets))], dtype=np.int64)
    edge_df = pd.DataFrame(accepted)
    edge_df.attrs["candidate_edges"] = len(candidates)
    edge_df.attrs["filtered_edges"] = len(filtered)
    edge_df.attrs["rejected_component"] = rejected_component
    edge_df.attrs["skipped_cycle"] = skipped_cycle
    return roots, edge_df


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
    metrics["recognition_note"] = "N/A: hdbscan_conservative_merge_sort.py does not predict LABEL."
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
    sort_batch_df.to_csv(metrics_dir / "hdbscan_conservative_sort_batch_eval.csv", index=False, encoding="utf-8-sig")
    sort_target_df.to_csv(metrics_dir / "hdbscan_conservative_sort_target_accuracy.csv", index=False, encoding="utf-8-sig")
    sort_target_beat_df.to_csv(metrics_dir / "hdbscan_conservative_sort_target_beat_eval.csv", index=False, encoding="utf-8-sig")
    sort_beat_df.to_csv(metrics_dir / "hdbscan_conservative_sort_beat_eval.csv", index=False, encoding="utf-8-sig")
    (metrics_dir / "hdbscan_conservative_sort_metrics.json").write_text(
        json.dumps(dbscan.json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dbscan.json_safe(metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = hdbsort.build_parser()
    parser.description = "HDBSCAN sorter with conservative cross-window merging."
    parser.set_defaults(
        output_file=Path("./Sorted_PDW_pred_hdbscan_conservative.txt"),
        report_json=Path("./outputs_xgboost_pdw/hdbscan_conservative_sort_summary.json"),
        window_report_csv=Path("./outputs_xgboost_pdw/hdbscan_conservative_windows.csv"),
        edge_report_csv=Path("./outputs_xgboost_pdw/hdbscan_conservative_edges.csv"),
        metrics_dir=Path("./outputs_xgboost_pdw"),
    )
    parser.set_defaults(cons_mutual_nearest=True, cons_check_component=True)
    parser.add_argument("--cons_mutual_nearest", dest="cons_mutual_nearest", action="store_true")
    parser.add_argument("--no_cons_mutual_nearest", dest="cons_mutual_nearest", action="store_false")
    parser.add_argument("--cons_check_component", dest="cons_check_component", action="store_true")
    parser.add_argument("--no_cons_check_component", dest="cons_check_component", action="store_false")
    parser.add_argument("--cons_component_span", type=float, default=2.8)
    parser.add_argument("--cons_component_angle_span", type=float, default=2.8)
    parser.add_argument("--cons_component_pri_span", type=float, default=2.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = dbscan.read_pdw(args.input_file)
    if args.max_pulses > 0:
        df = df.iloc[: args.max_pulses].reset_index(drop=True)
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    backend = hdbsort.resolve_backend(args)
    print(f"Input PDW: {args.input_file}")
    print(f"Pulses:    {len(df)}")
    print(f"HDBSCAN backend: {backend}")

    features = dbscan.robust_scaled_features(df, args.feature_mode)
    _, window_groups = dbscan.make_window_groups(toa, args.window_seconds)
    print(f"Windows:   {len(window_groups)} ({args.window_seconds:.3f}s)")

    tracklet_ids, tracklets, window_report = hdbsort.create_hdbscan_tracklets(df, features, window_groups, args)
    print(f"[conservative] merging tracklets: {len(tracklets)}")
    roots, edge_report = merge_tracklets_conservative(tracklets, args)
    sigidx = dbscan.dense_relabel(dbscan.roots_to_sigidx(tracklet_ids, roots))

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
            "candidate_edges": int(edge_report.attrs.get("candidate_edges", 0)),
            "filtered_edges": int(edge_report.attrs.get("filtered_edges", 0)),
            "rejected_component_edges": int(edge_report.attrs.get("rejected_component", 0)),
            "skipped_cycle_edges": int(edge_report.attrs.get("skipped_cycle", 0)),
            "window_seconds": float(args.window_seconds),
            "feature_mode": str(args.feature_mode),
            "hdbscan_backend": backend,
            "hdbscan_min_cluster_size": int(args.hdbscan_min_cluster_size),
            "hdbscan_min_samples": int(args.hdbscan_min_samples),
            "cons_mutual_nearest": bool(args.cons_mutual_nearest),
            "cons_check_component": bool(args.cons_check_component),
            "cons_component_span": float(args.cons_component_span),
            "cons_component_pri_span": float(args.cons_component_pri_span),
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
        print(f"MR          : {direct_metrics['MR']:.4f} ({direct_metrics['MR'] * 100:.2f}%)")
        print(f"MP          : {direct_metrics['MP']:.4f} ({direct_metrics['MP'] * 100:.2f}%)")
        print(f"MIOU        : {direct_metrics['MIOU']:.4f} ({direct_metrics['MIOU'] * 100:.2f}%)")
        print(f"Extra Batch : {direct_metrics['sample_extra_batch_rate']:.4f} ({direct_metrics['sample_extra_batch_rate'] * 100:.2f}%)")
        print(f"Wrong Batch : {direct_metrics['sample_wrong_batch_rate']:.4f} ({direct_metrics['sample_wrong_batch_rate'] * 100:.2f}%)")
        counts = direct_metrics.get("fail_reason_counts", {})
        rates = direct_metrics.get("fail_reason_rates_all", {})
        print(f"low_purity          : {counts.get('low_purity', 0)} ({rates.get('low_purity', 0.0) * 100:.2f}%)")
        print(f"low_target_fraction : {counts.get('low_target_fraction', 0)} ({rates.get('low_target_fraction', 0.0) * 100:.2f}%)")
    print(f"Saved sort file: {args.output_file}")


if __name__ == "__main__":
    main()

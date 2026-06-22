#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PA-TGR-HDBSCAN sorter.

This experimental entry keeps the PA-HDBSCAN front end from
``pa_tsr_hdbscan_sort.py`` and replaces the TSR tail with a temporal-graph
refinement stage. The goal is to reduce fragmentation-driven extra batches and
beat-level SigIdx switching by explicitly linking beat-local prediction nodes
into physically consistent temporal identities.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import dbscan_sort as dbscan
import hdbscan_conservative_merge_sort as conservative
import pa_tsr_hdbscan_sort as base


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


def _circular_abs_diff_deg(left: float, right: float) -> float:
    return float(abs((float(left) - float(right) + 180.0) % 360.0 - 180.0))


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _weighted_circular_mean_deg(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(values) == 0:
        return float("nan")
    radians = np.deg2rad(np.mod(values, 360.0))
    sin_sum = float(np.sum(np.sin(radians) * weights))
    cos_sum = float(np.sum(np.cos(radians) * weights))
    if abs(sin_sum) < 1e-12 and abs(cos_sum) < 1e-12:
        return float(np.mod(values[0], 360.0))
    return float((np.rad2deg(np.arctan2(sin_sum, cos_sum)) + 360.0) % 360.0)


def _dense_positive_map(values: Sequence[int]) -> Dict[int, int]:
    uniq = sorted(int(v) for v in set(int(v) for v in values))
    return {old: new for new, old in enumerate(uniq, start=1)}


def _sigidx_summary(sigidx: np.ndarray) -> Dict[str, int]:
    positive = sigidx[sigidx > 0]
    return {
        "positive_sigidx_count": int(len(np.unique(positive))) if len(positive) else 0,
        "noise_or_unassigned_count": int(np.sum(sigidx <= 0)),
    }


def _inspect_hdbscan_runtime() -> Dict[str, object]:
    spec = importlib.util.find_spec("hdbscan")
    return {
        "python_executable": sys.executable,
        "external_hdbscan_visible": bool(spec is not None),
        "external_hdbscan_origin": getattr(spec, "origin", "") if spec is not None else "",
    }


def _build_beat_ids(toa: np.ndarray, chunk_seconds: float) -> Tuple[np.ndarray, float]:
    if len(toa) == 0:
        return np.zeros(0, dtype=np.int64), 0.0
    t0 = float(np.min(toa))
    if chunk_seconds <= 0:
        return np.zeros((len(toa),), dtype=np.int64), t0
    beat_ids = np.floor((toa.astype(np.float64) - t0) / float(chunk_seconds)).astype(np.int64)
    return beat_ids, t0


def build_temporal_node_report(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    chunk_seconds: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Summarize each positive SigIdx inside each beat into one graph node."""
    sigidx = sigidx.astype(np.int64, copy=False)
    toa = pdw["TOA(s)"].to_numpy(dtype=np.float64)
    beat_ids, _ = _build_beat_ids(toa, chunk_seconds)
    positive_idx = np.flatnonzero(sigidx > 0)
    columns = [
        "node_id",
        "beat",
        "source_sigidx",
        "count",
        "start_toa",
        "end_toa",
        "mid_toa",
        "Param1",
        "Param2",
        "Param4",
        "Param5",
        "pri_us",
        "pri_iqr_us",
    ]
    if len(positive_idx) == 0:
        return pd.DataFrame(columns=columns), beat_ids

    key_df = pd.DataFrame(
        {
            "beat": beat_ids[positive_idx],
            "source_sigidx": sigidx[positive_idx],
        }
    )
    grouped = key_df.groupby(["beat", "source_sigidx"], sort=True).indices
    p1 = pdw["Param1"].to_numpy(dtype=np.float64)
    p2 = pdw["Param2"].to_numpy(dtype=np.float64)
    p4 = pdw["Param4"].to_numpy(dtype=np.float64)
    p5 = pdw["Param5"].to_numpy(dtype=np.float64)

    rows = []
    for node_id, ((beat, source_sigidx), rel_pos) in enumerate(grouped.items(), start=1):
        idx = positive_idx[np.asarray(rel_pos, dtype=np.int64)]
        toa_group = toa[idx]
        pri_us, pri_iqr_us = dbscan.estimate_pri_us(toa_group)
        rows.append(
            {
                "node_id": int(node_id),
                "beat": int(beat),
                "source_sigidx": int(source_sigidx),
                "count": int(len(idx)),
                "start_toa": float(np.min(toa_group)),
                "end_toa": float(np.max(toa_group)),
                "mid_toa": float(0.5 * (np.min(toa_group) + np.max(toa_group))),
                "Param1": float(np.median(p1[idx])),
                "Param2": float(np.median(p2[idx])),
                "Param4": float(np.median(p4[idx])),
                "Param5": dbscan.circular_mean_deg(p5[idx]),
                "pri_us": float(pri_us),
                "pri_iqr_us": float(pri_iqr_us),
            }
        )

    return pd.DataFrame(rows, columns=columns), beat_ids


def build_temporal_edge_report(node_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Enumerate beat-local graph edges between physically compatible nodes."""
    if len(node_df) == 0:
        return pd.DataFrame()

    by_beat = {
        int(beat): np.asarray(pos, dtype=np.int64)
        for beat, pos in node_df.groupby("beat", sort=True).indices.items()
    }
    p1 = node_df["Param1"].to_numpy(dtype=np.float64)
    p2 = node_df["Param2"].to_numpy(dtype=np.float64)
    p4 = node_df["Param4"].to_numpy(dtype=np.float64)
    p5 = node_df["Param5"].to_numpy(dtype=np.float64)
    pri = node_df["pri_us"].to_numpy(dtype=np.float64)
    counts = node_df["count"].to_numpy(dtype=np.int64)
    beats = node_df["beat"].to_numpy(dtype=np.int64)
    source = node_df["source_sigidx"].to_numpy(dtype=np.int64)
    node_ids = node_df["node_id"].to_numpy(dtype=np.int64)

    rows = []
    for beat in sorted(by_beat):
        left = by_beat[beat]
        for gap in range(1, int(args.tgr_max_beat_gap) + 1):
            right = by_beat.get(int(beat + gap))
            if right is None or len(right) == 0:
                continue
            for left_pos in left:
                d_p1 = np.abs(p1[right] - p1[left_pos]) / max(float(args.tgr_tol_p1), 1e-12)
                d_p2 = np.abs(p2[right] - p2[left_pos]) / max(float(args.tgr_tol_p2), 1e-12)
                d_p4 = np.abs(p4[right] - p4[left_pos]) / max(float(args.tgr_tol_p4), 1e-12)
                d_p5 = circular_abs_diff_deg(p5[right], float(p5[left_pos])) / max(float(args.tgr_tol_p5_deg), 1e-12)

                pri_left = float(pri[left_pos])
                pri_right = pri[right]
                pri_valid = (
                    np.isfinite(pri_right)
                    & np.isfinite(pri_left)
                    & (pri_right > args.min_valid_pri_us)
                    & (pri_left > args.min_valid_pri_us)
                )
                d_pri = np.full(len(right), float(args.tgr_missing_pri_penalty), dtype=np.float64)
                d_pri[pri_valid] = np.abs(pri_right[pri_valid] - pri_left) / max(float(args.tgr_tol_pri_us), 1e-12)

                hard_ok = (
                    (d_p1 <= args.tgr_hard_gate)
                    & (d_p2 <= args.tgr_hard_gate)
                    & (d_p4 <= args.tgr_hard_gate)
                    & (d_p5 <= args.tgr_hard_gate)
                    & (d_pri <= args.tgr_hard_gate_pri)
                )
                if not np.any(hard_ok):
                    continue

                param_dist = (
                    args.tgr_w_p1 * d_p1
                    + args.tgr_w_p2 * d_p2
                    + args.tgr_w_p4 * d_p4
                    + args.tgr_w_p5 * d_p5
                    + args.tgr_w_pri * d_pri
                )
                gap_penalty = float(gap - 1) * float(args.tgr_gap_penalty)
                size_ratio = np.maximum(counts[left_pos], counts[right]) / np.maximum(
                    np.minimum(counts[left_pos], counts[right]),
                    1,
                )
                size_penalty = float(args.tgr_size_ratio_penalty) * np.maximum(size_ratio - 1.0, 0.0)
                same_sig_bonus = np.where(source[right] == source[left_pos], float(args.tgr_same_sig_bonus), 0.0)
                total_cost = param_dist + gap_penalty + size_penalty - same_sig_bonus
                take = hard_ok & (total_cost <= args.tgr_link_thresh)
                for right_pos in np.flatnonzero(take):
                    rows.append(
                        {
                            "node_a_pos": int(left_pos),
                            "node_b_pos": int(right[int(right_pos)]),
                            "node_a": int(node_ids[left_pos]),
                            "node_b": int(node_ids[int(right[right_pos])]),
                            "beat_a": int(beats[left_pos]),
                            "beat_b": int(beats[int(right[right_pos])]),
                            "beat_gap": int(gap),
                            "source_sigidx_a": int(source[left_pos]),
                            "source_sigidx_b": int(source[int(right[right_pos])]),
                            "count_a": int(counts[left_pos]),
                            "count_b": int(counts[int(right[right_pos])]),
                            "param_dist": float(param_dist[right_pos]),
                            "pri_dist": float(d_pri[right_pos]),
                            "size_ratio": float(size_ratio[right_pos]),
                            "same_sig": bool(source[int(right[right_pos])] == source[left_pos]),
                            "total_cost": float(total_cost[right_pos]),
                        }
                    )

    return pd.DataFrame(rows)


def temporal_component_is_consistent(
    node_df: pd.DataFrame,
    member_pos: List[int],
    args: argparse.Namespace,
) -> bool:
    if len(member_pos) <= 2:
        return True
    sub = node_df.iloc[member_pos]
    p1 = sub["Param1"].to_numpy(dtype=np.float64)
    p2 = sub["Param2"].to_numpy(dtype=np.float64)
    p4 = sub["Param4"].to_numpy(dtype=np.float64)
    p5 = sub["Param5"].to_numpy(dtype=np.float64)

    if (np.max(p1) - np.min(p1)) / max(float(args.tgr_tol_p1), 1e-12) > args.tgr_component_span:
        return False
    if (np.max(p2) - np.min(p2)) / max(float(args.tgr_tol_p2), 1e-12) > args.tgr_component_span:
        return False
    if (np.max(p4) - np.min(p4)) / max(float(args.tgr_tol_p4), 1e-12) > args.tgr_component_span:
        return False
    p5_center = dbscan.circular_mean_deg(p5)
    if np.max(circular_abs_diff_deg(p5, p5_center)) / max(float(args.tgr_tol_p5_deg), 1e-12) > args.tgr_component_angle_span:
        return False

    pri = sub["pri_us"].to_numpy(dtype=np.float64)
    pri = pri[np.isfinite(pri) & (pri > args.min_valid_pri_us)]
    if len(pri) >= 2 and (np.max(pri) - np.min(pri)) / max(float(args.tgr_tol_pri_us), 1e-12) > args.tgr_component_pri_span:
        return False
    return True


def link_temporal_nodes(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Link beat-level nodes into temporal identities."""
    if len(node_df) == 0:
        return node_df.copy(), {"enabled": True, "candidate_edges": 0, "accepted_edges": 0, "tracks_after_link": 0}, pd.DataFrame()

    node_out = node_df.reset_index(drop=True).copy()
    if len(edge_df) == 0:
        node_out["track_root"] = np.arange(len(node_out), dtype=np.int64)
        node_out["track_sigidx"] = np.arange(1, len(node_out) + 1, dtype=np.int64)
        summary = {
            "enabled": True,
            "candidate_edges": 0,
            "selected_edges": 0,
            "accepted_edges": 0,
            "tracks_after_link": int(len(node_out)),
            "reason": "no_candidate_edges",
        }
        return node_out, summary, pd.DataFrame()

    best_out: Dict[int, Dict[str, object]] = {}
    best_in: Dict[int, Dict[str, object]] = {}
    edge_records = [row._asdict() for row in edge_df.itertuples(index=False)]
    for record in edge_records:
        a = int(record["node_a_pos"])
        b = int(record["node_b_pos"])
        if a not in best_out or float(record["total_cost"]) < float(best_out[a]["total_cost"]):
            best_out[a] = record
        if b not in best_in or float(record["total_cost"]) < float(best_in[b]["total_cost"]):
            best_in[b] = record

    selected_rows = []
    if args.tgr_mutual_nearest:
        for record in edge_records:
            a = int(record["node_a_pos"])
            b = int(record["node_b_pos"])
            left = best_out.get(a)
            right = best_in.get(b)
            if left is None or right is None:
                continue
            if int(left["node_b_pos"]) == b and int(right["node_a_pos"]) == a:
                selected_rows.append(record)
    else:
        selected_rows = edge_records
    selected_df = pd.DataFrame(selected_rows).sort_values("total_cost", kind="mergesort").reset_index(drop=True) if selected_rows else pd.DataFrame(columns=edge_df.columns)

    uf = UnionFind(len(node_out))
    members: Dict[int, List[int]] = {i: [i] for i in range(len(node_out))}
    beat_sets: Dict[int, set[int]] = {
        i: {int(node_out.iloc[i]["beat"])}
        for i in range(len(node_out))
    }
    accepted_rows = []
    rejected_same_beat = 0
    rejected_component = 0
    skipped_cycle = 0

    for row in selected_df.itertuples(index=False):
        a = int(row.node_a_pos)
        b = int(row.node_b_pos)
        ra = uf.find(a)
        rb = uf.find(b)
        if ra == rb:
            skipped_cycle += 1
            continue
        if beat_sets[ra] & beat_sets[rb]:
            rejected_same_beat += 1
            continue
        combined_members = members[ra] + members[rb]
        if args.tgr_check_component and not temporal_component_is_consistent(node_out, combined_members, args):
            rejected_component += 1
            continue
        combined_beats = beat_sets[ra] | beat_sets[rb]
        new_root = uf.union_roots(ra, rb)
        old_root = rb if new_root == ra else ra
        members[new_root] = combined_members
        beat_sets[new_root] = combined_beats
        members.pop(old_root, None)
        beat_sets.pop(old_root, None)
        accepted_rows.append(row._asdict())

    track_root = np.array([uf.find(i) for i in range(len(node_out))], dtype=np.int64)
    dense_map = _dense_positive_map(track_root.tolist())
    node_out["track_root"] = track_root
    node_out["track_sigidx"] = np.array([dense_map[int(v)] for v in track_root], dtype=np.int64)
    summary = {
        "enabled": True,
        "candidate_edges": int(len(edge_df)),
        "selected_edges": int(len(selected_df)),
        "accepted_edges": int(len(accepted_rows)),
        "rejected_same_beat": int(rejected_same_beat),
        "rejected_component": int(rejected_component),
        "skipped_cycle": int(skipped_cycle),
        "tracks_after_link": int(node_out["track_sigidx"].nunique()) if len(node_out) else 0,
        "mutual_nearest": bool(args.tgr_mutual_nearest),
    }
    return node_out, summary, pd.DataFrame(accepted_rows)


def build_track_report_from_nodes(
    node_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, set[int]], Dict[int, set[int]]]:
    if len(node_df) == 0 or "track_sigidx" not in node_df.columns:
        return pd.DataFrame(), {}, {}

    rows = []
    beat_sets: Dict[int, set[int]] = {}
    source_sets: Dict[int, set[int]] = {}
    for track_sigidx, group in node_df.groupby("track_sigidx", sort=True):
        weights = group["count"].to_numpy(dtype=np.float64)
        pri_vals = group["pri_us"].to_numpy(dtype=np.float64)
        pri_weights = weights[np.isfinite(pri_vals) & (pri_vals > 0)]
        pri_valid = pri_vals[np.isfinite(pri_vals) & (pri_vals > 0)]
        rows.append(
            {
                "track_sigidx": int(track_sigidx),
                "total_pulses": int(group["count"].sum()),
                "num_beats": int(group["beat"].nunique()),
                "first_beat": int(group["beat"].min()),
                "last_beat": int(group["beat"].max()),
                "start_toa": float(group["start_toa"].min()),
                "end_toa": float(group["end_toa"].max()),
                "source_sigidx_count": int(group["source_sigidx"].nunique()),
                "source_sigidxs": ",".join(str(int(v)) for v in sorted(group["source_sigidx"].unique())),
                "Param1": _weighted_average(group["Param1"].to_numpy(dtype=np.float64), weights),
                "Param2": _weighted_average(group["Param2"].to_numpy(dtype=np.float64), weights),
                "Param4": _weighted_average(group["Param4"].to_numpy(dtype=np.float64), weights),
                "Param5": _weighted_circular_mean_deg(group["Param5"].to_numpy(dtype=np.float64), weights),
                "pri_us": _weighted_average(pri_valid, pri_weights) if len(pri_valid) else float("nan"),
            }
        )
        beat_sets[int(track_sigidx)] = set(int(v) for v in group["beat"].tolist())
        source_sets[int(track_sigidx)] = set(int(v) for v in group["source_sigidx"].tolist())
    return pd.DataFrame(rows), beat_sets, source_sets


def build_source_report_from_nodes(
    node_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[int, set[int]], Dict[int, set[int]], Dict[int, set[int]]]:
    if len(node_df) == 0:
        return pd.DataFrame(), {}, {}, {}

    rows = []
    beat_sets: Dict[int, set[int]] = {}
    track_sets: Dict[int, set[int]] = {}
    member_sets: Dict[int, set[int]] = {}
    has_track_sigidx = "track_sigidx" in node_df.columns

    for source_sigidx, group in node_df.groupby("source_sigidx", sort=True):
        weights = group["count"].to_numpy(dtype=np.float64)
        pri_vals = group["pri_us"].to_numpy(dtype=np.float64)
        pri_valid = np.isfinite(pri_vals) & (pri_vals > 0)
        rows.append(
            {
                "source_sigidx": int(source_sigidx),
                "total_pulses": int(group["count"].sum()),
                "num_beats": int(group["beat"].nunique()),
                "num_tracks": int(group["track_sigidx"].nunique()) if has_track_sigidx else 0,
                "first_beat": int(group["beat"].min()),
                "last_beat": int(group["beat"].max()),
                "Param1": _weighted_average(group["Param1"].to_numpy(dtype=np.float64), weights),
                "Param2": _weighted_average(group["Param2"].to_numpy(dtype=np.float64), weights),
                "Param4": _weighted_average(group["Param4"].to_numpy(dtype=np.float64), weights),
                "Param5": _weighted_circular_mean_deg(group["Param5"].to_numpy(dtype=np.float64), weights),
                "pri_us": _weighted_average(pri_vals[pri_valid], weights[pri_valid]) if np.any(pri_valid) else float("nan"),
            }
        )
        source_id = int(source_sigidx)
        beat_sets[source_id] = set(int(v) for v in group["beat"].tolist())
        track_sets[source_id] = set(int(v) for v in group["track_sigidx"].tolist()) if has_track_sigidx else set()
        member_sets[source_id] = {source_id}

    return pd.DataFrame(rows), beat_sets, track_sets, member_sets


def _find_parent(parent: Dict[int, int], x: int) -> int:
    path = []
    while parent[x] != x:
        path.append(x)
        x = int(parent[x])
    for item in path:
        parent[item] = x
    return int(x)


def _track_beat_gap(left: Dict[str, object], right: Dict[str, object]) -> int:
    left_first = int(left["first_beat"])
    left_last = int(left["last_beat"])
    right_first = int(right["first_beat"])
    right_last = int(right["last_beat"])
    if left_last < right_first:
        return max(right_first - left_last - 1, 0)
    if right_last < left_first:
        return max(left_first - right_last - 1, 0)
    return 0


def absorb_fragment_tracks(
    node_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Absorb weak temporal tracks into stronger compatible tracks."""
    if len(node_df) == 0 or "track_sigidx" not in node_df.columns:
        empty_summary = {
            "enabled": True,
            "merged_tracks": 0,
            "tracks_after_absorb": 0,
            "candidate_tracks": 0,
            "merge_candidates": 0,
        }
        return node_df.copy(), empty_summary, pd.DataFrame(), pd.DataFrame()

    track_report, beat_sets, source_sets = build_track_report_from_nodes(node_df)
    if len(track_report) == 0:
        empty_summary = {
            "enabled": True,
            "merged_tracks": 0,
            "tracks_after_absorb": 0,
            "candidate_tracks": 0,
            "merge_candidates": 0,
        }
        return node_df.copy(), empty_summary, pd.DataFrame(), pd.DataFrame()

    track_lookup = {
        int(row.track_sigidx): {
            "track_sigidx": int(row.track_sigidx),
            "total_pulses": int(row.total_pulses),
            "num_beats": int(row.num_beats),
            "first_beat": int(row.first_beat),
            "last_beat": int(row.last_beat),
            "source_sigidx_count": int(row.source_sigidx_count),
            "Param1": float(row.Param1),
            "Param2": float(row.Param2),
            "Param4": float(row.Param4),
            "Param5": float(row.Param5),
            "pri_us": float(row.pri_us),
        }
        for row in track_report.itertuples(index=False)
    }
    parent = {track_id: track_id for track_id in track_lookup}
    rows = []

    def merge_track_states(target_id: int, source_id: int) -> None:
        target = track_lookup[target_id]
        source = track_lookup[source_id]
        target_weight = float(target["total_pulses"])
        source_weight = float(source["total_pulses"])
        weights = np.asarray([target_weight, source_weight], dtype=np.float64)
        pri_values = np.asarray([float(target["pri_us"]), float(source["pri_us"])], dtype=np.float64)
        pri_valid = np.isfinite(pri_values) & (pri_values > args.min_valid_pri_us)

        merged_sources = source_sets[target_id] | source_sets[source_id]
        merged_beats = beat_sets[target_id] | beat_sets[source_id]
        source_sets[target_id] = merged_sources
        beat_sets[target_id] = merged_beats

        target["total_pulses"] = int(target["total_pulses"]) + int(source["total_pulses"])
        target["num_beats"] = int(len(merged_beats))
        target["first_beat"] = int(min(int(target["first_beat"]), int(source["first_beat"])))
        target["last_beat"] = int(max(int(target["last_beat"]), int(source["last_beat"])))
        target["source_sigidx_count"] = int(len(merged_sources))
        target["Param1"] = _weighted_average(
            np.asarray([float(target["Param1"]), float(source["Param1"])], dtype=np.float64),
            weights,
        )
        target["Param2"] = _weighted_average(
            np.asarray([float(target["Param2"]), float(source["Param2"])], dtype=np.float64),
            weights,
        )
        target["Param4"] = _weighted_average(
            np.asarray([float(target["Param4"]), float(source["Param4"])], dtype=np.float64),
            weights,
        )
        target["Param5"] = _weighted_circular_mean_deg(
            np.asarray([float(target["Param5"]), float(source["Param5"])], dtype=np.float64),
            weights,
        )
        target["pri_us"] = (
            _weighted_average(pri_values[pri_valid], weights[pri_valid])
            if np.any(pri_valid)
            else float("nan")
        )

    candidate_ids = [
        int(row.track_sigidx)
        for row in track_report.itertuples(index=False)
        if int(row.total_pulses) < args.tgr_min_track_pulses and int(row.num_beats) <= args.tgr_min_track_beats
    ]
    candidate_ids.sort(key=lambda tid: (track_lookup[tid]["num_beats"], track_lookup[tid]["total_pulses"], tid))
    rejected_overlap = 0
    rejected_gap = 0
    rejected_source_growth = 0
    rejected_union_sources = 0
    rejected_target_complexity = 0
    rejected_hard_gate = 0

    for source_id in candidate_ids:
        source_root = _find_parent(parent, int(source_id))
        source = track_lookup[source_root]
        if source["total_pulses"] >= args.tgr_min_track_pulses or source["num_beats"] > args.tgr_min_track_beats:
            continue

        best_target = None
        best_cost = float("inf")
        for target_id in track_lookup:
            target_root = _find_parent(parent, int(target_id))
            if target_root == source_root:
                continue
            target = track_lookup[target_root]
            if target["total_pulses"] < source["total_pulses"] and target["num_beats"] < source["num_beats"]:
                continue

            overlap_beats = len(beat_sets[source_root] & beat_sets[target_root])
            if overlap_beats > args.tgr_absorb_max_overlap_beats:
                rejected_overlap += 1
                continue

            beat_gap = _track_beat_gap(source, target)
            if beat_gap > args.tgr_absorb_max_beat_gap:
                rejected_gap += 1
                continue

            target_source_count = int(target["source_sigidx_count"])
            if target_source_count > args.tgr_absorb_max_target_sources:
                rejected_target_complexity += 1
                continue

            union_source_count = len(source_sets[source_root] | source_sets[target_root])
            if union_source_count > args.tgr_absorb_max_union_sources:
                rejected_union_sources += 1
                continue
            if union_source_count - target_source_count > args.tgr_absorb_max_source_growth:
                rejected_source_growth += 1
                continue

            d_p1 = abs(float(source["Param1"]) - float(target["Param1"])) / max(float(args.tgr_tol_p1), 1e-12)
            d_p2 = abs(float(source["Param2"]) - float(target["Param2"])) / max(float(args.tgr_tol_p2), 1e-12)
            d_p4 = abs(float(source["Param4"]) - float(target["Param4"])) / max(float(args.tgr_tol_p4), 1e-12)
            d_p5 = abs((float(source["Param5"]) - float(target["Param5"]) + 180.0) % 360.0 - 180.0) / max(float(args.tgr_tol_p5_deg), 1e-12)
            pri_left = float(source["pri_us"])
            pri_right = float(target["pri_us"])
            pri_valid = np.isfinite(pri_left) and np.isfinite(pri_right) and pri_left > args.min_valid_pri_us and pri_right > args.min_valid_pri_us
            d_pri = (
                abs(pri_left - pri_right) / max(float(args.tgr_tol_pri_us), 1e-12)
                if pri_valid
                else float(args.tgr_missing_pri_penalty)
            )
            hard_ok = (
                d_p1 <= args.tgr_absorb_hard_gate
                and d_p2 <= args.tgr_absorb_hard_gate
                and d_p4 <= args.tgr_absorb_hard_gate
                and d_p5 <= args.tgr_absorb_hard_gate
                and d_pri <= args.tgr_absorb_hard_gate_pri
            )
            if not hard_ok:
                rejected_hard_gate += 1
                continue

            same_source_bonus = float(args.tgr_absorb_same_source_bonus) if source_sets[source_root] & source_sets[target_root] else 0.0
            cost = (
                args.tgr_w_p1 * d_p1
                + args.tgr_w_p2 * d_p2
                + args.tgr_w_p4 * d_p4
                + args.tgr_w_p5 * d_p5
                + args.tgr_w_pri * d_pri
                + float(args.tgr_absorb_gap_penalty) * float(beat_gap)
                + float(args.tgr_absorb_overlap_penalty) * float(overlap_beats)
                - same_source_bonus
            )
            if cost < best_cost and cost <= args.tgr_absorb_thresh:
                best_cost = float(cost)
                best_target = int(target_root)

        if best_target is not None:
            union_source_count = int(len(source_sets[source_root] | source_sets[best_target]))
            parent[source_root] = int(best_target)
            rows.append(
                {
                    "source_track_sigidx": int(source_root),
                    "target_track_sigidx": int(best_target),
                    "source_total_pulses": int(source["total_pulses"]),
                    "source_num_beats": int(source["num_beats"]),
                    "target_total_pulses": int(track_lookup[best_target]["total_pulses"]),
                    "target_num_beats": int(track_lookup[best_target]["num_beats"]),
                    "cost": float(best_cost),
                    "overlap_beats": int(len(beat_sets[source_root] & beat_sets[best_target])),
                    "beat_gap": int(_track_beat_gap(source, track_lookup[best_target])),
                    "source_source_sigidx_count": int(source["source_sigidx_count"]),
                    "target_source_sigidx_count": int(track_lookup[best_target]["source_sigidx_count"]),
                    "union_source_sigidx_count": int(union_source_count),
                }
            )
            merge_track_states(int(best_target), int(source_root))

    mapped = np.array([_find_parent(parent, int(v)) for v in node_df["track_sigidx"].to_numpy(dtype=np.int64)], dtype=np.int64)
    dense_map = _dense_positive_map(mapped.tolist())
    node_out = node_df.copy()
    node_out["track_sigidx_before_absorb"] = node_out["track_sigidx"].astype(np.int64)
    node_out["track_sigidx"] = np.array([dense_map[int(v)] for v in mapped], dtype=np.int64)
    final_track_report, _, _ = build_track_report_from_nodes(node_out)
    summary = {
        "enabled": True,
        "candidate_tracks": int(len(track_report)),
        "merge_candidates": int(len(candidate_ids)),
        "merged_tracks": int(len(rows)),
        "tracks_after_absorb": int(len(final_track_report)),
        "rejected_overlap": int(rejected_overlap),
        "rejected_gap": int(rejected_gap),
        "rejected_source_growth": int(rejected_source_growth),
        "rejected_union_sources": int(rejected_union_sources),
        "rejected_target_complexity": int(rejected_target_complexity),
        "rejected_hard_gate": int(rejected_hard_gate),
    }
    return node_out, summary, pd.DataFrame(rows), final_track_report


def merge_sources_by_temporal_tracks(
    node_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    """Merge only weak baseline source SigIdx values with strong temporal evidence."""
    if len(node_df) == 0 or "track_sigidx" not in node_df.columns:
        empty_summary = {
            "enabled": True,
            "candidate_sources": 0,
            "merge_candidates": 0,
            "merged_sources": 0,
            "positive_sigidx_after_merge": 0,
        }
        return node_df.copy(), empty_summary, pd.DataFrame()

    source_report, beat_sets, track_sets, member_sets = build_source_report_from_nodes(node_df)
    if len(source_report) == 0:
        empty_summary = {
            "enabled": True,
            "candidate_sources": 0,
            "merge_candidates": 0,
            "merged_sources": 0,
            "positive_sigidx_after_merge": 0,
        }
        node_out = node_df.copy()
        node_out["merged_source_sigidx"] = node_out["source_sigidx"].astype(np.int64)
        return node_out, empty_summary, pd.DataFrame()

    source_lookup = {
        int(row.source_sigidx): {
            "source_sigidx": int(row.source_sigidx),
            "total_pulses": int(row.total_pulses),
            "num_beats": int(row.num_beats),
            "num_tracks": int(row.num_tracks),
            "first_beat": int(row.first_beat),
            "last_beat": int(row.last_beat),
            "Param1": float(row.Param1),
            "Param2": float(row.Param2),
            "Param4": float(row.Param4),
            "Param5": float(row.Param5),
            "pri_us": float(row.pri_us),
            "member_source_count": 1,
        }
        for row in source_report.itertuples(index=False)
    }

    track_source = (
        node_df.groupby(["track_sigidx", "source_sigidx"], sort=True)
        .agg(track_source_pulses=("count", "sum"), track_source_beats=("beat", "nunique"))
        .reset_index()
    )
    track_total = track_source.groupby("track_sigidx", sort=True)["track_source_pulses"].sum().rename("track_total")
    track_source = track_source.join(track_total, on="track_sigidx")
    track_source["track_fraction"] = track_source["track_source_pulses"] / np.maximum(track_source["track_total"], 1)

    source_track = track_source.merge(
        source_report[["source_sigidx", "total_pulses", "num_beats", "num_tracks"]],
        on="source_sigidx",
        how="left",
    )
    source_track["source_fraction"] = source_track["track_source_pulses"] / np.maximum(source_track["total_pulses"], 1)

    track_groups = {
        int(track_sigidx): group.sort_values("track_source_pulses", ascending=False).reset_index(drop=True)
        for track_sigidx, group in track_source.groupby("track_sigidx", sort=True)
    }
    source_groups = {
        int(source_sigidx): group.sort_values("track_source_pulses", ascending=False).reset_index(drop=True)
        for source_sigidx, group in source_track.groupby("source_sigidx", sort=True)
    }

    def compute_source_metrics(left: Dict[str, object], right: Dict[str, object]) -> Dict[str, float]:
        d_p1 = abs(float(left["Param1"]) - float(right["Param1"])) / max(float(args.tgr_tol_p1), 1e-12)
        d_p2 = abs(float(left["Param2"]) - float(right["Param2"])) / max(float(args.tgr_tol_p2), 1e-12)
        d_p4 = abs(float(left["Param4"]) - float(right["Param4"])) / max(float(args.tgr_tol_p4), 1e-12)
        d_p5 = _circular_abs_diff_deg(float(left["Param5"]), float(right["Param5"])) / max(float(args.tgr_tol_p5_deg), 1e-12)
        pri_left = float(left["pri_us"])
        pri_right = float(right["pri_us"])
        pri_valid = (
            np.isfinite(pri_left)
            and np.isfinite(pri_right)
            and pri_left > args.min_valid_pri_us
            and pri_right > args.min_valid_pri_us
        )
        d_pri = (
            abs(pri_left - pri_right) / max(float(args.tgr_tol_pri_us), 1e-12)
            if pri_valid
            else float(args.tgr_missing_pri_penalty)
        )
        return {
            "d_p1": float(d_p1),
            "d_p2": float(d_p2),
            "d_p4": float(d_p4),
            "d_p5": float(d_p5),
            "d_pri": float(d_pri),
        }

    candidate_rows = []
    candidate_sources = 0
    rejected_dominant = 0
    rejected_anchor = 0
    rejected_overlap = 0
    rejected_hard_gate = 0

    for source_id, source in sorted(source_lookup.items()):
        if source["total_pulses"] > args.tgr_source_merge_max_pulses:
            continue
        if source["num_beats"] > args.tgr_source_merge_max_beats:
            continue
        if source["num_tracks"] > args.tgr_source_merge_max_tracks:
            continue
        src_tracks = source_groups.get(int(source_id))
        if src_tracks is None or len(src_tracks) == 0:
            continue
        candidate_sources += 1
        dominant = src_tracks.iloc[0]
        dominant_track_sigidx = int(dominant["track_sigidx"])
        dominant_track_fraction = float(dominant["source_fraction"])
        source_track_pulses = int(dominant["track_source_pulses"])
        if dominant_track_fraction < args.tgr_source_merge_min_dominant_frac:
            rejected_dominant += 1
            continue

        best_record = None
        best_cost = float("inf")
        for peer in track_groups.get(dominant_track_sigidx, pd.DataFrame()).itertuples(index=False):
            anchor_id = int(peer.source_sigidx)
            if anchor_id == source_id:
                continue
            anchor = source_lookup[anchor_id]
            anchor_track_pulses = int(peer.track_source_pulses)
            anchor_track_fraction = float(peer.track_fraction)
            if anchor["total_pulses"] < source["total_pulses"] or anchor_track_pulses < source_track_pulses:
                rejected_anchor += 1
                continue
            if (
                anchor_track_pulses < args.tgr_source_merge_anchor_min_pulses
                and anchor_track_fraction < args.tgr_source_merge_anchor_track_frac_min
            ):
                rejected_anchor += 1
                continue

            overlap_beats = len(beat_sets[source_id] & beat_sets[anchor_id])
            overlap_ratio = overlap_beats / max(min(source["num_beats"], anchor["num_beats"]), 1)
            if overlap_beats > args.tgr_source_merge_max_overlap_beats or overlap_ratio > args.tgr_source_merge_max_overlap_ratio:
                rejected_overlap += 1
                continue

            metrics = compute_source_metrics(source, anchor)
            hard_ok = (
                metrics["d_p1"] <= args.tgr_source_merge_hard_gate
                and metrics["d_p2"] <= args.tgr_source_merge_hard_gate
                and metrics["d_p4"] <= args.tgr_source_merge_hard_gate
                and metrics["d_p5"] <= args.tgr_source_merge_hard_gate
                and metrics["d_pri"] <= args.tgr_source_merge_hard_gate_pri
            )
            if not hard_ok:
                rejected_hard_gate += 1
                continue

            cost = (
                args.tgr_w_p1 * metrics["d_p1"]
                + args.tgr_w_p2 * metrics["d_p2"]
                + args.tgr_w_p4 * metrics["d_p4"]
                + args.tgr_w_p5 * metrics["d_p5"]
                + args.tgr_w_pri * metrics["d_pri"]
                + float(args.tgr_source_merge_overlap_penalty) * float(overlap_beats)
                + float(args.tgr_source_merge_track_spread_penalty) * max(int(source["num_tracks"]) - 1, 0)
            )
            if cost > args.tgr_source_merge_thresh:
                continue
            if cost < best_cost:
                best_cost = float(cost)
                best_record = {
                    "source_sigidx": int(source_id),
                    "anchor_source_sigidx": int(anchor_id),
                    "source_total_pulses": int(source["total_pulses"]),
                    "source_num_beats": int(source["num_beats"]),
                    "source_num_tracks": int(source["num_tracks"]),
                    "anchor_total_pulses": int(anchor["total_pulses"]),
                    "anchor_num_beats": int(anchor["num_beats"]),
                    "anchor_num_tracks": int(anchor["num_tracks"]),
                    "dominant_track_sigidx": int(dominant_track_sigidx),
                    "source_track_pulses": int(source_track_pulses),
                    "anchor_track_pulses": int(anchor_track_pulses),
                    "dominant_track_fraction": float(dominant_track_fraction),
                    "anchor_track_fraction": float(anchor_track_fraction),
                    "overlap_beats": int(overlap_beats),
                    "overlap_ratio": float(overlap_ratio),
                    "d_p1": float(metrics["d_p1"]),
                    "d_p2": float(metrics["d_p2"]),
                    "d_p4": float(metrics["d_p4"]),
                    "d_p5": float(metrics["d_p5"]),
                    "d_pri": float(metrics["d_pri"]),
                    "cost": float(cost),
                }

        if best_record is not None:
            candidate_rows.append(best_record)

    candidate_df = (
        pd.DataFrame(candidate_rows)
        .sort_values(["cost", "dominant_track_fraction", "source_total_pulses"], ascending=[True, False, True], kind="mergesort")
        .reset_index(drop=True)
        if candidate_rows
        else pd.DataFrame()
    )

    parent = {source_id: source_id for source_id in source_lookup}
    group_lookup = {source_id: dict(stats) for source_id, stats in source_lookup.items()}

    def merge_source_states(target_id: int, source_id: int) -> None:
        target = group_lookup[target_id]
        source = group_lookup[source_id]
        target_weight = float(target["total_pulses"])
        source_weight = float(source["total_pulses"])
        weights = np.asarray([target_weight, source_weight], dtype=np.float64)
        pri_values = np.asarray([float(target["pri_us"]), float(source["pri_us"])], dtype=np.float64)
        pri_valid = np.isfinite(pri_values) & (pri_values > args.min_valid_pri_us)

        merged_beats = beat_sets[target_id] | beat_sets[source_id]
        merged_tracks = track_sets[target_id] | track_sets[source_id]
        merged_members = member_sets[target_id] | member_sets[source_id]
        beat_sets[target_id] = merged_beats
        track_sets[target_id] = merged_tracks
        member_sets[target_id] = merged_members

        target["total_pulses"] = int(target["total_pulses"]) + int(source["total_pulses"])
        target["num_beats"] = int(len(merged_beats))
        target["num_tracks"] = int(len(merged_tracks))
        target["first_beat"] = int(min(int(target["first_beat"]), int(source["first_beat"])))
        target["last_beat"] = int(max(int(target["last_beat"]), int(source["last_beat"])))
        target["member_source_count"] = int(len(merged_members))
        target["Param1"] = _weighted_average(
            np.asarray([float(target["Param1"]), float(source["Param1"])], dtype=np.float64),
            weights,
        )
        target["Param2"] = _weighted_average(
            np.asarray([float(target["Param2"]), float(source["Param2"])], dtype=np.float64),
            weights,
        )
        target["Param4"] = _weighted_average(
            np.asarray([float(target["Param4"]), float(source["Param4"])], dtype=np.float64),
            weights,
        )
        target["Param5"] = _weighted_circular_mean_deg(
            np.asarray([float(target["Param5"]), float(source["Param5"])], dtype=np.float64),
            weights,
        )
        target["pri_us"] = (
            _weighted_average(pri_values[pri_valid], weights[pri_valid])
            if np.any(pri_valid)
            else float("nan")
        )

    accepted_rows = []
    rejected_group_overlap = 0
    rejected_group_size = 0
    rejected_group_hard_gate = 0
    rejected_group_cost = 0

    for row in candidate_df.itertuples(index=False):
        source_root = _find_parent(parent, int(row.source_sigidx))
        anchor_root = _find_parent(parent, int(row.anchor_source_sigidx))
        if source_root == anchor_root:
            continue
        if source_root != int(row.source_sigidx):
            continue

        source = group_lookup[source_root]
        anchor = group_lookup[anchor_root]
        if anchor["total_pulses"] < source["total_pulses"]:
            continue
        if len(member_sets[source_root] | member_sets[anchor_root]) > args.tgr_source_merge_group_max_members:
            rejected_group_size += 1
            continue

        overlap_beats = len(beat_sets[source_root] & beat_sets[anchor_root])
        overlap_ratio = overlap_beats / max(min(source["num_beats"], anchor["num_beats"]), 1)
        if overlap_beats > args.tgr_source_merge_max_overlap_beats or overlap_ratio > args.tgr_source_merge_max_overlap_ratio:
            rejected_group_overlap += 1
            continue

        metrics = compute_source_metrics(source, anchor)
        hard_ok = (
            metrics["d_p1"] <= args.tgr_source_merge_hard_gate
            and metrics["d_p2"] <= args.tgr_source_merge_hard_gate
            and metrics["d_p4"] <= args.tgr_source_merge_hard_gate
            and metrics["d_p5"] <= args.tgr_source_merge_hard_gate
            and metrics["d_pri"] <= args.tgr_source_merge_hard_gate_pri
        )
        if not hard_ok:
            rejected_group_hard_gate += 1
            continue

        cost = (
            args.tgr_w_p1 * metrics["d_p1"]
            + args.tgr_w_p2 * metrics["d_p2"]
            + args.tgr_w_p4 * metrics["d_p4"]
            + args.tgr_w_p5 * metrics["d_p5"]
            + args.tgr_w_pri * metrics["d_pri"]
            + float(args.tgr_source_merge_overlap_penalty) * float(overlap_beats)
            + float(args.tgr_source_merge_track_spread_penalty) * max(int(source["num_tracks"]) - 1, 0)
        )
        if cost > args.tgr_source_merge_thresh:
            rejected_group_cost += 1
            continue

        parent[source_root] = int(anchor_root)
        accepted_rows.append(
            {
                "source_sigidx": int(source_root),
                "anchor_source_sigidx": int(anchor_root),
                "source_total_pulses": int(source["total_pulses"]),
                "source_num_beats": int(source["num_beats"]),
                "source_num_tracks": int(source["num_tracks"]),
                "anchor_total_pulses": int(anchor["total_pulses"]),
                "anchor_num_beats": int(anchor["num_beats"]),
                "anchor_num_tracks": int(anchor["num_tracks"]),
                "merged_member_source_count": int(len(member_sets[source_root] | member_sets[anchor_root])),
                "overlap_beats": int(overlap_beats),
                "overlap_ratio": float(overlap_ratio),
                "d_p1": float(metrics["d_p1"]),
                "d_p2": float(metrics["d_p2"]),
                "d_p4": float(metrics["d_p4"]),
                "d_p5": float(metrics["d_p5"]),
                "d_pri": float(metrics["d_pri"]),
                "cost": float(cost),
            }
        )
        merge_source_states(int(anchor_root), int(source_root))

    source_map = {source_id: _find_parent(parent, int(source_id)) for source_id in source_lookup}
    node_out = node_df.copy()
    node_out["merged_source_sigidx"] = node_out["source_sigidx"].map(source_map).astype(np.int64)
    summary = {
        "enabled": True,
        "candidate_sources": int(candidate_sources),
        "merge_candidates": int(len(candidate_df)),
        "merged_sources": int(len(accepted_rows)),
        "positive_sigidx_after_merge": int(len(set(source_map.values()))),
        "rejected_dominant": int(rejected_dominant),
        "rejected_anchor": int(rejected_anchor),
        "rejected_overlap": int(rejected_overlap),
        "rejected_hard_gate": int(rejected_hard_gate),
        "rejected_group_overlap": int(rejected_group_overlap),
        "rejected_group_size": int(rejected_group_size),
        "rejected_group_hard_gate": int(rejected_group_hard_gate),
        "rejected_group_cost": int(rejected_group_cost),
    }
    return node_out, summary, pd.DataFrame(accepted_rows)


def assign_merged_source_sigidx(sigidx: np.ndarray, node_df: pd.DataFrame) -> np.ndarray:
    """Relabel pulses using a conservative source-level merge map."""
    out = sigidx.astype(np.int64).copy()
    if len(node_df) == 0 or "merged_source_sigidx" not in node_df.columns:
        return out

    mapping = (
        node_df[["source_sigidx", "merged_source_sigidx"]]
        .drop_duplicates()
        .set_index("source_sigidx")["merged_source_sigidx"]
    )
    positive_mask = out > 0
    positive_values = out[positive_mask]
    mapped = mapping.reindex(positive_values).to_numpy()
    valid = ~pd.isna(mapped)
    if np.any(valid):
        positive_idx = np.flatnonzero(positive_mask)
        out[positive_idx[valid]] = np.asarray(mapped[valid], dtype=np.int64)
    return out.astype(np.int64)


def assign_temporal_track_sigidx(
    sigidx: np.ndarray,
    beat_ids: np.ndarray,
    node_df: pd.DataFrame,
) -> np.ndarray:
    """Relabel pulses by the temporal track assigned to their beat-local node."""
    out = sigidx.astype(np.int64).copy()
    if len(node_df) == 0:
        return out

    positive_mask = out > 0
    if not np.any(positive_mask):
        return out

    max_sig = max(int(np.max(out[positive_mask])), int(node_df["source_sigidx"].max()))
    key_base = max_sig + 1
    pulse_keys = beat_ids[positive_mask].astype(np.int64) * key_base + out[positive_mask].astype(np.int64)
    node_keys = (
        node_df["beat"].to_numpy(dtype=np.int64) * key_base
        + node_df["source_sigidx"].to_numpy(dtype=np.int64)
    )
    mapping = pd.Series(node_df["track_sigidx"].to_numpy(dtype=np.int64), index=node_keys)
    mapped = mapping.reindex(pulse_keys).to_numpy()
    valid = ~pd.isna(mapped)
    positive_idx = np.flatnonzero(positive_mask)
    if np.any(valid):
        out[positive_idx[valid]] = np.asarray(mapped[valid], dtype=np.int64)
    return out.astype(np.int64)


def suppress_tail_batches_after_rescue(
    sigidx: np.ndarray,
    beat_ids: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Map tiny short-lived SigIdx batches back to noise before a final rescue pass."""
    out = sigidx.astype(np.int64).copy()
    if not bool(getattr(args, "tgr_tail_cleanup", False)):
        return out, {"enabled": False}

    min_pulses = int(max(getattr(args, "tgr_tail_min_pulses", 0), 0))
    max_beats = int(getattr(args, "tgr_tail_max_beats", -1))
    summary: Dict[str, object] = {
        "enabled": True,
        "min_pulses": int(min_pulses),
        "max_beats": int(max_beats),
        "candidate_batches": 0,
        "suppressed_batches": 0,
        "suppressed_pulses": 0,
        "positive_sigidx_after_suppress": int(len(np.unique(out[out > 0]))) if np.any(out > 0) else 0,
    }
    if min_pulses <= 0 or max_beats < 0:
        summary["reason"] = "non_positive_threshold"
        return out, summary

    positive_mask = out > 0
    if not np.any(positive_mask):
        summary["reason"] = "no_positive_sigidx"
        return out, summary

    work = pd.DataFrame(
        {
            "sigidx": out[positive_mask].astype(np.int64),
            "beat": beat_ids[positive_mask].astype(np.int64),
        }
    )
    stats = (
        work.groupby("sigidx", sort=True)
        .agg(total_pulses=("sigidx", "size"), num_beats=("beat", "nunique"))
        .reset_index()
    )
    summary["candidate_batches"] = int(len(stats))
    tail_mask = (
        (stats["total_pulses"] < int(min_pulses))
        & (stats["num_beats"] <= int(max_beats))
    )
    tail_ids = stats.loc[tail_mask, "sigidx"].to_numpy(dtype=np.int64)
    summary["suppressed_batches"] = int(len(tail_ids))
    summary["suppressed_pulses"] = int(stats.loc[tail_mask, "total_pulses"].sum()) if len(stats) else 0
    if len(tail_ids) == 0:
        summary["reason"] = "no_tail_batches"
        return out, summary

    out[np.isin(out, tail_ids)] = 0
    summary["positive_sigidx_after_suppress"] = int(len(np.unique(out[out > 0]))) if np.any(out > 0) else 0
    return out.astype(np.int64), summary


def suppress_thin_persistent_batches(
    sigidx: np.ndarray,
    beat_ids: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Suppress long-lived but thin SigIdx batches that mainly inflate extra-batch counts."""
    out = sigidx.astype(np.int64).copy()
    if not bool(getattr(args, "tgr_thin_cleanup", False)):
        return out, {"enabled": False}

    max_total_pulses = int(max(getattr(args, "tgr_thin_max_total_pulses", 0), 0))
    max_avg_per_beat = float(max(getattr(args, "tgr_thin_max_avg_pulses_per_beat", 0.0), 0.0))
    min_beats = int(max(getattr(args, "tgr_thin_min_beats", 0), 0))
    summary: Dict[str, object] = {
        "enabled": True,
        "max_total_pulses": int(max_total_pulses),
        "max_avg_pulses_per_beat": float(max_avg_per_beat),
        "min_beats": int(min_beats),
        "candidate_batches": 0,
        "suppressed_batches": 0,
        "suppressed_pulses": 0,
        "positive_sigidx_after_suppress": int(len(np.unique(out[out > 0]))) if np.any(out > 0) else 0,
    }
    if max_total_pulses <= 0 or max_avg_per_beat <= 0 or min_beats <= 0:
        summary["reason"] = "non_positive_threshold"
        return out, summary

    positive_mask = out > 0
    if not np.any(positive_mask):
        summary["reason"] = "no_positive_sigidx"
        return out, summary

    work = pd.DataFrame(
        {
            "sigidx": out[positive_mask].astype(np.int64),
            "beat": beat_ids[positive_mask].astype(np.int64),
        }
    )
    stats = (
        work.groupby("sigidx", sort=True)
        .agg(total_pulses=("sigidx", "size"), num_beats=("beat", "nunique"))
        .reset_index()
    )
    if len(stats) == 0:
        summary["reason"] = "no_grouped_batches"
        return out, summary

    stats["avg_pulses_per_beat"] = stats["total_pulses"] / np.maximum(stats["num_beats"], 1)
    summary["candidate_batches"] = int(len(stats))
    thin_mask = (
        (stats["total_pulses"] < int(max_total_pulses))
        & (stats["avg_pulses_per_beat"] < float(max_avg_per_beat))
        & (stats["num_beats"] >= int(min_beats))
    )
    thin_ids = stats.loc[thin_mask, "sigidx"].to_numpy(dtype=np.int64)
    summary["suppressed_batches"] = int(len(thin_ids))
    summary["suppressed_pulses"] = int(stats.loc[thin_mask, "total_pulses"].sum())
    if len(thin_ids) == 0:
        summary["reason"] = "no_thin_batches"
        return out, summary

    out[np.isin(out, thin_ids)] = 0
    summary["positive_sigidx_after_suppress"] = int(len(np.unique(out[out > 0]))) if np.any(out > 0) else 0
    return out.astype(np.int64), summary


def apply_temporal_graph_refinement(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply temporal-graph refinement on top of raw PA-HDBSCAN output."""
    out = sigidx.astype(np.int64).copy()
    summary: Dict[str, object] = {
        "mode": "tgr_merge_only",
        **_sigidx_summary(out),
    }
    band_report = pd.DataFrame()

    if args.tgr_pre_reduce:
        out, reduce_summary, _ = base.reducer.reduce_sigidx_fixed(
            pdw=pdw,
            sigidx=out,
            min_cluster_size=args.tsr_reduce_min_cluster_size,
            merge_thresh=args.tsr_reduce_merge_thresh,
            min_batch_fraction=args.tsr_reduce_min_batch_fraction,
            weights=(args.tsr_reduce_w_p1, args.tsr_reduce_w_p2, args.tsr_reduce_w_p4, args.tsr_reduce_w_p5),
            tolerances=(args.tsr_reduce_tol_p1, args.tsr_reduce_tol_p2, args.tsr_reduce_tol_p4, args.tsr_reduce_tol_p5_deg),
            hard_gates=(args.tsr_reduce_hard_gate,) * 4,
            dense_relabel_output=False,
            split_large_batches=False,
        )
        summary["tgr_pre_reduce"] = reduce_summary
    else:
        summary["tgr_pre_reduce"] = {"enabled": False}

    band_requested = bool(base._parse_band_split_groups(args)) or bool(getattr(args, "tsr_band_split_auto", False))
    if band_requested:
        out = dbscan.dense_relabel(out)
    out, band_summary, band_report = base.tsr_param_band_split(pdw, out, args)
    summary["tgr_band_init"] = band_summary

    node_report, beat_ids = build_temporal_node_report(pdw, out, args.tgr_chunk_seconds)
    linked_nodes, link_summary, edge_report = link_temporal_nodes(node_report, build_temporal_edge_report(node_report, args), args)
    summary["tgr_link"] = link_summary

    absorbed_nodes, absorb_summary, absorb_report, track_report = absorb_fragment_tracks(linked_nodes, args)
    summary["tgr_absorb"] = absorb_summary

    merged_nodes, source_merge_summary, source_merge_report = merge_sources_by_temporal_tracks(absorbed_nodes, args)
    summary["tgr_source_merge"] = source_merge_summary

    out = assign_merged_source_sigidx(out, merged_nodes)
    out = dbscan.dense_relabel(out)
    summary["after_source_merge"] = _sigidx_summary(out)

    if args.tgr_noise_rescue:
        out, rescue_summary = base.tsr_rescue_noise_by_physics(pdw, out, args)
        summary["tgr_rescue_noise"] = rescue_summary
    else:
        summary["tgr_rescue_noise"] = {"enabled": False}
    out = dbscan.dense_relabel(out)
    summary["after_primary_rescue_dense_relabel"] = _sigidx_summary(out)

    eval_beat_ids, _ = _build_beat_ids(pdw["TOA(s)"].to_numpy(dtype=np.float64), float(args.sort_chunk_seconds))
    out, tail_cleanup_summary = suppress_tail_batches_after_rescue(out, eval_beat_ids, args)
    summary["tgr_tail_cleanup"] = tail_cleanup_summary

    reran_tail_rescue = (
        bool(args.tgr_noise_rescue)
        and int(tail_cleanup_summary.get("suppressed_batches", 0)) > 0
    )
    if reran_tail_rescue:
        out, tail_rescue_summary = base.tsr_rescue_noise_by_physics(pdw, out, args)
        tail_rescue_summary = dict(tail_rescue_summary)
        tail_rescue_summary["reran"] = True
        summary["tgr_tail_post_rescue"] = tail_rescue_summary
    else:
        reason = "tail_cleanup_disabled_or_empty"
        if not bool(args.tgr_noise_rescue):
            reason = "primary_noise_rescue_disabled"
        summary["tgr_tail_post_rescue"] = {
            "enabled": bool(args.tgr_noise_rescue),
            "reran": False,
            "reason": reason,
        }
    out = dbscan.dense_relabel(out)
    summary["after_tail_cleanup_dense_relabel"] = _sigidx_summary(out)

    out, thin_cleanup_summary = suppress_thin_persistent_batches(out, eval_beat_ids, args)
    summary["tgr_thin_cleanup"] = thin_cleanup_summary
    reran_thin_rescue = (
        bool(args.tgr_noise_rescue)
        and int(thin_cleanup_summary.get("suppressed_batches", 0)) > 0
    )
    if reran_thin_rescue:
        out, thin_rescue_summary = base.tsr_rescue_noise_by_physics(pdw, out, args)
        thin_rescue_summary = dict(thin_rescue_summary)
        thin_rescue_summary["reran"] = True
        summary["tgr_thin_post_rescue"] = thin_rescue_summary
    else:
        reason = "thin_cleanup_disabled_or_empty"
        if not bool(args.tgr_noise_rescue):
            reason = "primary_noise_rescue_disabled"
        summary["tgr_thin_post_rescue"] = {
            "enabled": bool(args.tgr_noise_rescue),
            "reran": False,
            "reason": reason,
        }
    out = dbscan.dense_relabel(out)
    summary["after_final_dense_relabel"] = _sigidx_summary(out)
    return out.astype(np.int64), summary, merged_nodes, edge_report, track_report, absorb_report, source_merge_report


def run_pa_tgr_sort(args: argparse.Namespace) -> Dict[str, object]:
    """Run the complete PA-TGR-HDBSCAN sorter and save reports."""
    df = dbscan.read_pdw(args.input_file)
    if args.max_pulses > 0:
        df = df.iloc[: args.max_pulses].reset_index(drop=True)
    toa = df["TOA(s)"].to_numpy(dtype=np.float64)
    runtime_info = _inspect_hdbscan_runtime()
    backend = base.hdbsort.resolve_backend(args)

    print(f"Input PDW: {args.input_file}")
    print(f"Pulses:    {len(df)}")
    print(f"Python:    {runtime_info['python_executable']}")
    if runtime_info["external_hdbscan_visible"]:
        print(f"hdbscan pkg: {runtime_info['external_hdbscan_origin']}")
    else:
        print("hdbscan pkg: not visible in current interpreter")
    print(f"HDBSCAN backend: {backend}")
    print(f"PA distance: {args.pa_distance_mode}")

    _, window_groups = dbscan.make_window_groups(toa, args.window_seconds)
    print(f"Windows:   {len(window_groups)} ({args.window_seconds:.3f}s)")
    if args.use_existing_raw_output:
        raw_table = pd.read_csv(args.raw_output_file, sep=r"\s+")
        if "SigIdx" not in raw_table.columns:
            raise ValueError(f"raw output file has no SigIdx column: {args.raw_output_file}")
        raw_sigidx = raw_table["SigIdx"].to_numpy(dtype=np.int64)
        if len(raw_sigidx) != len(df):
            raise ValueError(f"raw output length mismatch: {len(raw_sigidx)} vs input pulses {len(df)}")
        print(f"Reusing raw sort: {args.raw_output_file}")
        feature_info = {
            "pa_profile": str(args.pa_profile),
            "pa_distance_mode": str(args.pa_distance_mode),
            "reused_raw_output": True,
        }
        tracklets = []
        window_report = pd.DataFrame()
        edge_report = pd.DataFrame()
    else:
        pa_features, pa_context, feature_info = base.build_pa_features(df, args)
        print(f"Features:  {pa_features.shape[1]} PA dims ({args.pa_profile})")

        tracklet_ids, tracklets, window_report = base.create_pa_hdbscan_tracklets(df, pa_features, pa_context, window_groups, args)
        print(f"[pa-tgr] merging tracklets: {len(tracklets)} via {args.pa_initial_merge}")
        if args.pa_initial_merge == "conservative":
            roots, edge_report = conservative.merge_tracklets_conservative(tracklets, args)
        else:
            roots, edge_report = dbscan.merge_tracklets(tracklets, args)

        raw_sigidx = dbscan.dense_relabel(dbscan.roots_to_sigidx(tracklet_ids, roots))
        base._ensure_parent(args.raw_output_file)
        dbscan.write_sort_file(df, raw_sigidx, args.raw_output_file, emit_label99=args.emit_label99)

    final_sigidx, tgr_summary, node_report, temporal_edge_report, track_report, absorb_report, source_merge_report = apply_temporal_graph_refinement(df, raw_sigidx, args)
    base._ensure_parent(args.output_file)
    dbscan.write_sort_file(df, final_sigidx, args.output_file, emit_label99=args.emit_label99)

    base._ensure_parent(args.window_report_csv)
    base._ensure_parent(args.edge_report_csv)
    base._ensure_parent(args.tgr_node_report_csv)
    base._ensure_parent(args.tgr_edge_report_csv)
    base._ensure_parent(args.tgr_track_report_csv)
    base._ensure_parent(args.tgr_absorb_report_csv)
    base._ensure_parent(args.tgr_source_merge_report_csv)
    window_report.to_csv(args.window_report_csv, index=False, encoding="utf-8-sig")
    edge_report.to_csv(args.edge_report_csv, index=False, encoding="utf-8-sig")
    if len(node_report) > 0:
        node_report.to_csv(args.tgr_node_report_csv, index=False, encoding="utf-8-sig")
    if len(temporal_edge_report) > 0:
        temporal_edge_report.to_csv(args.tgr_edge_report_csv, index=False, encoding="utf-8-sig")
    if len(track_report) > 0:
        track_report.to_csv(args.tgr_track_report_csv, index=False, encoding="utf-8-sig")
    if len(absorb_report) > 0:
        absorb_report.to_csv(args.tgr_absorb_report_csv, index=False, encoding="utf-8-sig")
    if len(source_merge_report) > 0:
        source_merge_report.to_csv(args.tgr_source_merge_report_csv, index=False, encoding="utf-8-sig")

    metrics = None
    if not args.skip_metrics:
        if args.truth_file.exists():
            metrics = base.compute_and_save_sort_metrics(args.truth_file, final_sigidx, args.metrics_dir, args, prefix="pa_tgr_hdbscan")
        else:
            metrics = {"skipped": True, "reason": f"truth file not found: {args.truth_file}"}

    summary = {
        "method": "PA-TGR-HDBSCAN",
        "input_file": str(args.input_file),
        "raw_output_file": str(args.raw_output_file),
        "final_output_file": str(args.output_file),
        "frontend_reused": bool(args.use_existing_raw_output),
        "total_pulses": int(len(df)),
        "num_windows": int(len(window_groups)),
        "num_tracklets": int(len(tracklets)),
        "hdbscan_backend": backend,
        "hdbscan_runtime": runtime_info,
        "window_seconds": float(args.window_seconds),
        "candidate_edges": int(edge_report.attrs.get("candidate_edges", len(edge_report))),
        "filtered_edges": int(edge_report.attrs.get("filtered_edges", len(edge_report))),
        "accepted_edges": int(len(edge_report)),
        "raw_sort_summary": _sigidx_summary(raw_sigidx),
        "final_sort_summary": _sigidx_summary(final_sigidx),
        "pa_features": feature_info,
        "tgr": tgr_summary,
        "metrics": metrics,
    }
    base._ensure_parent(args.report_json)
    args.report_json.write_text(json.dumps(base._safe_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return base._safe_json(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = "End-to-end PA-TGR-HDBSCAN sorter: PDW -> HDBSCAN -> temporal graph refinement."
    parser.set_defaults(
        output_file=Path("./Sorted_PDW_pred_pa_tgr_hdbscan.txt"),
        raw_output_file=Path("./outputs_pa_tgr/raw_pa_hdbscan.txt"),
        report_json=Path("./outputs_pa_tgr/pa_tgr_hdbscan_summary.json"),
        metrics_dir=Path("./outputs_pa_tgr"),
        window_report_csv=Path("./outputs_pa_tgr/pa_hdbscan_windows.csv"),
        edge_report_csv=Path("./outputs_pa_tgr/pa_hdbscan_edges.csv"),
    )

    parser.set_defaults(tgr_pre_reduce=True, tgr_noise_rescue=True, tgr_mutual_nearest=True, tgr_check_component=True)
    parser.add_argument("--use_existing_raw_output", action="store_true", help="Skip the PA-HDBSCAN front-end and reuse raw_output_file as the baseline SigIdx input.")
    parser.add_argument("--no_tgr_pre_reduce", dest="tgr_pre_reduce", action="store_false", help="Disable physical pre-reduction before temporal graph linking.")
    parser.add_argument("--no_tgr_noise_rescue", dest="tgr_noise_rescue", action="store_false", help="Disable the final noise rescue stage.")
    parser.add_argument("--no_tgr_mutual_nearest", dest="tgr_mutual_nearest", action="store_false", help="Allow non-mutual temporal graph edges.")
    parser.add_argument("--no_tgr_check_component", dest="tgr_check_component", action="store_false", help="Disable component compactness checks during temporal linking.")

    parser.add_argument("--tgr_chunk_seconds", type=float, default=0.2, help="Beat duration used to build temporal graph nodes.")
    parser.add_argument("--tgr_max_beat_gap", type=int, default=1, help="Maximum beat gap when linking temporal graph nodes.")
    parser.add_argument("--tgr_link_thresh", type=float, default=2.6, help="Maximum temporal edge cost accepted for node linking.")
    parser.add_argument("--tgr_hard_gate", type=float, default=1.5, help="Per-feature hard gate for temporal node linking.")
    parser.add_argument("--tgr_hard_gate_pri", type=float, default=2.0, help="PRI hard gate for temporal node linking.")
    parser.add_argument("--tgr_tol_p1", type=float, default=150.0, help="Temporal graph tolerance for Param1.")
    parser.add_argument("--tgr_tol_p2", type=float, default=0.2, help="Temporal graph tolerance for Param2.")
    parser.add_argument("--tgr_tol_p4", type=float, default=10.0, help="Temporal graph tolerance for Param4.")
    parser.add_argument("--tgr_tol_p5_deg", type=float, default=20.0, help="Temporal graph tolerance for Param5.")
    parser.add_argument("--tgr_tol_pri_us", type=float, default=40.0, help="Temporal graph tolerance for PRI.")
    parser.add_argument("--tgr_w_p1", type=float, default=1.4, help="Temporal graph weight for Param1.")
    parser.add_argument("--tgr_w_p2", type=float, default=1.0, help="Temporal graph weight for Param2.")
    parser.add_argument("--tgr_w_p4", type=float, default=0.8, help="Temporal graph weight for Param4.")
    parser.add_argument("--tgr_w_p5", type=float, default=1.2, help="Temporal graph weight for Param5.")
    parser.add_argument("--tgr_w_pri", type=float, default=1.0, help="Temporal graph weight for PRI.")
    parser.add_argument("--tgr_gap_penalty", type=float, default=0.35, help="Penalty per missing beat gap in temporal linking.")
    parser.add_argument("--tgr_same_sig_bonus", type=float, default=0.35, help="Bonus when two temporal nodes come from the same source SigIdx.")
    parser.add_argument("--tgr_size_ratio_penalty", type=float, default=0.05, help="Penalty for linking nodes with very different pulse counts.")
    parser.add_argument("--tgr_missing_pri_penalty", type=float, default=0.5, help="PRI penalty used when one side has no stable PRI estimate.")
    parser.add_argument("--tgr_component_span", type=float, default=3.0, help="Maximum normalized Param1/2/4 span of a linked temporal component.")
    parser.add_argument("--tgr_component_angle_span", type=float, default=3.0, help="Maximum normalized Param5 span of a linked temporal component.")
    parser.add_argument("--tgr_component_pri_span", type=float, default=2.5, help="Maximum normalized PRI span of a linked temporal component.")

    parser.add_argument("--tgr_min_track_pulses", type=int, default=1200, help="Tracks smaller than this are candidates for fragment absorption.")
    parser.add_argument("--tgr_min_track_beats", type=int, default=2, help="Tracks with at most this many beats are candidates for fragment absorption.")
    parser.add_argument("--tgr_absorb_thresh", type=float, default=2.0, help="Maximum cost when absorbing a weak temporal track into a stronger one.")
    parser.add_argument("--tgr_absorb_hard_gate", type=float, default=1.5, help="Per-feature hard gate for track absorption.")
    parser.add_argument("--tgr_absorb_hard_gate_pri", type=float, default=2.0, help="PRI hard gate for track absorption.")
    parser.add_argument("--tgr_absorb_gap_penalty", type=float, default=0.75, help="Penalty per beat gap during track absorption.")
    parser.add_argument("--tgr_absorb_overlap_penalty", type=float, default=0.25, help="Penalty per overlapping beat during track absorption.")
    parser.add_argument("--tgr_absorb_same_source_bonus", type=float, default=0.40, help="Bonus when weak and strong tracks share source SigIdx ancestry.")
    parser.add_argument("--tgr_absorb_max_overlap_beats", type=int, default=0, help="Maximum allowed beat overlap when absorbing fragment tracks.")
    parser.add_argument("--tgr_absorb_max_beat_gap", type=int, default=1, help="Maximum allowed empty beat gap when absorbing fragment tracks.")
    parser.add_argument("--tgr_absorb_max_target_sources", type=int, default=4, help="Do not absorb into already-complex targets with more than this many source SigIdx values.")
    parser.add_argument("--tgr_absorb_max_union_sources", type=int, default=6, help="Maximum number of distinct source SigIdx values allowed after one absorption.")
    parser.add_argument("--tgr_absorb_max_source_growth", type=int, default=1, help="Maximum increase in source SigIdx count contributed by one absorbed fragment.")
    parser.add_argument("--tgr_source_merge_max_pulses", type=int, default=600, help="Only source SigIdx values with at most this many pulses are eligible for conservative source merging.")
    parser.add_argument("--tgr_source_merge_max_beats", type=int, default=3, help="Only source SigIdx values with at most this many beats are eligible for conservative source merging.")
    parser.add_argument("--tgr_source_merge_max_tracks", type=int, default=3, help="Only source SigIdx values spread across at most this many temporal tracks are eligible for conservative source merging.")
    parser.add_argument("--tgr_source_merge_min_dominant_frac", type=float, default=0.85, help="A source SigIdx must place at least this fraction of its pulses into one temporal track before it can be merged.")
    parser.add_argument("--tgr_source_merge_thresh", type=float, default=2.0, help="Maximum cost accepted when merging a weak source SigIdx into a stronger anchor source.")
    parser.add_argument("--tgr_source_merge_hard_gate", type=float, default=1.0, help="Per-feature hard gate used for conservative source merging.")
    parser.add_argument("--tgr_source_merge_hard_gate_pri", type=float, default=1.5, help="PRI hard gate used for conservative source merging.")
    parser.add_argument("--tgr_source_merge_max_overlap_beats", type=int, default=1, help="Maximum allowed beat overlap between two source groups during conservative source merging.")
    parser.add_argument("--tgr_source_merge_max_overlap_ratio", type=float, default=0.35, help="Maximum allowed normalized beat overlap between two source groups during conservative source merging.")
    parser.add_argument("--tgr_source_merge_anchor_min_pulses", type=int, default=200, help="An anchor source must have at least this many pulses inside the dominant temporal track unless it also dominates that track by fraction.")
    parser.add_argument("--tgr_source_merge_anchor_track_frac_min", type=float, default=0.40, help="Minimum within-track fraction for a smaller anchor when conservative source merging is considered.")
    parser.add_argument("--tgr_source_merge_group_max_members", type=int, default=6, help="Maximum number of original source SigIdx values allowed inside one merged source group.")
    parser.add_argument("--tgr_source_merge_overlap_penalty", type=float, default=0.30, help="Penalty per overlapping beat during conservative source merging.")
    parser.add_argument("--tgr_source_merge_track_spread_penalty", type=float, default=0.12, help="Penalty for source SigIdx values already spread across multiple temporal tracks.")
    parser.set_defaults(tgr_tail_cleanup=True)
    parser.add_argument("--no_tgr_tail_cleanup", dest="tgr_tail_cleanup", action="store_false", help="Disable the post-rescue cleanup that maps tiny short-lived SigIdx batches back to noise before one last rescue pass.")
    parser.add_argument("--tgr_tail_min_pulses", type=int, default=1000, help="After the primary rescue, suppress positive SigIdx batches smaller than this pulse count when they are also short-lived in beat space.")
    parser.add_argument("--tgr_tail_max_beats", type=int, default=20, help="Maximum evaluation-beat span allowed for a tiny tail batch to be suppressed before the final rescue rerun.")
    parser.set_defaults(tgr_thin_cleanup=True)
    parser.add_argument("--no_tgr_thin_cleanup", dest="tgr_thin_cleanup", action="store_false", help="Disable the persistent thin-batch cleanup that targets long-lived low-strength extra batches after tail cleanup.")
    parser.add_argument("--tgr_thin_max_total_pulses", type=int, default=1800, help="Suppress long-lived batches below this total pulse count when they also remain too thin per active beat.")
    parser.add_argument("--tgr_thin_max_avg_pulses_per_beat", type=float, default=140.0, help="Maximum average pulses per active beat allowed for the persistent thin-batch cleanup.")
    parser.add_argument("--tgr_thin_min_beats", type=int, default=10, help="Only batches active on at least this many evaluation beats are considered by the persistent thin-batch cleanup.")

    parser.add_argument("--tgr_node_report_csv", type=Path, default=Path("./outputs_pa_tgr/pa_tgr_nodes.csv"))
    parser.add_argument("--tgr_edge_report_csv", type=Path, default=Path("./outputs_pa_tgr/pa_tgr_edges.csv"))
    parser.add_argument("--tgr_track_report_csv", type=Path, default=Path("./outputs_pa_tgr/pa_tgr_tracks.csv"))
    parser.add_argument("--tgr_absorb_report_csv", type=Path, default=Path("./outputs_pa_tgr/pa_tgr_absorb.csv"))
    parser.add_argument("--tgr_source_merge_report_csv", type=Path, default=Path("./outputs_pa_tgr/pa_tgr_source_merge.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_pa_tgr_sort(args)
    print("Summary:")
    print(json.dumps(base._safe_json(summary), indent=2, ensure_ascii=False))
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

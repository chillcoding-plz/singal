#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tracklet graph sorting prototype.

This script keeps an existing PA-TSR/PA-TGR sort as the front-end seed, then
treats each positive SigIdx as a trajectory fragment node. It builds a
physics-consistency graph between fragments and merges connected components
under conservative component gates.

The goal is to make the sorting core more "tracklet/trajectory graph" shaped
without changing recognition or relying on labels during sorting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


UNKNOWN_LABEL = 99


@dataclass(frozen=True)
class RunSpec:
    name: str
    pdw_file: Path
    truth_file: Path
    seed_sort_file: Path
    output_dir: Path

    @property
    def output_file(self) -> Path:
        return self.output_dir / f"{self.name}_tracklet_graph_sort.txt"

    @property
    def edge_file(self) -> Path:
        return self.output_dir / f"{self.name}_tracklet_graph_edges.csv"

    @property
    def component_file(self) -> Path:
        return self.output_dir / f"{self.name}_tracklet_graph_components.csv"

    @property
    def summary_file(self) -> Path:
        return self.output_dir / f"{self.name}_tracklet_graph_summary.json"


class UnionFind:
    def __init__(self, nodes: Iterable[int]) -> None:
        self.parent = {int(node): int(node) for node in nodes}
        self.rank = {int(node): 0 for node in nodes}

    def find(self, node: int) -> int:
        node = int(node)
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


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


def default_specs(root: Path, output_root: Path, seed_stage: str) -> List[RunSpec]:
    seed_name = "sort_before_agile_fusion" if seed_stage == "before_agile" else "sort"
    return [
        RunSpec(
            "sample1",
            root / "edata/Test_Data/Sample_1/Merge_PDW_Data.txt",
            root / "edata/Test_Data/Sample_1/Sorted_PDW.txt",
            root / f"outputs_performance_first/sample1/sample1_{seed_name}.txt",
            output_root / "sample1",
        ),
        RunSpec(
            "sample2",
            root / "edata/Test_Data/Sample_2/Merge_PDW_Data.txt",
            root / "edata/Test_Data/Sample_2/Sorted_PDW.txt",
            root / f"outputs_performance_first/sample2/sample2_{seed_name}.txt",
            output_root / "sample2",
        ),
    ]


def circular_delta_deg(a, b) -> np.ndarray:
    return np.abs((np.asarray(a, dtype=np.float64) - float(b) + 180.0) % 360.0 - 180.0)


def circular_span_deg(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    center = circular_mean_deg(values)
    return float(np.max(circular_delta_deg(values, center)))


def circular_mean_deg(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    radians = np.deg2rad(np.mod(values, 360.0))
    return float((np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) + 360.0) % 360.0)


def robust_iqr(values: np.ndarray, floor: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float(floor)
    q75, q25 = np.percentile(values, [75, 25])
    return float(max(q75 - q25, floor))


def read_pdw(path: Path, max_rows: int = 0) -> pd.DataFrame:
    usecols = ["TOA(s)", "Param1", "Param2", "Param4", "Param5"]
    df = pd.read_csv(path, sep=r"\s+", engine="python", usecols=usecols)
    if max_rows > 0:
        df = df.iloc[:max_rows].reset_index(drop=True)
    return df


def read_sigidx(path: Path, max_rows: int = 0) -> np.ndarray:
    data = pd.read_csv(path, sep=r"\s+", engine="python", usecols=["SigIdx"])
    values = pd.to_numeric(data["SigIdx"], errors="raise").to_numpy(dtype=np.int64)
    if max_rows > 0:
        values = values[:max_rows]
    return values


def read_truth(path: Path, max_rows: int = 0) -> pd.DataFrame:
    truth = pd.read_csv(path, sep=r"\s+", engine="python", usecols=["SigIdx", "LABEL"])
    if max_rows > 0:
        truth = truth.iloc[:max_rows].reset_index(drop=True)
    return truth


def write_sort(path: Path, toa: np.ndarray, sigidx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"TOA(s)": toa, "SigIdx": sigidx.astype(np.int64), "LABEL": UNKNOWN_LABEL})
    out.to_csv(path, sep=" ", index=False)


def summarize_tracklets(pdw: pd.DataFrame, sigidx: np.ndarray, min_pulses: int) -> pd.DataFrame:
    work = pdw.copy()
    work["pred_sigidx"] = sigidx.astype(np.int64)
    rows = []
    for pred_id, sub in work[work["pred_sigidx"] > 0].groupby("pred_sigidx", sort=True):
        if len(sub) < min_pulses:
            continue
        toa = np.sort(sub["TOA(s)"].to_numpy(dtype=np.float64))
        diffs_us = np.diff(toa) * 1e6
        diffs_us = diffs_us[(diffs_us > 0.0) & np.isfinite(diffs_us)]
        pri = float(np.median(diffs_us)) if len(diffs_us) else float("nan")
        rows.append(
            {
                "pred_sigidx": int(pred_id),
                "num_pulses": int(len(sub)),
                "start_toa": float(toa[0]),
                "end_toa": float(toa[-1]),
                "duration": float(toa[-1] - toa[0]),
                "median_param1": float(sub["Param1"].median()),
                "median_param2": float(sub["Param2"].median()),
                "median_param4": float(sub["Param4"].median()),
                "median_param5": circular_mean_deg(sub["Param5"].to_numpy(dtype=np.float64)),
                "iqr_param1": robust_iqr(sub["Param1"].to_numpy(dtype=np.float64), 1e-6),
                "iqr_param2": robust_iqr(sub["Param2"].to_numpy(dtype=np.float64), 1e-6),
                "iqr_param4": robust_iqr(sub["Param4"].to_numpy(dtype=np.float64), 1e-6),
                "span_param5": circular_span_deg(sub["Param5"].to_numpy(dtype=np.float64)),
                "median_pri_us": pri,
            }
        )
    return pd.DataFrame(rows)


def feature_scales(summary: pd.DataFrame, args: argparse.Namespace) -> Dict[str, float]:
    return {
        "p1": robust_iqr(summary["median_param1"].to_numpy(dtype=np.float64), args.scale_p1_floor),
        "p2": robust_iqr(summary["median_param2"].to_numpy(dtype=np.float64), args.scale_p2_floor),
        "p4": robust_iqr(summary["median_param4"].to_numpy(dtype=np.float64), args.scale_p4_floor),
        "p5": float(args.scale_p5_deg),
        "pri": robust_iqr(
            summary["median_pri_us"].dropna().to_numpy(dtype=np.float64),
            args.scale_pri_floor_us,
        ),
    }


def node_feature_matrix(summary: pd.DataFrame, scales: Dict[str, float]) -> np.ndarray:
    angle = np.deg2rad(summary["median_param5"].to_numpy(dtype=np.float64))
    features = np.column_stack(
        [
            summary["median_param1"].to_numpy(dtype=np.float64) / scales["p1"],
            summary["median_param2"].to_numpy(dtype=np.float64) / scales["p2"],
            summary["median_param4"].to_numpy(dtype=np.float64) / scales["p4"],
            np.sin(angle) / scales["p5"],
            np.cos(angle) / scales["p5"],
        ]
    )
    return features.astype(np.float64)


def edge_distance(left: pd.Series, right: pd.Series, scales: Dict[str, float], args: argparse.Namespace) -> Dict[str, float]:
    d_p1 = abs(float(left["median_param1"]) - float(right["median_param1"])) / scales["p1"]
    d_p2 = abs(float(left["median_param2"]) - float(right["median_param2"])) / scales["p2"]
    d_p4 = abs(float(left["median_param4"]) - float(right["median_param4"])) / scales["p4"]
    d_p5 = float(circular_delta_deg(float(left["median_param5"]), float(right["median_param5"])) / scales["p5"])
    pri_l = float(left["median_pri_us"])
    pri_r = float(right["median_pri_us"])
    if np.isfinite(pri_l) and np.isfinite(pri_r):
        d_pri = abs(pri_l - pri_r) / scales["pri"]
        pri_ratio = max(pri_l, pri_r) / max(min(pri_l, pri_r), 1e-9)
        harmonic = min(abs(pri_ratio - round(pri_ratio)), abs((1.0 / pri_ratio) - round(1.0 / pri_ratio)))
    else:
        d_pri = float(args.missing_pri_penalty)
        harmonic = 1.0
    gap = max(0.0, max(float(left["start_toa"]), float(right["start_toa"])) - min(float(left["end_toa"]), float(right["end_toa"])))
    time_penalty = gap / max(float(args.time_gap_scale_s), 1e-9)
    distance = (
        args.w_p1 * d_p1
        + args.w_p2 * d_p2
        + args.w_p4 * d_p4
        + args.w_p5 * d_p5
        + args.w_pri * min(d_pri, args.pri_dist_cap)
        + args.w_time * time_penalty
    )
    hard_ok = (
        d_p1 <= args.hard_p1
        and d_p2 <= args.hard_p2
        and d_p4 <= args.hard_p4
        and d_p5 <= args.hard_p5
        and (d_pri <= args.hard_pri or harmonic <= args.harmonic_tol)
        and gap <= args.max_time_gap_s
    )
    return {
        "distance": float(distance),
        "d_p1": float(d_p1),
        "d_p2": float(d_p2),
        "d_p4": float(d_p4),
        "d_p5": float(d_p5),
        "d_pri": float(d_pri),
        "time_gap_s": float(gap),
        "hard_ok": bool(hard_ok),
    }


def build_edges(summary: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if len(summary) < 2:
        return pd.DataFrame(), {}
    scales = feature_scales(summary, args)
    features = node_feature_matrix(summary, scales)
    k = min(int(args.knn), len(summary) - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(features)
    _, indices = nbrs.kneighbors(features)
    lookup = summary.reset_index(drop=True)
    seen = set()
    rows = []
    for i in range(len(lookup)):
        left = lookup.iloc[i]
        for j in indices[i, 1:]:
            a, b = sorted((int(i), int(j)))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            right = lookup.iloc[b]
            metrics = edge_distance(lookup.iloc[a], right, scales, args)
            accepted = bool(metrics["hard_ok"] and metrics["distance"] <= args.link_thresh)
            rows.append(
                {
                    "src_sigidx": int(lookup.iloc[a]["pred_sigidx"]),
                    "dst_sigidx": int(right["pred_sigidx"]),
                    "accepted": accepted,
                    **metrics,
                }
            )
    return pd.DataFrame(rows), scales


def component_is_compact(component: pd.DataFrame, args: argparse.Namespace) -> bool:
    p1_span = float(component["median_param1"].max() - component["median_param1"].min())
    p2_span = float(component["median_param2"].max() - component["median_param2"].min())
    p4_span = float(component["median_param4"].max() - component["median_param4"].min())
    p5_span = circular_span_deg(component["median_param5"].to_numpy(dtype=np.float64))
    pri = component["median_pri_us"].dropna().to_numpy(dtype=np.float64)
    pri_span = float(np.max(pri) - np.min(pri)) if len(pri) >= 2 else 0.0

    compact_score = max(
        p1_span / max(float(args.max_component_p1_span), 1e-9),
        p2_span / max(float(args.max_component_p2_span), 1e-9),
        p4_span / max(float(args.max_component_p4_span), 1e-9),
        p5_span / max(float(args.max_component_p5_span_deg), 1e-9),
        pri_span / max(float(args.max_component_pri_span_us), 1e-9),
    )
    if compact_score > float(args.min_component_purity_score):
        return False
    if p1_span > float(args.max_component_p1_span):
        return False
    if p2_span > float(args.max_component_p2_span):
        return False
    if p4_span > float(args.max_component_p4_span):
        return False
    if p5_span > float(args.max_component_p5_span_deg):
        return False
    if pri_span > float(args.max_component_pri_span_us):
        return False
    return True


def merge_by_graph(sigidx: np.ndarray, summary: pd.DataFrame, edges: pd.DataFrame, args: argparse.Namespace) -> Tuple[np.ndarray, pd.DataFrame]:
    out = sigidx.astype(np.int64).copy()
    nodes = [int(v) for v in summary["pred_sigidx"].tolist()]
    uf = UnionFind(nodes)
    lookup = summary.set_index("pred_sigidx", drop=False)

    accepted_edges = edges[edges["accepted"].astype(bool)].sort_values("distance", kind="mergesort")
    for row in accepted_edges.itertuples(index=False):
        src = int(row.src_sigidx)
        dst = int(row.dst_sigidx)
        src_root = uf.find(src)
        dst_root = uf.find(dst)
        if src_root == dst_root:
            continue
        src_members = [node for node in nodes if uf.find(node) == src_root]
        dst_members = [node for node in nodes if uf.find(node) == dst_root]
        candidate_members = sorted(set(src_members + dst_members))
        candidate = lookup.loc[candidate_members].copy()
        if not component_is_compact(candidate, args):
            continue
        uf.union(src, dst)

    groups: Dict[int, List[int]] = {}
    for node in nodes:
        groups.setdefault(uf.find(node), []).append(node)

    rows = []
    for _, members in groups.items():
        if len(members) < int(args.min_component_batches):
            continue
        component = lookup.loc[members].copy()
        if int(component["num_pulses"].sum()) < int(args.min_component_pulses):
            continue
        fused_id = int(min(members))
        for member in members:
            out[sigidx == int(member)] = fused_id
        rows.append(
            {
                "fused_sigidx": fused_id,
                "merged_sigidx": ",".join(str(int(v)) for v in sorted(members)),
                "num_batches": int(len(members)),
                "num_pulses": int(component["num_pulses"].sum()),
                "median_param1_min": float(component["median_param1"].min()),
                "median_param1_max": float(component["median_param1"].max()),
                "median_param2_min": float(component["median_param2"].min()),
                "median_param2_max": float(component["median_param2"].max()),
                "median_param5_span_deg": circular_span_deg(component["median_param5"].to_numpy(dtype=np.float64)),
            }
        )
    return out, pd.DataFrame(rows)


def load_sort_metrics_module(root: Path):
    for path in [root / "识别/sort_metrics.py", root / "best_extra020_conservative_package/识别/sort_metrics.py"]:
        if path.exists():
            spec = importlib.util.spec_from_file_location("sort_metrics_local", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find sort_metrics.py")


def evaluate(root: Path, pdw: pd.DataFrame, truth_file: Path, sigidx: np.ndarray, args: argparse.Namespace, out_prefix: Path) -> Dict[str, object]:
    truth = read_truth(truth_file, int(args.max_pulses))
    metrics_mod = load_sort_metrics_module(root)
    ids = np.unique(sigidx[sigidx > 0])
    batch_stub = pd.DataFrame({"pred_sigidx": ids, "batch_pred_label": np.full(len(ids), UNKNOWN_LABEL)})
    batch_df, target_df, target_beat_df, beat_df, metrics = metrics_mod.compute_sort_metrics_by_beat(
        pdw["TOA(s)"].to_numpy(dtype=np.float64),
        truth["SigIdx"].to_numpy(dtype=np.int64),
        sigidx.astype(np.int64),
        truth["LABEL"].to_numpy(dtype=np.int64),
        float(args.sort_purity_threshold),
        float(args.sort_min_target_fraction),
        int(args.sort_mix_fail_min_pulses),
        batch_stub,
        float(args.sort_chunk_seconds),
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    batch_df.to_csv(out_prefix.with_name(out_prefix.name + "_batch_eval.csv"), index=False, encoding="utf-8-sig")
    target_df.to_csv(out_prefix.with_name(out_prefix.name + "_target_eval.csv"), index=False, encoding="utf-8-sig")
    target_beat_df.to_csv(out_prefix.with_name(out_prefix.name + "_target_beat_eval.csv"), index=False, encoding="utf-8-sig")
    beat_df.to_csv(out_prefix.with_name(out_prefix.name + "_beat_eval.csv"), index=False, encoding="utf-8-sig")
    out_prefix.with_name(out_prefix.name + "_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dict(metrics)


def run_spec(root: Path, spec: RunSpec, args: argparse.Namespace) -> Dict[str, object]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] {spec.name}: {spec.pdw_file}")
    pdw = read_pdw(spec.pdw_file, int(args.max_pulses))
    seed = read_sigidx(spec.seed_sort_file, int(args.max_pulses))
    if len(pdw) != len(seed):
        raise ValueError(f"{spec.name}: PDW rows ({len(pdw)}) and seed SigIdx rows ({len(seed)}) differ")

    print(f"[graph] {spec.name}: summarize tracklet nodes")
    summary = summarize_tracklets(pdw, seed, int(args.min_node_pulses))
    summary.to_csv(spec.output_dir / f"{spec.name}_tracklet_graph_nodes.csv", index=False, encoding="utf-8-sig")
    edges, scales = build_edges(summary, args)
    edges.to_csv(spec.edge_file, index=False, encoding="utf-8-sig")
    graph_sigidx, components = merge_by_graph(seed, summary, edges, args)
    components.to_csv(spec.component_file, index=False, encoding="utf-8-sig")
    write_sort(spec.output_file, pdw["TOA(s)"].to_numpy(dtype=np.float64), graph_sigidx)

    metrics = evaluate(
        root,
        pdw,
        spec.truth_file,
        graph_sigidx,
        args,
        spec.output_dir / "metrics" / f"{spec.name}_tracklet_graph",
    )
    baseline_metrics = evaluate(
        root,
        pdw,
        spec.truth_file,
        seed,
        args,
        spec.output_dir / "metrics" / f"{spec.name}_seed",
    )
    summary_json = {
        "sample": spec.name,
        "method": "Tracklet Physics-Consistency Graph Sorting",
        "pdw_file": str(spec.pdw_file),
        "seed_sort_file": str(spec.seed_sort_file),
        "output_file": str(spec.output_file),
        "num_pulses": int(len(pdw)),
        "num_nodes": int(len(summary)),
        "num_edges": int(len(edges)),
        "num_accepted_edges": int(edges["accepted"].sum()) if len(edges) else 0,
        "num_components": int(len(components)),
        "num_batches_before": int(len(np.unique(seed[seed > 0]))),
        "num_batches_after": int(len(np.unique(graph_sigidx[graph_sigidx > 0]))),
        "num_changed_pulses": int(np.sum(seed != graph_sigidx)),
        "scales": scales,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
    }
    spec.summary_file.write_text(json.dumps(json_safe(summary_json), indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[done] {spec.name}: sort={metrics.get('sample_sort_acc', float('nan')):.4f}, "
        f"extra={metrics.get('sample_extra_batch_rate', float('nan')):.4f}, "
        f"wrong={metrics.get('sample_wrong_batch_rate', float('nan')):.4f}, "
        f"track={metrics.get('sample_signal_tracking_stability', float('nan')):.4f}"
    )
    return summary_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tracklet graph sorting prototype for sample1/sample2.")
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--sample", choices=["sample1", "sample2", "all"], default="all")
    parser.add_argument("--output_root", type=Path, default=Path("outputs_tracklet_graph"))
    parser.add_argument("--seed_stage", choices=["before_agile", "final"], default="final")
    parser.add_argument("--max_pulses", type=int, default=0)

    parser.add_argument("--min_node_pulses", type=int, default=800)
    parser.add_argument("--knn", type=int, default=18)
    parser.add_argument("--link_thresh", type=float, default=0.18)
    parser.add_argument("--hard_p1", type=float, default=0.12)
    parser.add_argument("--hard_p2", type=float, default=0.08)
    parser.add_argument("--hard_p4", type=float, default=0.20)
    parser.add_argument("--hard_p5", type=float, default=0.06)
    parser.add_argument("--hard_pri", type=float, default=0.45)
    parser.add_argument("--harmonic_tol", type=float, default=0.03)
    parser.add_argument("--missing_pri_penalty", type=float, default=1.5)
    parser.add_argument("--pri_dist_cap", type=float, default=3.0)
    parser.add_argument("--max_time_gap_s", type=float, default=1.0)
    parser.add_argument("--time_gap_scale_s", type=float, default=0.25)

    parser.add_argument("--w_p1", type=float, default=0.65)
    parser.add_argument("--w_p2", type=float, default=0.55)
    parser.add_argument("--w_p4", type=float, default=0.40)
    parser.add_argument("--w_p5", type=float, default=0.85)
    parser.add_argument("--w_pri", type=float, default=0.50)
    parser.add_argument("--w_time", type=float, default=0.10)

    parser.add_argument("--scale_p1_floor", type=float, default=80.0)
    parser.add_argument("--scale_p2_floor", type=float, default=0.20)
    parser.add_argument("--scale_p4_floor", type=float, default=8.0)
    parser.add_argument("--scale_p5_deg", type=float, default=0.55)
    parser.add_argument("--scale_pri_floor_us", type=float, default=40.0)

    parser.add_argument("--min_component_batches", type=int, default=2)
    parser.add_argument("--min_component_pulses", type=int, default=2500)
    parser.add_argument("--min_component_purity_score", type=float, default=0.75)
    parser.add_argument("--max_component_p1_span", type=float, default=5.0)
    parser.add_argument("--max_component_p2_span", type=float, default=0.05)
    parser.add_argument("--max_component_p4_span", type=float, default=2.0)
    parser.add_argument("--max_component_p5_span_deg", type=float, default=0.02)
    parser.add_argument("--max_component_pri_span_us", type=float, default=25.0)

    parser.add_argument("--sort_purity_threshold", type=float, default=0.90)
    parser.add_argument("--sort_min_target_fraction", type=float, default=0.10)
    parser.add_argument("--sort_mix_fail_min_pulses", type=int, default=150)
    parser.add_argument("--sort_chunk_seconds", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    specs = default_specs(root, args.output_root, str(args.seed_stage))
    if args.sample != "all":
        specs = [spec for spec in specs if spec.name == args.sample]
    all_summaries = [run_spec(root, spec, args) for spec in specs]
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "tracklet_graph_all_summary.json").write_text(
        json.dumps(json_safe(all_summaries), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

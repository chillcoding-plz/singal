#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Performance-first single-file entry for PDW sorting and recognition.

This file is intentionally an entry-point wrapper around the strongest local
building blocks in this workspace:

1. PA-TGR / PA-TSR sorting for high beat-level sorting accuracy and stable
   SigIdx tracking.
2. XGBoost pulse recognition with sort-aware features.
3. Batch-level kNN conformal OOD rejection for unknown target labels.

The design choice is deliberate: the generic non-HDBSCAN clustering trials were
less competitive on this imbalanced dataset, so the default keeps the strongest
physics-aware sorter and concentrates new orchestration on metric-first model
selection, stable outputs, and reproducible end-to-end evaluation.
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
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


UNKNOWN_LABEL = 99
STRICT_PRIOR_CACHE: Dict[Tuple[str, int], Dict[str, float]] = {}


@dataclass
class RunSpec:
    name: str
    pdw_file: Path
    truth_file: Optional[Path]
    output_dir: Path

    @property
    def sort_file(self) -> Path:
        return self.output_dir / f"{self.name}_sort.txt"

    @property
    def final_file(self) -> Path:
        return self.output_dir / f"{self.name}_final.txt"

    @property
    def sort_report(self) -> Path:
        return self.output_dir / f"{self.name}_sort_summary.json"

    @property
    def raw_sort_file(self) -> Path:
        return self.output_dir / f"{self.name}_raw_sort.txt"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "sort_metrics"

    @property
    def recognition_dir(self) -> Path:
        return self.output_dir / "recognition"


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
        x = float(value)
        return None if not np.isfinite(x) else x
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def default_truth_for_pdw(pdw_file: Path) -> Optional[Path]:
    candidate = pdw_file.parent / "Sorted_PDW.txt"
    return candidate if candidate.exists() else None


def infer_name_from_path(pdw_file: Path) -> str:
    text = str(pdw_file).replace("\\", "/").lower()
    if "sample_1" in text or "sample1" in text:
        return "sample1"
    if "sample_2" in text or "sample2" in text:
        return "sample2"
    return pdw_file.stem.lower().replace("merge_pdw_data", "sample")


def resolve_run_specs(args: argparse.Namespace) -> List[RunSpec]:
    root = Path(args.output_root)
    specs: List[RunSpec] = []
    if args.sample in {"sample1", "all"}:
        pdw = Path("edata/Test_Data/Sample_1/Merge_PDW_Data.txt")
        specs.append(RunSpec("sample1", pdw, default_truth_for_pdw(pdw), root / "sample1"))
    if args.sample in {"sample2", "all"}:
        pdw = Path("edata/Test_Data/Sample_2/Merge_PDW_Data.txt")
        specs.append(RunSpec("sample2", pdw, default_truth_for_pdw(pdw), root / "sample2"))
    if args.sample == "custom":
        if args.pdw_file is None:
            raise ValueError("--pdw_file is required when --sample custom")
        pdw = Path(args.pdw_file)
        truth = Path(args.truth_file) if args.truth_file else default_truth_for_pdw(pdw)
        name = args.name or infer_name_from_path(pdw)
        specs.append(RunSpec(name, pdw, truth, root / name))
    return specs


def import_sort_backend(backend: str):
    if backend == "pa_tgr":
        import pa_tgr_hdbscan_sort as sorter

        return sorter, sorter.run_pa_tgr_sort
    if backend == "pa_tsr":
        import pa_tsr_hdbscan_sort as sorter

        return sorter, sorter.run_pa_tsr_sort
    raise ValueError(f"Unsupported sort backend: {backend}")


def load_sort_metrics_module():
    for path in [Path("识别/sort_metrics.py"), Path("sort_metrics.py")]:
        if path.exists():
            spec = importlib.util.spec_from_file_location("sort_metrics_local", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find sort_metrics.py")


def read_sort_sigidx(path: Path) -> np.ndarray:
    data = pd.read_csv(path, sep=r"\s+", engine="python")
    if "SigIdx" in data.columns:
        col = "SigIdx"
    elif data.shape[1] >= 2:
        col = data.columns[1]
    else:
        raise ValueError(f"Missing SigIdx column in {path}")
    return pd.to_numeric(data[col], errors="raise").to_numpy(dtype=np.int64)


def write_sort_sigidx(path: Path, toa: np.ndarray, sigidx: np.ndarray) -> None:
    out = pd.DataFrame(
        {
            "TOA(s)": [f"{float(v):.9f}" for v in toa],
            "SigIdx": sigidx.astype(np.int64),
        }
    )
    out.to_csv(path, sep=" ", index=False)


def evaluate_sort_sigidx(
    pdw: pd.DataFrame,
    truth_file: Optional[Path],
    sigidx: np.ndarray,
    args: argparse.Namespace,
    out_prefix: Path,
) -> Dict[str, object]:
    if truth_file is None or not truth_file.exists():
        return {"skipped": True, "reason": "truth file not found"}

    truth = pd.read_csv(truth_file, sep=r"\s+", engine="python").iloc[:, :3].copy()
    truth.columns = ["TOA(s)", "SigIdx", "LABEL"]
    if len(truth) != len(sigidx):
        raise ValueError(f"Truth rows ({len(truth)}) and SigIdx rows ({len(sigidx)}) do not match.")

    metrics_mod = load_sort_metrics_module()
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
    target_df.to_csv(out_prefix.with_name(out_prefix.name + "_target_accuracy.csv"), index=False, encoding="utf-8-sig")
    target_beat_df.to_csv(out_prefix.with_name(out_prefix.name + "_target_beat_eval.csv"), index=False, encoding="utf-8-sig")
    beat_df.to_csv(out_prefix.with_name(out_prefix.name + "_beat_eval.csv"), index=False, encoding="utf-8-sig")
    out_prefix.with_name(out_prefix.name + "_metrics.json").write_text(
        json.dumps(json_safe(metrics), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dict(metrics)


def agile_fusion_enabled(spec: RunSpec, args: argparse.Namespace) -> bool:
    if args.agile_fusion == "on":
        return True
    if args.agile_fusion == "off":
        return False
    return True


def summarize_pred_batches(pdw: pd.DataFrame, sigidx: np.ndarray) -> pd.DataFrame:
    work = pdw.copy()
    work["Pred"] = sigidx
    rows = []
    for pred_id, sub in work[work["Pred"] > 0].groupby("Pred", sort=True):
        toa = np.sort(sub["TOA(s)"].to_numpy(dtype=np.float64))
        dtoa_us = np.diff(toa) * 1e6
        dtoa_us = dtoa_us[np.isfinite(dtoa_us) & (dtoa_us > 0.0)]
        if dtoa_us.size > 0:
            median_pri_us = float(np.median(dtoa_us))
            pri_iqr_us = robust_iqr(dtoa_us, 0.0)
        else:
            median_pri_us = float("nan")
            pri_iqr_us = float("nan")
        rows.append(
            {
                "pred_sigidx": int(pred_id),
                "num_pulses": int(len(sub)),
                "start_toa": float(toa[0]) if len(toa) else float("nan"),
                "end_toa": float(toa[-1]) if len(toa) else float("nan"),
                "median_param1": float(sub["Param1"].median()),
                "median_param2": float(sub["Param2"].median()),
                "median_param4": float(sub["Param4"].median()),
                "median_param5": float(sub["Param5"].median()),
                "median_pri_us": median_pri_us,
                "pri_iqr_us": pri_iqr_us,
            }
        )
    return pd.DataFrame(rows)


def circular_degrees_delta(a, b):
    delta = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return np.minimum(delta, 360.0 - delta)


def robust_iqr(values, floor: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return floor
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    return max(float(q75 - q25), floor)


def read_core_pdw(path: Path, max_rows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python").iloc[:, :8].copy()
    df.columns = ["TOA(s)", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6", "Param7"]
    if max_rows > 0 and len(df) > max_rows:
        step = max(len(df) // max_rows, 1)
        df = df.iloc[::step].head(max_rows).reset_index(drop=True)
    return df


def load_strict_train_priors(train_dir: Path, sample_rows_per_class: int = 50000) -> Dict[str, float]:
    key = (str(train_dir.resolve()), int(sample_rows_per_class))
    if key in STRICT_PRIOR_CACHE:
        return STRICT_PRIOR_CACHE[key]

    p1_values: List[np.ndarray] = []
    p2_values: List[np.ndarray] = []
    p4_values: List[np.ndarray] = []
    for path in sorted(train_dir.glob("Class_*.txt")):
        df = read_core_pdw(path, max_rows=sample_rows_per_class)
        p1_values.append(df["Param1"].to_numpy(dtype=np.float64))
        p2_values.append(df["Param2"].to_numpy(dtype=np.float64))
        p4_values.append(df["Param4"].to_numpy(dtype=np.float64))

    priors = {
        "param1_iqr": robust_iqr(np.concatenate(p1_values) if p1_values else [], 25.0),
        "param2_iqr": robust_iqr(np.concatenate(p2_values) if p2_values else [], 0.10),
        "param4_iqr": robust_iqr(np.concatenate(p4_values) if p4_values else [], 2.0),
    }
    STRICT_PRIOR_CACHE[key] = priors
    return priors


def derive_strict_feature_scales(report: pd.DataFrame, priors: Dict[str, float]) -> Dict[str, float]:
    p5_sorted = np.sort(report["median_param5"].to_numpy(dtype=np.float64))
    p5_gaps = np.diff(p5_sorted)
    p5_gaps = p5_gaps[p5_gaps > 0]
    p5_gap_scale = float(np.median(p5_gaps)) if len(p5_gaps) > 0 else 0.05
    return {
        "param1": max(float(priors["param1_iqr"]), robust_iqr(report["median_param1"], 25.0), 25.0),
        "param2": max(float(priors["param2_iqr"]), robust_iqr(report["median_param2"], 0.10), 0.10),
        "param4": max(float(priors["param4_iqr"]) * 3.0, robust_iqr(report["median_param4"], 4.0), 4.0),
        "param5": max(robust_iqr(report["median_param5"], 0.15), p5_gap_scale * 3.0, 0.15),
    }


def strict_feature_matrix(report: pd.DataFrame, scales: Dict[str, float]) -> np.ndarray:
    return np.column_stack(
        [
            0.20 * report["median_param1"].to_numpy(dtype=np.float64) / scales["param1"],
            0.30 * report["median_param2"].to_numpy(dtype=np.float64) / scales["param2"],
            0.30 * report["median_param4"].to_numpy(dtype=np.float64) / scales["param4"],
            1.00 * report["median_param5"].to_numpy(dtype=np.float64) / scales["param5"],
        ]
    )


def choose_strict_dbscan_eps(features: np.ndarray) -> float:
    if len(features) <= 2:
        return 0.60
    neighbors = min(4, len(features) - 1)
    nbrs = NearestNeighbors(n_neighbors=neighbors + 1, metric="euclidean")
    nbrs.fit(features)
    distances, _ = nbrs.kneighbors(features)
    kdist = distances[:, -1]
    q25, q75 = np.percentile(kdist, [25.0, 75.0])
    eps = float(np.quantile(kdist, 0.70) + max(q75 - q25, 0.0))
    return float(np.clip(eps, 0.40, 1.80))


def strict_candidate_min_pulses(report: pd.DataFrame) -> int:
    q = float(np.quantile(report["num_pulses"].to_numpy(dtype=np.float64), 0.25))
    return int(np.clip(q, 50, 1500))


def build_strict_dbscan_groups(report: pd.DataFrame, priors: Dict[str, float]) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, float]]:
    candidate_min = strict_candidate_min_pulses(report)
    candidates = report.loc[report["num_pulses"] >= candidate_min].copy()
    if len(candidates) < 2:
        return [], {"candidate_min_pulses": float(candidate_min), "eps": 0.0, "candidate_batches": float(len(candidates))}

    scales = derive_strict_feature_scales(candidates, priors)
    features = strict_feature_matrix(candidates, scales)
    eps = choose_strict_dbscan_eps(features)
    labels = DBSCAN(eps=eps, min_samples=2, metric="euclidean").fit_predict(features)

    groups: List[Tuple[str, pd.DataFrame]] = []
    min_group_pulses = max(2000, candidate_min * 2)
    for label in sorted(v for v in np.unique(labels) if int(v) >= 0):
        sub = candidates.loc[labels == label].copy()
        if len(sub) < 2:
            continue
        if int(sub["num_pulses"].sum()) < min_group_pulses:
            continue
        groups.append((f"strict_dbscan_family_{int(label)}", sub))
    meta = {
        "candidate_min_pulses": float(candidate_min),
        "eps": float(eps),
        "candidate_batches": float(len(candidates)),
        "param1_scale": float(scales["param1"]),
        "param2_scale": float(scales["param2"]),
        "param4_scale": float(scales["param4"]),
        "param5_scale": float(scales["param5"]),
    }
    return groups, meta


def build_strict_residual_suppression_groups(report: pd.DataFrame, priors: Dict[str, float]) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, float]]:
    if len(report) < 2:
        return [], {"small_threshold": 0.0, "close_threshold": 0.0}

    scales = derive_strict_feature_scales(report, priors)
    features = strict_feature_matrix(report, scales)
    sizes = report["num_pulses"].to_numpy(dtype=np.float64)
    small_threshold = int(np.clip(np.quantile(sizes, 0.60), 500, 10000))

    dist = np.sqrt(((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    larger_neighbor_dist = np.full(len(report), np.inf, dtype=np.float64)
    for idx in range(len(report)):
        larger_mask = sizes >= max(sizes[idx] * 1.25, sizes[idx] + 1.0)
        if np.any(larger_mask):
            larger_neighbor_dist[idx] = float(np.min(dist[idx, larger_mask]))
    finite = larger_neighbor_dist[np.isfinite(larger_neighbor_dist)]
    close_threshold = float(np.clip(np.quantile(finite, 0.65), 0.45, 1.75)) if len(finite) > 0 else 0.95

    base_mask = (sizes < small_threshold) & (larger_neighbor_dist <= close_threshold)
    micro_mask = (sizes < max(200, small_threshold // 3)) & (larger_neighbor_dist <= close_threshold * 1.15)

    groups: List[Tuple[str, pd.DataFrame]] = []
    if np.any(base_mask):
        groups.append(("strict_small_residual_shards", report.loc[base_mask].copy()))
    if np.any(micro_mask & ~base_mask):
        groups.append(("strict_micro_residual_shards", report.loc[micro_mask & ~base_mask].copy()))
    return groups, {"small_threshold": float(small_threshold), "close_threshold": float(close_threshold)}


def build_frequency_agile_pri_groups(
    report: pd.DataFrame,
    priors: Dict[str, float],
) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, float]]:
    """Find same-emitter frequency-agile fragments without using labels.

    Frequency-agile radars may occupy several Param1 bands while preserving pulse
    width/modulation-like attributes, direction of arrival, and PRI.  Therefore
    Param1 is only a soft consistency term here; Param2, DOA, and PRI carry the
    hard physical gates.
    """
    if len(report) < 2:
        return [], {"candidate_batches": 0.0}

    pri_values = report["median_pri_us"].to_numpy(dtype=np.float64)
    pri_values = pri_values[np.isfinite(pri_values) & (pri_values > 0.0)]
    if len(pri_values) < 2:
        return [], {"candidate_batches": 0.0, "reason": "insufficient_pri"}

    candidate_min = int(np.clip(np.quantile(report["num_pulses"].to_numpy(dtype=np.float64), 0.20), 100, 1500))
    candidates = report.loc[
        (report["num_pulses"] >= candidate_min)
        & np.isfinite(report["median_pri_us"].to_numpy(dtype=np.float64))
        & (report["median_pri_us"] > 0.0)
    ].copy()
    if len(candidates) < 2:
        return [], {"candidate_batches": float(len(candidates)), "candidate_min_pulses": float(candidate_min)}

    rows = candidates.reset_index(drop=True)
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    p1 = rows["median_param1"].to_numpy(dtype=np.float64)
    p2 = rows["median_param2"].to_numpy(dtype=np.float64)
    p4 = rows["median_param4"].to_numpy(dtype=np.float64)
    p5 = rows["median_param5"].to_numpy(dtype=np.float64)
    pri = rows["median_pri_us"].to_numpy(dtype=np.float64)

    accepted_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_p1_abs = abs(p1[i] - p1[j])
            d_p2_abs = abs(p2[i] - p2[j])
            d_p4_abs = abs(p4[i] - p4[j])
            d_p5_abs = float(circular_degrees_delta(p5[i], p5[j]))
            d_pri_abs = abs(pri[i] - pri[j])
            pri_gate = max(1.0, 0.03 * min(pri[i], pri[j]))
            if d_p2_abs > 0.05 or d_p4_abs > 5.0 or d_p5_abs > 0.35 or d_pri_abs > pri_gate or d_p1_abs > 3000.0:
                continue
            union(i, j)
            accepted_edges += 1

    groups: List[Tuple[str, pd.DataFrame]] = []
    for root in sorted({find(i) for i in range(n)}):
        member_idx = [i for i in range(n) if find(i) == root]
        if len(member_idx) < 2:
            continue
        sub = rows.iloc[member_idx].copy()
        total_pulses = int(sub["num_pulses"].sum())
        param1_span = float(sub["median_param1"].max() - sub["median_param1"].min())
        if total_pulses < max(50000, candidate_min * 3):
            continue
        if param1_span < 80.0:
            continue
        if float(circular_degrees_delta(sub["median_param5"].min(), sub["median_param5"].max())) > 0.60:
            continue
        min_pri = float(sub["median_pri_us"].min())
        if float(sub["median_pri_us"].max() - sub["median_pri_us"].min()) > max(1.0, 0.03 * min_pri):
            continue
        if float(sub["median_param2"].max() - sub["median_param2"].min()) > 0.05:
            continue
        if float(sub["median_param4"].max() - sub["median_param4"].min()) > 5.0:
            continue
        if param1_span > 3000.0:
            continue
        groups.append(("frequency_agile_pri_family", sub))

    meta = {
        "candidate_min_pulses": float(candidate_min),
        "candidate_batches": float(len(candidates)),
        "accepted_edges": float(accepted_edges),
        "groups": float(len(groups)),
        "param1_agility_gate": 3000.0,
        "param1_min_agility_span": 80.0,
        "param2_gate": 0.05,
        "param4_gate": 5.0,
        "doa_gate_deg": 0.35,
        "pri_relative_gate": 0.03,
        "pri_min_gate_us": 1.0,
    }
    return groups, meta


def build_amplitude_split_pri_groups(report: pd.DataFrame) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, float]]:
    """Merge batches split only by amplitude-like Param4.

    Some emitters keep carrier-like parameters, DOA, and PRI nearly invariant
    while received amplitude changes with scan phase or propagation.  In that
    case a Param4-only split is an over-segmentation, not a new source.
    """
    if len(report) < 2:
        return [], {"candidate_batches": 0.0}

    candidate_min = int(np.clip(np.quantile(report["num_pulses"].to_numpy(dtype=np.float64), 0.20), 100, 1500))
    candidates = report.loc[
        (report["num_pulses"] >= candidate_min)
        & np.isfinite(report["median_pri_us"].to_numpy(dtype=np.float64))
        & (report["median_pri_us"] > 0.0)
    ].copy()
    if len(candidates) < 2:
        return [], {"candidate_batches": float(len(candidates)), "candidate_min_pulses": float(candidate_min)}

    rows = candidates.reset_index(drop=True)
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    p1 = rows["median_param1"].to_numpy(dtype=np.float64)
    p2 = rows["median_param2"].to_numpy(dtype=np.float64)
    p4 = rows["median_param4"].to_numpy(dtype=np.float64)
    p5 = rows["median_param5"].to_numpy(dtype=np.float64)
    pri = rows["median_pri_us"].to_numpy(dtype=np.float64)

    accepted_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_p1_abs = abs(p1[i] - p1[j])
            d_p2_abs = abs(p2[i] - p2[j])
            d_p4_abs = abs(p4[i] - p4[j])
            d_p5_abs = float(circular_degrees_delta(p5[i], p5[j]))
            d_pri_abs = abs(pri[i] - pri[j])
            pri_gate = max(0.5, 0.01 * min(pri[i], pri[j]))
            if d_p1_abs > 10.0 or d_p2_abs > 0.02 or d_p5_abs > 0.20 or d_pri_abs > pri_gate:
                continue
            if d_p4_abs < 15.0:
                continue
            union(i, j)
            accepted_edges += 1

    groups: List[Tuple[str, pd.DataFrame]] = []
    for root in sorted({find(i) for i in range(n)}):
        member_idx = [i for i in range(n) if find(i) == root]
        if len(member_idx) < 2:
            continue
        sub = rows.iloc[member_idx].copy()
        if int(sub["num_pulses"].sum()) < max(50000, candidate_min * 3):
            continue
        if float(sub["median_param1"].max() - sub["median_param1"].min()) > 10.0:
            continue
        if float(sub["median_param2"].max() - sub["median_param2"].min()) > 0.02:
            continue
        if float(circular_degrees_delta(sub["median_param5"].min(), sub["median_param5"].max())) > 0.20:
            continue
        min_pri = float(sub["median_pri_us"].min())
        if float(sub["median_pri_us"].max() - sub["median_pri_us"].min()) > max(0.5, 0.01 * min_pri):
            continue
        if float(sub["median_param4"].max() - sub["median_param4"].min()) < 15.0:
            continue
        groups.append(("amplitude_split_pri_family", sub))

    meta = {
        "candidate_min_pulses": float(candidate_min),
        "candidate_batches": float(len(candidates)),
        "accepted_edges": float(accepted_edges),
        "groups": float(len(groups)),
        "param1_gate": 10.0,
        "param2_gate": 0.02,
        "doa_gate_deg": 0.20,
        "pri_relative_gate": 0.01,
        "pri_min_gate_us": 0.5,
        "param4_min_split": 15.0,
    }
    return groups, meta


def build_pri_harmonic_groups(report: pd.DataFrame) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, float]]:
    """Merge fragments whose PRI estimates are integer harmonics.

    When a pulse train is intermittently missed or interleaved, local PRI
    estimation can lock onto 2x/3x (or reciprocal) periods.  If RF-like
    parameters and DOA remain tight, this is treated as a timing-estimation
    artifact rather than a separate emitter.
    """
    if len(report) < 2:
        return [], {"candidate_batches": 0.0}

    candidate_min = int(np.clip(np.quantile(report["num_pulses"].to_numpy(dtype=np.float64), 0.20), 100, 1500))
    candidates = report.loc[
        (report["num_pulses"] >= candidate_min)
        & np.isfinite(report["median_pri_us"].to_numpy(dtype=np.float64))
        & (report["median_pri_us"] > 0.0)
    ].copy()
    if len(candidates) < 2:
        return [], {"candidate_batches": float(len(candidates)), "candidate_min_pulses": float(candidate_min)}

    rows = candidates.reset_index(drop=True)
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    p1 = rows["median_param1"].to_numpy(dtype=np.float64)
    p2 = rows["median_param2"].to_numpy(dtype=np.float64)
    p4 = rows["median_param4"].to_numpy(dtype=np.float64)
    p5 = rows["median_param5"].to_numpy(dtype=np.float64)
    pri = rows["median_pri_us"].to_numpy(dtype=np.float64)

    accepted_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(p1[i] - p1[j]) > 300.0:
                continue
            if abs(p2[i] - p2[j]) > 0.10:
                continue
            if abs(p4[i] - p4[j]) > 3.0:
                continue
            if float(circular_degrees_delta(p5[i], p5[j])) > 0.70:
                continue
            small, large = sorted((pri[i], pri[j]))
            ratio = large / small
            harmonic = min(abs(ratio - 2.0), abs(ratio - 3.0), abs(ratio - 4.0))
            if harmonic > 0.04:
                continue
            union(i, j)
            accepted_edges += 1

    groups: List[Tuple[str, pd.DataFrame]] = []
    for root in sorted({find(i) for i in range(n)}):
        member_idx = [i for i in range(n) if find(i) == root]
        if len(member_idx) < 2:
            continue
        sub = rows.iloc[member_idx].copy()
        if int(sub["num_pulses"].sum()) < max(50000, candidate_min * 3):
            continue
        if float(sub["median_param1"].max() - sub["median_param1"].min()) > 300.0:
            continue
        if float(sub["median_param2"].max() - sub["median_param2"].min()) > 0.10:
            continue
        if float(sub["median_param4"].max() - sub["median_param4"].min()) > 3.0:
            continue
        if float(circular_degrees_delta(sub["median_param5"].min(), sub["median_param5"].max())) > 0.70:
            continue
        groups.append(("pri_harmonic_family", sub))

    meta = {
        "candidate_min_pulses": float(candidate_min),
        "candidate_batches": float(len(candidates)),
        "accepted_edges": float(accepted_edges),
        "groups": float(len(groups)),
        "param1_gate": 300.0,
        "param2_gate": 0.10,
        "param4_gate": 3.0,
        "doa_gate_deg": 0.70,
        "harmonic_ratio_tolerance": 0.04,
    }
    return groups, meta


def build_same_carrier_multimode_groups(report: pd.DataFrame) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, float]]:
    """Merge same-carrier, same-PRI fragments split by pulse mode/amplitude.

    This rule is stricter on carrier-like Param1, DOA, and PRI, but deliberately
    allows Param2/Param4 changes.  It models a radar that keeps carrier and scan
    direction stable while switching pulse width/modulation or received level.
    """
    if len(report) < 2:
        return [], {"candidate_batches": 0.0}

    candidate_min = int(np.clip(np.quantile(report["num_pulses"].to_numpy(dtype=np.float64), 0.20), 100, 1500))
    candidates = report.loc[
        (report["num_pulses"] >= candidate_min)
        & np.isfinite(report["median_pri_us"].to_numpy(dtype=np.float64))
        & (report["median_pri_us"] > 0.0)
    ].copy()
    if len(candidates) < 2:
        return [], {"candidate_batches": float(len(candidates)), "candidate_min_pulses": float(candidate_min)}

    rows = candidates.reset_index(drop=True)
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    p1 = rows["median_param1"].to_numpy(dtype=np.float64)
    p5 = rows["median_param5"].to_numpy(dtype=np.float64)
    pri = rows["median_pri_us"].to_numpy(dtype=np.float64)

    accepted_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_p1_abs = abs(p1[i] - p1[j])
            d_p5_abs = float(circular_degrees_delta(p5[i], p5[j]))
            d_pri_abs = abs(pri[i] - pri[j])
            pri_gate = max(0.5, 0.01 * min(pri[i], pri[j]))
            if d_p1_abs > 120.0 or d_p5_abs > 0.30 or d_pri_abs > pri_gate:
                continue
            union(i, j)
            accepted_edges += 1

    groups: List[Tuple[str, pd.DataFrame]] = []
    for root in sorted({find(i) for i in range(n)}):
        member_idx = [i for i in range(n) if find(i) == root]
        if len(member_idx) < 2:
            continue
        sub = rows.iloc[member_idx].copy()
        if int(sub["num_pulses"].sum()) < max(50000, candidate_min * 3):
            continue
        if float(sub["median_param1"].max() - sub["median_param1"].min()) > 120.0:
            continue
        if float(circular_degrees_delta(sub["median_param5"].min(), sub["median_param5"].max())) > 0.30:
            continue
        min_pri = float(sub["median_pri_us"].min())
        if float(sub["median_pri_us"].max() - sub["median_pri_us"].min()) > max(0.5, 0.01 * min_pri):
            continue
        groups.append(("same_carrier_multimode_family", sub))

    meta = {
        "candidate_min_pulses": float(candidate_min),
        "candidate_batches": float(len(candidates)),
        "accepted_edges": float(accepted_edges),
        "groups": float(len(groups)),
        "min_group_pulses": 15000.0,
        "param1_gate": 120.0,
        "doa_gate_deg": 0.30,
        "pri_relative_gate": 0.01,
        "pri_min_gate_us": 0.5,
    }
    return groups, meta


def build_same_bearing_staggered_pri_groups(report: pd.DataFrame) -> Tuple[List[Tuple[str, pd.DataFrame]], Dict[str, float]]:
    """Merge high-evidence groups with common bearing and staggered PRI.

    Some emitters use multiple PRI states and pulse modes under the same scan
    direction.  The rule ignores labels and only joins large, stable families
    whose DOA is tight and whose PRI values occupy a bounded stagger set.
    """
    return [], {"enabled": False, "reason": "disabled in conservative extra<0.2 release"}

    candidate_min = int(np.clip(np.quantile(report["num_pulses"].to_numpy(dtype=np.float64), 0.20), 100, 1500))
    candidates = report.loc[
        (report["num_pulses"] >= candidate_min)
        & np.isfinite(report["median_pri_us"].to_numpy(dtype=np.float64))
        & (report["median_pri_us"] > 0.0)
    ].copy()
    if len(candidates) < 3:
        return [], {"candidate_batches": float(len(candidates)), "candidate_min_pulses": float(candidate_min)}

    rows = candidates.reset_index(drop=True)
    n = len(rows)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    p5 = rows["median_param5"].to_numpy(dtype=np.float64)
    pri = rows["median_pri_us"].to_numpy(dtype=np.float64)

    accepted_edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_p5 = float(circular_degrees_delta(p5[i], p5[j]))
            small, large = sorted((pri[i], pri[j]))
            rel_span = (large - small) / max(small, 1e-9)
            if d_p5 <= 0.70 and rel_span <= 0.25:
                union(i, j)
                accepted_edges += 1

    groups: List[Tuple[str, pd.DataFrame]] = []
    for root in sorted({find(i) for i in range(n)}):
        member_idx = [i for i in range(n) if find(i) == root]
        if len(member_idx) < 3:
            continue
        sub = rows.iloc[member_idx].copy()
        if int(sub["num_pulses"].sum()) < max(30000, candidate_min * 5):
            continue
        if float(circular_degrees_delta(sub["median_param5"].min(), sub["median_param5"].max())) > 0.80:
            continue
        min_pri = float(sub["median_pri_us"].min())
        max_pri = float(sub["median_pri_us"].max())
        if (max_pri - min_pri) / max(min_pri, 1e-9) > 0.30:
            continue
        groups.append(("same_bearing_staggered_pri_family", sub))

    meta = {
        "candidate_min_pulses": float(candidate_min),
        "candidate_batches": float(len(candidates)),
        "accepted_edges": float(accepted_edges),
        "groups": float(len(groups)),
        "min_group_pulses": 30000.0,
        "min_group_batches": 3.0,
        "doa_gate_deg": 0.70,
        "group_doa_gate_deg": 0.80,
        "pair_pri_relative_span": 0.25,
        "group_pri_relative_span": 0.30,
    }
    return groups, meta


def suppress_sparse_beat_segments(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    chunk_seconds: float,
    min_pulses: int,
    hetero_max_pulses: int,
    hetero_param1_iqr: float,
    hetero_doa_iqr_deg: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if len(sigidx) == 0:
        return sigidx, {"enabled": False}

    fused = sigidx.copy()
    toa = pdw["TOA(s)"].to_numpy(dtype=np.float64)
    beat = np.floor((toa - float(np.min(toa))) / float(chunk_seconds)).astype(np.int64)
    work = pd.DataFrame({"beat": beat, "sigidx": fused})
    positive = work["sigidx"] > 0
    counts = work.loc[positive].groupby(["beat", "sigidx"], sort=False).size()

    suppressed_segments = 0
    suppressed_pulses = 0
    for (beat_id, pred_id), count in counts.items():
        mask = (beat == int(beat_id)) & (fused == int(pred_id))
        pulse_count = int(mask.sum())
        if pulse_count == 0:
            continue
        suppress = min_pulses > 1 and pulse_count < int(min_pulses)
        if (
            not suppress
            and hetero_max_pulses > 1
            and pulse_count < int(hetero_max_pulses)
            and pulse_count >= 2
        ):
            sub = pdw.loc[mask]
            p1_iqr = robust_iqr(sub["Param1"].to_numpy(dtype=np.float64), 0.0)
            doa_iqr = robust_iqr(sub["Param5"].to_numpy(dtype=np.float64), 0.0)
            suppress = p1_iqr > float(hetero_param1_iqr) and doa_iqr > float(hetero_doa_iqr_deg)
        if not suppress:
            continue
        fused[mask] = 0
        suppressed_segments += 1
        suppressed_pulses += pulse_count

    meta = {
        "enabled": True,
        "chunk_seconds": float(chunk_seconds),
        "min_pulses": int(min_pulses),
        "hetero_max_pulses": int(hetero_max_pulses),
        "hetero_param1_iqr": float(hetero_param1_iqr),
        "hetero_doa_iqr_deg": float(hetero_doa_iqr_deg),
        "suppressed_segments": int(suppressed_segments),
        "suppressed_pulses": int(suppressed_pulses),
    }
    return fused, meta


def suppress_duplicate_beat_fragments(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    chunk_seconds: float,
    duplicate_ratio: float,
    min_pri_us: float,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, object]]:
    if duplicate_ratio <= 0.0 or len(sigidx) == 0:
        return sigidx, pd.DataFrame(), {"enabled": False}

    fused = sigidx.copy()
    global_report = summarize_pred_batches(pdw, fused)
    if len(global_report) < 2:
        return fused, pd.DataFrame(), {"enabled": True, "suppressed_segments": 0, "suppressed_pulses": 0}
    centers = global_report.set_index("pred_sigidx")

    toa = pdw["TOA(s)"].to_numpy(dtype=np.float64)
    beat = np.floor((toa - float(np.min(toa))) / float(chunk_seconds)).astype(np.int64)
    work = pd.DataFrame({"beat": beat, "sigidx": fused})
    counts = work.loc[work["sigidx"] > 0].groupby(["beat", "sigidx"], sort=False).size().reset_index(name="count")

    rows: List[Dict[str, object]] = []
    for beat_id, sub_counts in counts.groupby("beat", sort=False):
        ids = [int(v) for v in sub_counts["sigidx"].tolist() if int(v) in centers.index]
        if len(ids) < 2:
            continue
        id_to_count = {int(r["sigidx"]): int(r["count"]) for _, r in sub_counts.iterrows()}
        parent = {pid: pid for pid in ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, a in enumerate(ids):
            ca = centers.loc[a]
            for b in ids[i + 1 :]:
                cb = centers.loc[b]
                d_p1 = abs(float(ca["median_param1"]) - float(cb["median_param1"]))
                d_p5 = float(circular_degrees_delta(float(ca["median_param5"]), float(cb["median_param5"])))
                pri_a = float(ca["median_pri_us"])
                pri_b = float(cb["median_pri_us"])
                if not (np.isfinite(pri_a) and np.isfinite(pri_b) and pri_a > 0.0 and pri_b > 0.0):
                    continue
                if min(pri_a, pri_b) < float(min_pri_us):
                    continue
                d_pri = abs(pri_a - pri_b)
                pri_gate = max(0.5, 0.01 * min(pri_a, pri_b))
                if d_p1 <= 120.0 and d_p5 <= 0.30 and d_pri <= pri_gate:
                    union(a, b)

        groups: Dict[int, List[int]] = {}
        for pid in ids:
            groups.setdefault(find(pid), []).append(pid)
        for members in groups.values():
            if len(members) < 2:
                continue
            anchor = max(members, key=lambda pid: id_to_count.get(pid, 0))
            anchor_count = id_to_count.get(anchor, 0)
            if anchor_count <= 0:
                continue
            for pid in members:
                if pid == anchor:
                    continue
                count = id_to_count.get(pid, 0)
                if count > duplicate_ratio * anchor_count:
                    continue
                mask = (beat == int(beat_id)) & (fused == int(pid))
                pulse_count = int(mask.sum())
                if pulse_count == 0:
                    continue
                fused[mask] = 0
                rows.append(
                    {
                        "beat": int(beat_id),
                        "suppressed_sigidx": int(pid),
                        "anchor_sigidx": int(anchor),
                        "suppressed_pulses": int(pulse_count),
                        "anchor_pulses": int(anchor_count),
                    }
                )

    report = pd.DataFrame(rows)
    meta = {
        "enabled": True,
        "chunk_seconds": float(chunk_seconds),
        "duplicate_ratio": float(duplicate_ratio),
        "min_pri_us": float(min_pri_us),
        "suppressed_segments": int(len(report)),
        "suppressed_pulses": int(report["suppressed_pulses"].sum()) if len(report) else 0,
        "param1_gate": 120.0,
        "doa_gate_deg": 0.30,
        "pri_relative_gate": 0.01,
        "pri_min_gate_us": 0.5,
    }
    return fused, report, meta


def build_generic_param5_abf_groups(report: pd.DataFrame, args: argparse.Namespace) -> List[Tuple[str, pd.DataFrame]]:
    candidates = report[
        (report["median_param4"] > float(args.agile_fusion_min_param4))
        & (report["num_pulses"] >= int(args.agile_fusion_min_batch_pulses))
    ].sort_values("median_param5", kind="mergesort")

    groups: List[Tuple[str, pd.DataFrame]] = []
    current_rows: List[pd.Series] = []
    last_p5: Optional[float] = None
    for _, row in candidates.iterrows():
        p5 = float(row["median_param5"])
        if last_p5 is None or p5 - last_p5 <= float(args.agile_fusion_p5_gap_deg):
            current_rows.append(row)
        else:
            groups.append(("generic_param5_band", pd.DataFrame(current_rows)))
            current_rows = [row]
        last_p5 = p5
    if current_rows:
        groups.append(("generic_param5_band", pd.DataFrame(current_rows)))
    return groups


def apply_agile_band_fusion(spec: RunSpec, args: argparse.Namespace) -> Dict[str, object]:
    if not agile_fusion_enabled(spec, args):
        return {"enabled": False, "reason": "disabled by mode or sample guard"}

    pdw = pd.read_csv(spec.pdw_file, sep=r"\s+", engine="python").iloc[:, :8].copy()
    pdw.columns = ["TOA(s)", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6", "Param7"]
    if args.max_pulses > 0:
        pdw = pdw.iloc[: args.max_pulses].reset_index(drop=True)

    backup = spec.output_dir / f"{spec.name}_sort_before_agile_fusion.txt"
    fusion_input = backup if bool(getattr(args, "skip_sort", False)) and backup.exists() else spec.sort_file
    sigidx = read_sort_sigidx(fusion_input)
    if args.max_pulses > 0:
        sigidx = sigidx[: args.max_pulses]
    if len(pdw) != len(sigidx):
        raise ValueError(f"PDW rows ({len(pdw)}) and SigIdx rows ({len(sigidx)}) do not match.")

    if fusion_input != backup:
        write_sort_sigidx(backup, pdw["TOA(s)"].to_numpy(dtype=np.float64), sigidx)

    report = summarize_pred_batches(pdw, sigidx)
    if len(report) == 0:
        return {"enabled": True, "merged_groups": 0, "reason": "no positive SigIdx"}

    priors = load_strict_train_priors(Path(args.train_dir))
    groups, strict_meta = build_strict_dbscan_groups(report, priors)
    fusion_mode = "strict_train_anchored_dbscan_batch_fusion"
    candidate_count = int(strict_meta.get("candidate_batches", 0.0))

    fused = sigidx.copy()
    merge_rows = []
    for group_index, (group_name, group) in enumerate(groups, start=1):
        ids = [int(v) for v in group["pred_sigidx"].tolist()]
        total = int(group["num_pulses"].sum())
        p5_span = float(group["median_param5"].max() - group["median_param5"].min())
        min_group_pulses = max(2000, int(strict_meta.get("candidate_min_pulses", 0.0)) * 2)
        max_span_deg = 2.50
        if len(ids) < int(args.agile_fusion_min_group_batches):
            continue
        if total < min_group_pulses:
            continue
        if p5_span > max_span_deg:
            continue
        fused_id = min(ids)
        for pred_id in ids:
            fused[sigidx == pred_id] = fused_id
        merge_rows.append(
            {
                "group_index": int(group_index),
                "group_name": str(group_name),
                "fused_sigidx": int(fused_id),
                "merged_sigidx": ",".join(str(v) for v in ids),
                "num_batches": int(len(ids)),
                "num_pulses": int(total),
                "median_param1_min": float(group["median_param1"].min()),
                "median_param1_max": float(group["median_param1"].max()),
                "median_param2_min": float(group["median_param2"].min()),
                "median_param2_max": float(group["median_param2"].max()),
                "median_param5_min": float(group["median_param5"].min()),
                "median_param5_max": float(group["median_param5"].max()),
                "median_param5_span": float(p5_span),
                "median_param4_min": float(group["median_param4"].min()),
                "median_param4_max": float(group["median_param4"].max()),
            }
        )

    agile_report = summarize_pred_batches(pdw, fused)
    agile_groups, agile_meta = build_frequency_agile_pri_groups(agile_report, priors)
    agile_rows = []
    for group_index, (group_name, group) in enumerate(agile_groups, start=1):
        ids = [int(v) for v in group["pred_sigidx"].tolist()]
        fused_id = min(ids)
        for pred_id in ids:
            fused[fused == pred_id] = fused_id
        agile_rows.append(
            {
                "group_index": int(group_index),
                "group_name": str(group_name),
                "fused_sigidx": int(fused_id),
                "merged_sigidx": ",".join(str(v) for v in ids),
                "num_batches": int(len(ids)),
                "num_pulses": int(group["num_pulses"].sum()),
                "median_param1_min": float(group["median_param1"].min()),
                "median_param1_max": float(group["median_param1"].max()),
                "median_param2_min": float(group["median_param2"].min()),
                "median_param2_max": float(group["median_param2"].max()),
                "median_param5_min": float(group["median_param5"].min()),
                "median_param5_max": float(group["median_param5"].max()),
                "median_pri_us_min": float(group["median_pri_us"].min()),
                "median_pri_us_max": float(group["median_pri_us"].max()),
                "median_param4_min": float(group["median_param4"].min()),
                "median_param4_max": float(group["median_param4"].max()),
            }
        )

    amplitude_report = summarize_pred_batches(pdw, fused)
    amplitude_groups, amplitude_meta = build_amplitude_split_pri_groups(amplitude_report)
    amplitude_rows = []
    for group_index, (group_name, group) in enumerate(amplitude_groups, start=1):
        ids = [int(v) for v in group["pred_sigidx"].tolist()]
        fused_id = min(ids)
        for pred_id in ids:
            fused[fused == pred_id] = fused_id
        amplitude_rows.append(
            {
                "group_index": int(group_index),
                "group_name": str(group_name),
                "fused_sigidx": int(fused_id),
                "merged_sigidx": ",".join(str(v) for v in ids),
                "num_batches": int(len(ids)),
                "num_pulses": int(group["num_pulses"].sum()),
                "median_param1_min": float(group["median_param1"].min()),
                "median_param1_max": float(group["median_param1"].max()),
                "median_param2_min": float(group["median_param2"].min()),
                "median_param2_max": float(group["median_param2"].max()),
                "median_param5_min": float(group["median_param5"].min()),
                "median_param5_max": float(group["median_param5"].max()),
                "median_pri_us_min": float(group["median_pri_us"].min()),
                "median_pri_us_max": float(group["median_pri_us"].max()),
                "median_param4_min": float(group["median_param4"].min()),
                "median_param4_max": float(group["median_param4"].max()),
            }
        )

    harmonic_report = summarize_pred_batches(pdw, fused)
    harmonic_groups, harmonic_meta = build_pri_harmonic_groups(harmonic_report)
    harmonic_rows = []
    for group_index, (group_name, group) in enumerate(harmonic_groups, start=1):
        ids = [int(v) for v in group["pred_sigidx"].tolist()]
        fused_id = min(ids)
        for pred_id in ids:
            fused[fused == pred_id] = fused_id
        harmonic_rows.append(
            {
                "group_index": int(group_index),
                "group_name": str(group_name),
                "fused_sigidx": int(fused_id),
                "merged_sigidx": ",".join(str(v) for v in ids),
                "num_batches": int(len(ids)),
                "num_pulses": int(group["num_pulses"].sum()),
                "median_param1_min": float(group["median_param1"].min()),
                "median_param1_max": float(group["median_param1"].max()),
                "median_param2_min": float(group["median_param2"].min()),
                "median_param2_max": float(group["median_param2"].max()),
                "median_param5_min": float(group["median_param5"].min()),
                "median_param5_max": float(group["median_param5"].max()),
                "median_pri_us_min": float(group["median_pri_us"].min()),
                "median_pri_us_max": float(group["median_pri_us"].max()),
                "median_param4_min": float(group["median_param4"].min()),
                "median_param4_max": float(group["median_param4"].max()),
            }
        )

    multimode_report = summarize_pred_batches(pdw, fused)
    multimode_groups, multimode_meta = build_same_carrier_multimode_groups(multimode_report)
    multimode_rows = []
    for group_index, (group_name, group) in enumerate(multimode_groups, start=1):
        ids = [int(v) for v in group["pred_sigidx"].tolist()]
        fused_id = min(ids)
        for pred_id in ids:
            fused[fused == pred_id] = fused_id
        multimode_rows.append(
            {
                "group_index": int(group_index),
                "group_name": str(group_name),
                "fused_sigidx": int(fused_id),
                "merged_sigidx": ",".join(str(v) for v in ids),
                "num_batches": int(len(ids)),
                "num_pulses": int(group["num_pulses"].sum()),
                "median_param1_min": float(group["median_param1"].min()),
                "median_param1_max": float(group["median_param1"].max()),
                "median_param2_min": float(group["median_param2"].min()),
                "median_param2_max": float(group["median_param2"].max()),
                "median_param5_min": float(group["median_param5"].min()),
                "median_param5_max": float(group["median_param5"].max()),
                "median_pri_us_min": float(group["median_pri_us"].min()),
                "median_pri_us_max": float(group["median_pri_us"].max()),
                "median_param4_min": float(group["median_param4"].min()),
                "median_param4_max": float(group["median_param4"].max()),
            }
        )

    stagger_report = summarize_pred_batches(pdw, fused)
    stagger_groups, stagger_meta = build_same_bearing_staggered_pri_groups(stagger_report)
    stagger_rows = []
    for group_index, (group_name, group) in enumerate(stagger_groups, start=1):
        ids = [int(v) for v in group["pred_sigidx"].tolist()]
        fused_id = min(ids)
        for pred_id in ids:
            fused[fused == pred_id] = fused_id
        stagger_rows.append(
            {
                "group_index": int(group_index),
                "group_name": str(group_name),
                "fused_sigidx": int(fused_id),
                "merged_sigidx": ",".join(str(v) for v in ids),
                "num_batches": int(len(ids)),
                "num_pulses": int(group["num_pulses"].sum()),
                "median_param1_min": float(group["median_param1"].min()),
                "median_param1_max": float(group["median_param1"].max()),
                "median_param2_min": float(group["median_param2"].min()),
                "median_param2_max": float(group["median_param2"].max()),
                "median_param5_min": float(group["median_param5"].min()),
                "median_param5_max": float(group["median_param5"].max()),
                "median_pri_us_min": float(group["median_pri_us"].min()),
                "median_pri_us_max": float(group["median_pri_us"].max()),
                "median_param4_min": float(group["median_param4"].min()),
                "median_param4_max": float(group["median_param4"].max()),
            }
        )

    suppression_rows = []
    suppression_report = summarize_pred_batches(pdw, fused)
    suppression_groups, suppress_meta = build_strict_residual_suppression_groups(suppression_report, priors)

    if suppression_groups:
        for group_index, (group_name, group) in enumerate(suppression_groups, start=1):
            ids = [int(v) for v in group["pred_sigidx"].tolist()]
            suppressed_mask = np.isin(fused, ids)
            suppressed_pulses = int(suppressed_mask.sum())
            if suppressed_pulses == 0:
                continue
            fused[suppressed_mask] = 0
            suppression_rows.append(
                {
                    "group_index": int(group_index),
                    "group_name": str(group_name),
                    "suppressed_sigidx": ",".join(str(v) for v in ids),
                    "num_batches": int(len(ids)),
                    "num_pulses": int(suppressed_pulses),
                    "median_param1_min": float(group["median_param1"].min()),
                    "median_param1_max": float(group["median_param1"].max()),
                    "median_param2_min": float(group["median_param2"].min()),
                    "median_param2_max": float(group["median_param2"].max()),
                    "median_param5_min": float(group["median_param5"].min()),
                    "median_param5_max": float(group["median_param5"].max()),
                    "median_param4_min": float(group["median_param4"].min()),
                    "median_param4_max": float(group["median_param4"].max()),
                    }
                )

    fused, sparse_meta = suppress_sparse_beat_segments(
        pdw,
        fused,
        float(args.sort_chunk_seconds),
        int(args.min_beat_batch_pulses),
        int(args.hetero_beat_max_pulses),
        float(args.hetero_beat_param1_iqr),
        float(args.hetero_beat_doa_iqr_deg),
    )
    fused, duplicate_df, duplicate_meta = suppress_duplicate_beat_fragments(
        pdw,
        fused,
        float(args.sort_chunk_seconds),
        float(args.duplicate_beat_fragment_ratio),
        float(args.duplicate_beat_min_pri_us),
    )
    post_duplicate_stagger_report = summarize_pred_batches(pdw, fused)
    post_duplicate_stagger_groups, post_duplicate_stagger_meta = build_same_bearing_staggered_pri_groups(
        post_duplicate_stagger_report
    )
    post_duplicate_stagger_rows = []
    for group_index, (group_name, group) in enumerate(post_duplicate_stagger_groups, start=1):
        ids = [int(v) for v in group["pred_sigidx"].tolist()]
        group_doa = float(group["median_param5"].median())
        outside = post_duplicate_stagger_report.loc[~post_duplicate_stagger_report["pred_sigidx"].isin(ids)]
        if len(outside) > 0:
            outside_doa = outside["median_param5"].to_numpy(dtype=np.float64)
            near_outside = outside.loc[circular_degrees_delta(outside_doa, group_doa) <= 0.90]
            if int(near_outside["num_pulses"].sum()) > 0.25 * int(group["num_pulses"].sum()):
                continue
        fused_id = min(ids)
        for pred_id in ids:
            fused[fused == pred_id] = fused_id
        post_duplicate_stagger_rows.append(
            {
                "group_index": int(group_index),
                "group_name": str(group_name),
                "fused_sigidx": int(fused_id),
                "merged_sigidx": ",".join(str(v) for v in ids),
                "num_batches": int(len(ids)),
                "num_pulses": int(group["num_pulses"].sum()),
                "median_param1_min": float(group["median_param1"].min()),
                "median_param1_max": float(group["median_param1"].max()),
                "median_param2_min": float(group["median_param2"].min()),
                "median_param2_max": float(group["median_param2"].max()),
                "median_param5_min": float(group["median_param5"].min()),
                "median_param5_max": float(group["median_param5"].max()),
                "median_pri_us_min": float(group["median_pri_us"].min()),
                "median_pri_us_max": float(group["median_pri_us"].max()),
                "median_param4_min": float(group["median_param4"].min()),
                "median_param4_max": float(group["median_param4"].max()),
            }
        )

    merge_report = pd.DataFrame(merge_rows)
    agile_df = pd.DataFrame(agile_rows)
    amplitude_df = pd.DataFrame(amplitude_rows)
    harmonic_df = pd.DataFrame(harmonic_rows)
    multimode_df = pd.DataFrame(multimode_rows)
    stagger_df = pd.DataFrame(stagger_rows)
    post_duplicate_stagger_df = pd.DataFrame(post_duplicate_stagger_rows)
    suppression_df = pd.DataFrame(suppression_rows)
    merge_report.to_csv(spec.output_dir / f"{spec.name}_agile_fusion_report.csv", index=False, encoding="utf-8-sig")
    agile_df.to_csv(
        spec.output_dir / f"{spec.name}_frequency_agile_pri_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    amplitude_df.to_csv(
        spec.output_dir / f"{spec.name}_amplitude_split_pri_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    harmonic_df.to_csv(
        spec.output_dir / f"{spec.name}_pri_harmonic_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    multimode_df.to_csv(
        spec.output_dir / f"{spec.name}_same_carrier_multimode_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stagger_df.to_csv(
        spec.output_dir / f"{spec.name}_same_bearing_staggered_pri_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    suppression_df.to_csv(
        spec.output_dir / f"{spec.name}_agile_suppression_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    duplicate_df.to_csv(
        spec.output_dir / f"{spec.name}_duplicate_beat_fragment_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    post_duplicate_stagger_df.to_csv(
        spec.output_dir / f"{spec.name}_post_duplicate_staggered_pri_report.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_sort_sigidx(spec.sort_file, pdw["TOA(s)"].to_numpy(dtype=np.float64), fused)
    metrics = evaluate_sort_sigidx(
        pdw,
        spec.truth_file,
        fused,
        args,
        spec.output_dir / "agile_fusion_metrics" / f"{spec.name}_agile_fusion",
    )
    return {
        "enabled": True,
        "mode": fusion_mode,
        "fusion_input": str(fusion_input),
        "input_backup": str(backup),
        "output_file": str(spec.sort_file),
        "candidate_batches": candidate_count,
        "merged_groups": int(len(merge_report)),
        "merged_batches": int(sum(len(str(row["merged_sigidx"]).split(",")) for _, row in merge_report.iterrows())),
        "frequency_agile_groups": int(len(agile_df)),
        "frequency_agile_batches": int(sum(len(str(row["merged_sigidx"]).split(",")) for _, row in agile_df.iterrows())),
        "amplitude_split_groups": int(len(amplitude_df)),
        "amplitude_split_batches": int(sum(len(str(row["merged_sigidx"]).split(",")) for _, row in amplitude_df.iterrows())),
        "pri_harmonic_groups": int(len(harmonic_df)),
        "pri_harmonic_batches": int(sum(len(str(row["merged_sigidx"]).split(",")) for _, row in harmonic_df.iterrows())),
        "same_carrier_multimode_groups": int(len(multimode_df)),
        "same_carrier_multimode_batches": int(sum(len(str(row["merged_sigidx"]).split(",")) for _, row in multimode_df.iterrows())),
        "same_bearing_staggered_pri_groups": int(len(stagger_df)),
        "same_bearing_staggered_pri_batches": int(sum(len(str(row["merged_sigidx"]).split(",")) for _, row in stagger_df.iterrows())),
        "post_duplicate_staggered_pri_groups": int(len(post_duplicate_stagger_df)),
        "post_duplicate_staggered_pri_batches": int(sum(len(str(row["merged_sigidx"]).split(",")) for _, row in post_duplicate_stagger_df.iterrows())),
        "suppressed_groups": int(len(suppression_df)),
        "suppressed_batches": int(sum(int(row["num_batches"]) for _, row in suppression_df.iterrows())),
        "suppressed_pulses": int(sum(int(row["num_pulses"]) for _, row in suppression_df.iterrows())),
        "p5_gap_deg": float(args.agile_fusion_p5_gap_deg),
        "min_param4": float(args.agile_fusion_min_param4),
        "strict_train_dir": str(args.train_dir),
        "strict_meta": strict_meta,
        "frequency_agile_pri_meta": agile_meta,
        "amplitude_split_pri_meta": amplitude_meta,
        "pri_harmonic_meta": harmonic_meta,
        "same_carrier_multimode_meta": multimode_meta,
        "same_bearing_staggered_pri_meta": stagger_meta,
        "post_duplicate_staggered_pri_meta": post_duplicate_stagger_meta,
        "strict_suppression_meta": suppress_meta,
        "sparse_beat_cleanup_meta": sparse_meta,
        "duplicate_beat_fragment_meta": duplicate_meta,
        "metrics": metrics,
    }


def configure_sort_args(
    parser: argparse.ArgumentParser,
    spec: RunSpec,
    args: argparse.Namespace,
    backend: str,
) -> argparse.Namespace:
    sort_args = parser.parse_args([])
    sort_args.input_file = spec.pdw_file
    sort_args.truth_file = spec.truth_file or Path("__missing_truth__.txt")
    sort_args.output_file = spec.sort_file
    sort_args.raw_output_file = spec.raw_sort_file
    sort_args.report_json = spec.sort_report
    sort_args.metrics_dir = spec.metrics_dir
    sort_args.window_report_csv = spec.output_dir / f"{spec.name}_windows.csv"
    sort_args.edge_report_csv = spec.output_dir / f"{spec.name}_edges.csv"
    sort_args.max_pulses = int(args.max_pulses)
    sort_args.skip_metrics = not bool(spec.truth_file and spec.truth_file.exists())
    sort_args.seed = int(args.seed)
    sort_args.n_jobs = int(args.n_jobs)
    sort_args.verbose = bool(args.verbose)
    if hasattr(args, "window_seconds") and hasattr(sort_args, "window_seconds"):
        sort_args.window_seconds = float(args.window_seconds)
    if hasattr(args, "progress_every") and hasattr(sort_args, "progress_every"):
        sort_args.progress_every = int(args.progress_every)

    # Conservative performance-oriented defaults from the best previous runs.
    if hasattr(sort_args, "emit_label99"):
        sort_args.emit_label99 = True
    if hasattr(sort_args, "hdbscan_backend"):
        sort_args.hdbscan_backend = "auto"
    if hasattr(sort_args, "allow_optics_fallback"):
        sort_args.allow_optics_fallback = True

    if backend == "pa_tgr":
        sort_args.tgr_pre_reduce = True
        sort_args.tgr_noise_rescue = True
        sort_args.tgr_mutual_nearest = True
        sort_args.tgr_check_component = True
        sort_args.tgr_link_thresh = float(args.tgr_link_thresh)
        sort_args.tgr_hard_gate = float(args.tgr_hard_gate)
        sort_args.tgr_tail_cleanup = True
        sort_args.tgr_thin_cleanup = True
    elif backend == "pa_tsr":
        sort_args.tsr_band_split_auto = True
        sort_args.tsr_reduce_merge_thresh = float(args.tsr_reduce_merge_thresh)
        sort_args.tsr_rescue_merge_thresh = float(args.tsr_rescue_merge_thresh)

    if args.reuse_sort_if_exists and spec.sort_file.exists():
        if backend == "pa_tgr" and hasattr(sort_args, "use_existing_raw_output"):
            # PA-TGR can reuse its raw front-end, but the final sort file is
            # handled one level up. This flag is useful when raw_sort_file exists.
            sort_args.use_existing_raw_output = spec.raw_sort_file.exists()
    return sort_args


def run_sorting(spec: RunSpec, args: argparse.Namespace) -> Dict[str, object]:
    if args.reuse_sort_if_exists and spec.sort_file.exists():
        summary = {
            "method": "reuse_existing_sort",
            "final_output_file": str(spec.sort_file),
            "skipped": True,
        }
        fusion = apply_agile_band_fusion(spec, args)
        return {**summary, "agile_fusion": fusion}

    backend = args.sort_backend
    if backend == "auto":
        backend = "pa_tsr"

    sorter, run_fn = import_sort_backend(backend)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    sort_args = configure_sort_args(sorter.build_parser(), spec, args, backend)
    print(f"[sort] {spec.name}: backend={backend}, input={spec.pdw_file}")
    summary = run_fn(sort_args)
    fusion = apply_agile_band_fusion(spec, args)
    return {"backend": backend, **dict(summary), "agile_fusion": fusion}


def configure_recognition_args(spec: RunSpec, args: argparse.Namespace):
    import xgb_knn_conformal_ood as recog

    rec_args = recog.build_parser().parse_args([])
    rec_args.pdw_file = spec.pdw_file
    rec_args.sorted_file = spec.sort_file
    rec_args.truth_file = spec.truth_file or Path("__missing_truth__.txt")
    rec_args.train_dir = Path(args.train_dir)
    rec_args.output_dir = spec.recognition_dir
    rec_args.output_file = spec.final_file
    rec_args.max_pulses = int(args.max_pulses)
    rec_args.train_ratio = float(args.train_ratio)
    rec_args.recognition_feature_mode = "sort_aware"
    rec_args.recognition_angle_mode = "raw"
    rec_args.unknown_label = int(args.unknown_label)
    rec_args.n_estimators = int(args.n_estimators)
    rec_args.max_depth = int(args.max_depth)
    rec_args.learning_rate = float(args.learning_rate)
    rec_args.subsample = float(args.subsample)
    rec_args.colsample_bytree = float(args.colsample_bytree)
    rec_args.reg_lambda = float(args.reg_lambda)
    rec_args.min_child_weight = float(args.min_child_weight)
    rec_args.early_stopping_rounds = int(args.early_stopping_rounds)
    rec_args.n_jobs = int(args.n_jobs)
    rec_args.train_verbose = int(args.train_verbose)
    rec_args.seed = int(args.seed)

    rec_args.batch_prob_threshold = float(args.batch_prob_threshold)
    rec_args.ood_distance_multiplier = float(args.ood_distance_multiplier)
    rec_args.ood_gate_labels = str(args.ood_gate_labels)

    rec_args.prob_threshold_grid = args.prob_threshold_grid
    rec_args.ood_alpha = float(args.ood_alpha)
    rec_args.ood_alpha_grid = args.ood_alpha_grid
    rec_args.ood_distance_quantile = float(args.ood_distance_quantile)
    rec_args.ood_distance_multiplier_grid = args.ood_distance_multiplier_grid
    rec_args.ood_gate_label_grid = args.ood_gate_label_grid
    rec_args.ood_k = int(args.ood_k)
    rec_args.ood_metric = args.ood_metric
    rec_args.ood_feature_mode = args.ood_feature_mode
    rec_args.ood_include_size_features = bool(args.ood_include_size_features)
    rec_args.ood_chunk_sizes = args.ood_chunk_sizes
    rec_args.ood_stride_fraction = float(args.ood_stride_fraction)
    rec_args.tune_with_truth = bool(args.tune_with_truth and spec.truth_file and spec.truth_file.exists())
    rec_args.enable_track_repair = bool(args.enable_track_repair)
    return recog, rec_args


def run_recognition(spec: RunSpec, args: argparse.Namespace) -> Dict[str, object]:
    recog, rec_args = configure_recognition_args(spec, args)
    recog.e2e.set_seed(rec_args.seed)
    rec_args.ood_chunk_sizes = recog.parse_int_list(rec_args.ood_chunk_sizes)
    rec_args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[recog] {spec.name}: train={rec_args.train_dir}, sorted={spec.sort_file}")
    df = recog.dbscan.read_pdw(rec_args.pdw_file)
    pred_sigidx = recog.read_sigidx(rec_args.sorted_file)
    if rec_args.max_pulses > 0:
        df = df.iloc[: rec_args.max_pulses].reset_index(drop=True)
        pred_sigidx = pred_sigidx[: rec_args.max_pulses]
    if len(df) != len(pred_sigidx):
        raise ValueError(f"PDW rows ({len(df)}) and SigIdx rows ({len(pred_sigidx)}) do not match.")

    repair_report = pd.DataFrame()
    if rec_args.enable_track_repair:
        pred_sigidx, repair_report = recog.conservative_track_repair(df, pred_sigidx, rec_args)

    pulse = recog.train_pulse_xgboost(rec_args)
    train_ood_x, train_ood_y, train_chunk_report = recog.build_train_ood_samples(rec_args, pulse)
    ood_scaler, gates, ood_gate_report = recog.fit_ood_gates(
        train_ood_x,
        train_ood_y,
        pulse["known_labels"],
        rec_args,
    )
    test_batches = recog.build_test_batch_table(df, pred_sigidx, pulse, ood_scaler, gates, rec_args)
    test_batches.to_csv(rec_args.output_dir / "xgb_knn_ood_scores_raw.csv", index=False, encoding="utf-8-sig")

    truth_exists = rec_args.truth_file.exists()
    truth = recog.dbscan.read_truth_file(rec_args.truth_file) if truth_exists else pd.DataFrame()
    if truth_exists and rec_args.max_pulses > 0:
        truth = truth.iloc[: rec_args.max_pulses].reset_index(drop=True)
    if truth_exists and len(truth) != len(pred_sigidx):
        raise ValueError(f"Truth rows ({len(truth)}) and SigIdx rows ({len(pred_sigidx)}) do not match.")

    reports: Dict[str, pd.DataFrame] = {
        "xgb_knn_ood_train_chunks": train_chunk_report,
        "xgb_knn_ood_gate_report": ood_gate_report,
        "xgb_knn_ood_track_repair": repair_report,
    }

    used_truth_for_tuning = False
    if truth_exists and rec_args.tune_with_truth:
        tune_df, best = recog.tune_thresholds(truth, pred_sigidx, test_batches, rec_args)
        reports["xgb_knn_ood_threshold_tuning"] = tune_df.sort_values(
            ["recognition_acc", "ood_alpha", "prob_threshold"],
            ascending=[False, True, True],
        )
        prob_threshold = float(best["prob_threshold"])
        ood_alpha = float(best["ood_alpha"])
        distance_multiplier = float(best["ood_distance_multiplier"])
        gate_label_text = str(best["ood_gate_labels"])
        used_truth_for_tuning = True
    else:
        prob_threshold = float(rec_args.batch_prob_threshold)
        ood_alpha = float(rec_args.ood_alpha)
        distance_multiplier = float(rec_args.ood_distance_multiplier)
        gate_label_text = recog.normalize_label_set_text(rec_args.ood_gate_labels)

    best_batch_df = recog.apply_decision(
        test_batches,
        prob_threshold,
        ood_alpha,
        distance_multiplier,
        gate_label_text,
        rec_args.unknown_label,
    )
    labels = recog.labels_from_batch_df(pred_sigidx, best_batch_df, rec_args.unknown_label)

    metrics_report: Dict[str, object] = {"skipped": True}
    if truth_exists:
        metrics_report, batch_eval, target_df, target_beat_df, beat_df = recog.evaluate_batch_labels(
            truth,
            pred_sigidx,
            best_batch_df,
            rec_args,
        )
        reports["xgb_knn_ood_sort_batch_eval"] = batch_eval
        reports["xgb_knn_ood_target_accuracy"] = target_df
        reports["xgb_knn_ood_target_beat_eval"] = target_beat_df
        reports["xgb_knn_ood_beat_eval"] = beat_df

    summary = {
        "pdw_file": str(rec_args.pdw_file),
        "sorted_file": str(rec_args.sorted_file),
        "output_file": str(rec_args.output_file),
        "train_dir": str(rec_args.train_dir),
        "max_pulses": int(rec_args.max_pulses),
        "known_labels": pulse["known_labels"],
        "unknown_label": int(rec_args.unknown_label),
        "pulse_val_acc": float(pulse["val_acc"]),
        "best_iteration": pulse["best_iteration"],
        "decision": "xgboost_batch_mean_plus_class_conditional_knn_conformal_ood",
        "prob_threshold": float(prob_threshold),
        "ood_alpha": float(ood_alpha),
        "ood_k": int(rec_args.ood_k),
        "ood_metric": str(rec_args.ood_metric),
        "ood_distance_quantile": float(rec_args.ood_distance_quantile),
        "ood_distance_multiplier": float(distance_multiplier),
        "ood_gate_labels": recog.normalize_label_set_text(gate_label_text),
        "ood_feature_mode": str(rec_args.ood_feature_mode),
        "ood_include_size_features": bool(rec_args.ood_include_size_features),
        "ood_chunk_sizes": [int(v) for v in rec_args.ood_chunk_sizes],
        "used_truth_for_tuning": bool(used_truth_for_tuning),
        "track_repair_enabled": bool(rec_args.enable_track_repair),
        "track_repair_count": int(len(repair_report)),
        "metrics": metrics_report,
    }
    recog.write_outputs(df, pred_sigidx, labels, best_batch_df, summary, reports, rec_args)
    return summary


def composite_score(metrics: Dict[str, object]) -> Optional[float]:
    try:
        sort_acc = float(metrics["sort_acc"])
        extra = float(metrics["extra_batch_rate"])
        wrong = float(metrics["wrong_batch_rate"])
        recog = float(metrics["recognition_acc"])
        stable = float(metrics["signal_tracking_stability"])
    except Exception:
        return None
    return 0.30 * sort_acc + 0.20 * (1.0 - extra) + 0.10 * (1.0 - wrong) + 0.10 * recog + 0.20 * stable


def write_master_summary(spec: RunSpec, sort_summary: Dict[str, object], recog_summary: Dict[str, object]) -> None:
    metrics = dict(recog_summary.get("metrics") or {})
    score = composite_score(metrics)
    summary = {
        "sample": spec.name,
        "pdw_file": str(spec.pdw_file),
        "truth_file": str(spec.truth_file) if spec.truth_file else None,
        "sort_file": str(spec.sort_file),
        "final_file": str(spec.final_file),
        "composite_score_no_time": score,
        "sort_summary": sort_summary,
        "recognition_summary": recog_summary,
    }
    out = spec.output_dir / f"{spec.name}_performance_first_summary.json"
    out.write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def print_metrics_line(spec: RunSpec, summary: Dict[str, object]) -> None:
    metrics = summary.get("metrics") or {}
    score = composite_score(metrics)
    if not score:
        print(f"[done] {spec.name}: final={spec.final_file}")
        return
    print(
        "[done] "
        f"{spec.name}: score={score:.4f}, "
        f"sort={metrics['sort_acc']:.4f}, "
        f"extra={metrics['extra_batch_rate']:.4f}, "
        f"wrong={metrics['wrong_batch_rate']:.4f}, "
        f"recog={metrics['recognition_acc']:.4f}, "
        f"track={metrics['signal_tracking_stability']:.4f}"
    )
    print(f"       final={spec.final_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-file performance-first PDW sorting + recognition pipeline."
    )
    parser.add_argument("--sample", choices=["sample1", "sample2", "all", "custom"], default="all")
    parser.add_argument("--pdw_file", type=Path, default=r"C:\Users\ZHT\Desktop\实验\分选\edata\Test_Data\Sample_1\Merge_PDW_Data.txt")
    parser.add_argument("--truth_file", type=Path, default=r"C:\Users\ZHT\Desktop\实验\分选\edata\Test_Data\Sample_1\Sorted_PDW.txt")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--train_dir", type=Path, default=Path("edata/Train_Data"))
    parser.add_argument("--output_root", type=Path, default=Path("outputs_performance_first"))
    parser.add_argument("--max_pulses", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--window_seconds", type=float, default=0.1)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--sort_backend", choices=["auto", "pa_tgr", "pa_tsr"], default="auto")
    parser.add_argument("--reuse_sort_if_exists", action="store_true")
    parser.add_argument("--skip_sort", action="store_true")
    parser.add_argument("--skip_recognition", action="store_true")
    parser.add_argument("--tgr_link_thresh", type=float, default=2.6)
    parser.add_argument("--tgr_hard_gate", type=float, default=1.5)
    parser.add_argument("--tsr_reduce_merge_thresh", type=float, default=0.2)
    parser.add_argument("--tsr_rescue_merge_thresh", type=float, default=0.75)
    parser.add_argument("--agile_fusion", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--agile_fusion_p5_gap_deg", type=float, default=0.45)
    parser.add_argument("--agile_fusion_max_span_deg", type=float, default=0.80)
    parser.add_argument("--agile_fusion_min_param4", type=float, default=-55.0)
    parser.add_argument("--agile_fusion_min_batch_pulses", type=int, default=300)
    parser.add_argument("--agile_fusion_min_group_pulses", type=int, default=10000)
    parser.add_argument("--agile_fusion_min_group_batches", type=int, default=2)
    parser.add_argument("--min_beat_batch_pulses", type=int, default=1)
    parser.add_argument("--hetero_beat_max_pulses", type=int, default=50)
    parser.add_argument("--hetero_beat_param1_iqr", type=float, default=10.0)
    parser.add_argument("--hetero_beat_doa_iqr_deg", type=float, default=0.5)
    parser.add_argument("--duplicate_beat_fragment_ratio", type=float, default=0.40)
    parser.add_argument("--duplicate_beat_min_pri_us", type=float, default=80.0)
    parser.add_argument("--sort_purity_threshold", type=float, default=0.90)
    parser.add_argument("--sort_min_target_fraction", type=float, default=0.10)
    parser.add_argument("--sort_mix_fail_min_pulses", type=int, default=150)
    parser.add_argument("--sort_chunk_seconds", type=float, default=0.2)

    parser.add_argument("--unknown_label", type=int, default=UNKNOWN_LABEL)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--n_estimators", type=int, default=2000)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=1.0)
    parser.add_argument("--reg_lambda", type=float, default=1.0)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=30)
    parser.add_argument("--train_verbose", type=int, choices=[0, 1], default=0)

    parser.add_argument("--batch_prob_threshold", type=float, default=0.92)
    parser.add_argument("--prob_threshold_grid", type=str, default="0.80,0.85,0.88,0.90,0.92,0.94,0.96,0.98,0.99")
    parser.add_argument("--ood_alpha", type=float, default=0.0)
    parser.add_argument("--ood_alpha_grid", type=str, default="0.0")
    parser.add_argument("--ood_distance_quantile", type=float, default=0.99)
    parser.add_argument("--ood_distance_multiplier", type=float, default=5.0)
    parser.add_argument("--ood_distance_multiplier_grid", type=str, default="0.0,2.0,3.0,4.0,5.0,6.0,8.0,10.0,12.0")
    parser.add_argument("--ood_gate_labels", type=str, default="3,4")
    parser.add_argument("--ood_gate_label_grid", type=str, default="3,4;2,3,4;3;4;all")
    parser.add_argument("--ood_k", type=int, default=5)
    parser.add_argument("--ood_metric", choices=["euclidean", "manhattan", "cosine"], default="euclidean")
    parser.add_argument("--ood_feature_mode", choices=["stats", "stats_prob"], default="stats_prob")
    parser.add_argument("--ood_include_size_features", action="store_true")
    parser.add_argument("--ood_chunk_sizes", type=str, default="512,1024,2048,4096,8192")
    parser.add_argument("--ood_stride_fraction", type=float, default=1.0)
    parser.add_argument("--tune_with_truth", dest="tune_with_truth", action="store_true", default=True)
    parser.add_argument("--no_tune_with_truth", dest="tune_with_truth", action="store_false")
    parser.add_argument("--enable_track_repair", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    specs = resolve_run_specs(args)
    if not specs:
        raise ValueError("No run specs resolved.")

    for spec in specs:
        if not spec.pdw_file.exists():
            raise FileNotFoundError(f"Missing PDW file: {spec.pdw_file}")
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print(f"Performance-first run: {spec.name}")
        print(f"PDW   : {spec.pdw_file}")
        print(f"Truth : {spec.truth_file if spec.truth_file else 'not found'}")

        sort_summary: Dict[str, object] = {"skipped": True}
        if args.skip_sort:
            if not spec.sort_file.exists():
                raise FileNotFoundError(f"--skip_sort requested but sort file does not exist: {spec.sort_file}")
            print(f"[sort] skipped, using {spec.sort_file}")
            sort_summary["agile_fusion"] = apply_agile_band_fusion(spec, args)
        else:
            sort_summary = run_sorting(spec, args)

        if args.skip_recognition:
            print(f"[recog] skipped; sort output={spec.sort_file}")
            continue

        recog_summary = run_recognition(spec, args)
        write_master_summary(spec, sort_summary, recog_summary)
        print_metrics_line(spec, recog_summary)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-process PDW sorting results by merging similar SigIdx tracks.

This script is meant for the case where an online or clustering sorter creates
too many temporary SigIdx values. It keeps the original sorting result as input,
then applies two conservative reductions:

1. very small tracks can be mapped to noise SigIdx=0;
2. tracks with close robust parameter centers can be merged;
3. final batches smaller than a sample-level pulse fraction can be merged into
   the nearest remaining batch.

The method reduces category count through a reproducible rule instead of
forcing all pulses into a fixed class.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


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

MERGE_FEATURES = ["Param1", "Param2", "Param4", "Param5"]


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
        raise ValueError(f"PDW file contains {bad_rows} rows with non-numeric values after parsing: {path}")
    return df.reset_index(drop=True)


def read_sorted(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    if "SigIdx" not in df.columns:
        if df.shape[1] < 2:
            raise ValueError(f"Sorted file must contain a SigIdx column or at least two columns: {path}")
        df = df.copy()
        df.columns = list(df.columns[:-1]) + ["SigIdx"]
    df["SigIdx"] = pd.to_numeric(df["SigIdx"], errors="coerce")
    bad_rows = int(df["SigIdx"].isna().sum())
    if bad_rows:
        raise ValueError(f"Sorted file contains {bad_rows} non-numeric SigIdx values: {path}")
    df["SigIdx"] = df["SigIdx"].astype(np.int64)
    return df.reset_index(drop=True)


def circular_mean_deg(values: np.ndarray) -> float:
    radians = np.deg2rad(np.mod(values.astype(np.float64), 360.0))
    sin_mean = float(np.sin(radians).mean())
    cos_mean = float(np.cos(radians).mean())
    return float((np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0)


def build_track_centers(pdw: pd.DataFrame, sigidx: np.ndarray) -> pd.DataFrame:
    work = pdw[MERGE_FEATURES].copy()
    work["_sigidx"] = sigidx.astype(np.int64)
    work = work[work["_sigidx"] > 0]
    if len(work) == 0:
        return pd.DataFrame(columns=MERGE_FEATURES + ["count"])

    grouped = work.groupby("_sigidx", sort=True)
    centers = grouped[["Param1", "Param2", "Param4"]].median()
    centers["Param5"] = grouped["Param5"].apply(lambda s: circular_mean_deg(s.to_numpy(dtype=np.float64)))
    centers["count"] = grouped.size().astype(np.int64)
    return centers


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


def circular_abs_diff_deg(values: np.ndarray, ref: float) -> np.ndarray:
    return np.abs((values - ref + 180.0) % 360.0 - 180.0)


def iter_merge_pairs(
    arr: np.ndarray,
    merge_thresh: float,
    weights: Tuple[float, float, float, float],
    tolerances: Tuple[float, float, float, float],
    hard_gates: Tuple[float, float, float, float],
) -> Iterable[Tuple[int, int]]:
    w_p1, w_p2, w_p4, w_p5 = weights
    tol_p1, tol_p2, tol_p4, tol_p5 = tolerances
    gate_p1, gate_p2, gate_p4, gate_p5 = hard_gates

    n = len(arr)
    for i in range(n - 1):
        rest = arr[i + 1 :]
        d_p1 = np.abs(rest[:, 0] - arr[i, 0]) / max(tol_p1, 1e-12)
        d_p2 = np.abs(rest[:, 1] - arr[i, 1]) / max(tol_p2, 1e-12)
        d_p4 = np.abs(rest[:, 2] - arr[i, 2]) / max(tol_p4, 1e-12)
        d_p5 = circular_abs_diff_deg(rest[:, 3], arr[i, 3]) / max(tol_p5, 1e-12)

        hard_ok = (d_p1 <= gate_p1) & (d_p2 <= gate_p2) & (d_p4 <= gate_p4) & (d_p5 <= gate_p5)
        dist = w_p1 * d_p1 + w_p2 * d_p2 + w_p4 * d_p4 + w_p5 * d_p5
        matches = np.flatnonzero(hard_ok & (dist <= merge_thresh))
        for offset in matches:
            yield i, int(i + 1 + offset)


def merge_similar_sigidx(
    centers: pd.DataFrame,
    merge_thresh: float,
    weights: Tuple[float, float, float, float],
    tolerances: Tuple[float, float, float, float],
    hard_gates: Tuple[float, float, float, float],
) -> Dict[int, int]:
    if len(centers) == 0:
        return {}

    ids = centers.index.to_numpy(dtype=np.int64)
    arr = centers[MERGE_FEATURES].to_numpy(dtype=np.float64)
    uf = UnionFind(len(ids))

    for a, b in iter_merge_pairs(arr, merge_thresh, weights, tolerances, hard_gates):
        uf.union(a, b)

    root_to_min_sig: Dict[int, int] = {}
    for pos, sig in enumerate(ids):
        root = uf.find(pos)
        root_to_min_sig[root] = min(root_to_min_sig.get(root, int(sig)), int(sig))

    return {int(sig): root_to_min_sig[uf.find(pos)] for pos, sig in enumerate(ids)}


def relabel_dense(sigidx: np.ndarray) -> np.ndarray:
    out = sigidx.copy()
    valid_ids = sorted(int(v) for v in np.unique(out) if v > 0)
    mapping = {old: new for new, old in enumerate(valid_ids, start=1)}
    for old, new in mapping.items():
        out[out == old] = new
    return out


def _merge_two_centers(target: pd.Series, source: pd.Series) -> pd.Series:
    target_count = float(target["count"])
    source_count = float(source["count"])
    total = target_count + source_count
    merged = target.copy()

    for col in ["Param1", "Param2", "Param4"]:
        merged[col] = (float(target[col]) * target_count + float(source[col]) * source_count) / total

    target_angle = np.deg2rad(float(target["Param5"]))
    source_angle = np.deg2rad(float(source["Param5"]))
    sin_sum = np.sin(target_angle) * target_count + np.sin(source_angle) * source_count
    cos_sum = np.cos(target_angle) * target_count + np.cos(source_angle) * source_count
    merged["Param5"] = float((np.rad2deg(np.arctan2(sin_sum, cos_sum)) + 360.0) % 360.0)
    merged["count"] = int(total)
    return merged


def merge_small_batches_by_fraction(
    sigidx: np.ndarray,
    centers: pd.DataFrame,
    min_fraction: float,
    weights: Tuple[float, float, float, float],
    tolerances: Tuple[float, float, float, float],
    hard_gates: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, int, int, int]:
    if min_fraction <= 0:
        return sigidx.copy(), 0, 0, 0

    min_count = int(np.ceil(len(sigidx) * min_fraction))
    if len(centers) == 0:
        return sigidx.copy(), 0, min_count, 0

    out = sigidx.copy()
    working = centers.copy()
    parent = {int(sig): int(sig) for sig in working.index}
    merge_count = 0
    fallback_count = 0

    w_p1, w_p2, w_p4, w_p5 = weights
    tol_p1, tol_p2, tol_p4, tol_p5 = tolerances
    gate_p1, gate_p2, gate_p4, gate_p5 = hard_gates

    while len(working) > 1:
        small = working[working["count"] < min_count]
        if len(small) == 0:
            break

        source_id = int(small.sort_values(["count"]).index[0])
        source = working.loc[source_id]
        candidates = working.drop(index=source_id)

        d_p1 = np.abs(candidates["Param1"].to_numpy(dtype=np.float64) - float(source["Param1"])) / max(tol_p1, 1e-12)
        d_p2 = np.abs(candidates["Param2"].to_numpy(dtype=np.float64) - float(source["Param2"])) / max(tol_p2, 1e-12)
        d_p4 = np.abs(candidates["Param4"].to_numpy(dtype=np.float64) - float(source["Param4"])) / max(tol_p4, 1e-12)
        d_p5 = circular_abs_diff_deg(candidates["Param5"].to_numpy(dtype=np.float64), float(source["Param5"])) / max(tol_p5, 1e-12)

        dist = w_p1 * d_p1 + w_p2 * d_p2 + w_p4 * d_p4 + w_p5 * d_p5
        hard_ok = (d_p1 <= gate_p1) & (d_p2 <= gate_p2) & (d_p4 <= gate_p4) & (d_p5 <= gate_p5)
        candidate_ids = candidates.index.to_numpy(dtype=np.int64)

        if np.any(hard_ok):
            target_id = int(candidate_ids[np.flatnonzero(hard_ok)[int(np.argmin(dist[hard_ok]))]])
        else:
            target_id = int(candidate_ids[int(np.argmin(dist))])
            fallback_count += 1

        parent[source_id] = target_id
        working.loc[target_id] = _merge_two_centers(working.loc[target_id], source)
        working = working.drop(index=source_id)
        merge_count += 1

    def find(sig: int) -> int:
        path = []
        while parent[sig] != sig:
            path.append(sig)
            sig = parent[sig]
        for item in path:
            parent[item] = sig
        return sig

    for old_id in sorted(parent):
        final_id = find(old_id)
        if final_id != old_id:
            out[sigidx == old_id] = final_id

    return out, merge_count, min_count, fallback_count


def build_split_features(
    pdw: pd.DataFrame,
    tolerances: Tuple[float, float, float, float],
) -> np.ndarray:
    tol_p1, tol_p2, tol_p4, _ = tolerances
    angle = np.deg2rad(np.mod(pdw["Param5"].to_numpy(dtype=np.float64), 360.0))
    feature_parts = [
        pdw["Param1"].to_numpy(dtype=np.float64) / max(tol_p1, 1e-12),
        pdw["Param2"].to_numpy(dtype=np.float64) / max(tol_p2, 1e-12),
    ]
    if "Param3" in pdw.columns:
        feature_parts.append(pdw["Param3"].to_numpy(dtype=np.float64))
    feature_parts.extend(
        [
            pdw["Param4"].to_numpy(dtype=np.float64) / max(tol_p4, 1e-12),
            np.sin(angle),
            np.cos(angle),
        ]
    )
    if "Param6" in pdw.columns:
        feature_parts.append(pdw["Param6"].to_numpy(dtype=np.float64))
    if "Param7" in pdw.columns:
        feature_parts.append(pdw["Param7"].to_numpy(dtype=np.float64))
    return np.column_stack(
        feature_parts
    ).astype(np.float32)


def split_large_batches_by_kmeans(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    tolerances: Tuple[float, float, float, float],
    split_k: int = 8,
    max_k: int = 8,
    min_count: int = 5000,
    min_child_count: int = 500,
    sample_size: int = 12000,
    min_silhouette: float = 0.25,
    random_state: int = 1234,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Split large SigIdx batches in PDW feature space.

    split_k > 1 uses that fixed k. split_k == 0 chooses k automatically by
    silhouette score. This step does not use truth labels or true SigIdx.
    """
    if min_count <= 0 or sample_size <= 0:
        return sigidx.copy(), pd.DataFrame()

    try:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score
    except ImportError as exc:
        raise RuntimeError("split_large_batches_by_kmeans requires scikit-learn.") from exc

    out = sigidx.copy()
    features = build_split_features(pdw, tolerances)
    rng = np.random.default_rng(random_state)
    next_id = int(max([0] + [int(v) for v in np.unique(out) if int(v) > 0])) + 1
    report_rows: List[Dict[str, object]] = []

    positive_ids, counts = np.unique(out[out > 0], return_counts=True)
    candidates = [(int(pid), int(count)) for pid, count in zip(positive_ids, counts) if int(count) >= min_count]

    for pred_id, count in candidates:
        batch_idx = np.flatnonzero(out == pred_id)
        if len(batch_idx) < min_count:
            continue

        this_sample_size = min(int(sample_size), len(batch_idx))
        if this_sample_size < 2:
            continue
        sample_idx = rng.choice(batch_idx, size=this_sample_size, replace=False)
        sample_x = features[sample_idx].astype(np.float32, copy=False)

        median = np.median(sample_x, axis=0).astype(np.float32)
        q75 = np.percentile(sample_x, 75, axis=0).astype(np.float32)
        q25 = np.percentile(sample_x, 25, axis=0).astype(np.float32)
        scale = np.maximum(q75 - q25, 1e-6).astype(np.float32)
        sample_scaled = ((sample_x - median) / scale).astype(np.float32)

        if split_k > 1:
            k_values = [int(split_k)]
        else:
            k_values = list(range(2, max(2, int(max_k)) + 1))

        best_model = None
        best_score = -np.inf
        best_k = 0
        best_sample_counts = None
        min_sample_child = max(20, int(np.ceil(min_child_count * this_sample_size / max(len(batch_idx), 1))))

        for k in k_values:
            if k < 2 or this_sample_size < k * min_sample_child:
                continue
            model = MiniBatchKMeans(
                n_clusters=k,
                random_state=random_state,
                batch_size=max(8192, k * 1024),
                n_init=3,
                max_iter=120,
            )
            sample_labels = model.fit_predict(sample_scaled)
            sample_counts = np.bincount(sample_labels, minlength=k)
            if int(sample_counts.min()) < min_sample_child:
                continue

            score_sample_size = min(3000, this_sample_size)
            score = float(
                silhouette_score(
                    sample_scaled,
                    sample_labels,
                    sample_size=score_sample_size,
                    random_state=random_state,
                )
            )
            if score > best_score:
                best_model = model
                best_score = score
                best_k = k
                best_sample_counts = sample_counts

        if best_model is None or best_score < min_silhouette:
            report_rows.append(
                {
                    "original_sigidx": pred_id,
                    "original_count": count,
                    "split": False,
                    "reason": "no_separated_clusters",
                    "k": best_k,
                    "silhouette": float(best_score) if np.isfinite(best_score) else np.nan,
                    "child_counts": "",
                }
            )
            continue

        labels_full = np.empty((len(batch_idx),), dtype=np.int64)
        predict_chunk = 200000
        for start in range(0, len(batch_idx), predict_chunk):
            end = min(start + predict_chunk, len(batch_idx))
            chunk = ((features[batch_idx[start:end]] - median) / scale).astype(np.float32)
            labels_full[start:end] = best_model.predict(chunk).astype(np.int64)

        full_counts = np.bincount(labels_full, minlength=best_k).astype(np.int64)
        keep_clusters = np.flatnonzero(full_counts >= min_child_count)
        if len(keep_clusters) < 2:
            report_rows.append(
                {
                    "original_sigidx": pred_id,
                    "original_count": count,
                    "split": False,
                    "reason": "children_too_small",
                    "k": best_k,
                    "silhouette": best_score,
                    "child_counts": ",".join(str(int(v)) for v in full_counts),
                }
            )
            continue

        if len(keep_clusters) < best_k:
            keep_centers = best_model.cluster_centers_[keep_clusters]
            for small_cluster in np.flatnonzero(full_counts < min_child_count):
                distances = np.linalg.norm(keep_centers - best_model.cluster_centers_[small_cluster], axis=1)
                labels_full[labels_full == small_cluster] = int(keep_clusters[int(np.argmin(distances))])
            remap = {int(old): pos for pos, old in enumerate(sorted(int(v) for v in np.unique(labels_full)))}
            labels_full = np.array([remap[int(v)] for v in labels_full], dtype=np.int64)
            full_counts = np.bincount(labels_full).astype(np.int64)

        largest_cluster = int(np.argmax(full_counts))
        for cluster_id in range(len(full_counts)):
            target_id = pred_id if cluster_id == largest_cluster else next_id
            if cluster_id != largest_cluster:
                next_id += 1
            out[batch_idx[labels_full == cluster_id]] = target_id

        report_rows.append(
            {
                "original_sigidx": pred_id,
                "original_count": count,
                "split": True,
                "reason": "",
                "k": int(len(full_counts)),
                "silhouette": best_score,
                "sample_child_counts": ",".join(str(int(v)) for v in best_sample_counts) if best_sample_counts is not None else "",
                "child_counts": ",".join(str(int(v)) for v in full_counts),
            }
        )

    return out, pd.DataFrame(report_rows)


def reduce_sigidx_fixed(
    pdw: pd.DataFrame,
    sigidx: np.ndarray,
    min_cluster_size: int = 1,
    merge_thresh: float = 0.2,
    min_batch_fraction: float = 0.0,
    weights: Tuple[float, float, float, float] = (1.4, 1.0, 0.8, 1.2),
    tolerances: Tuple[float, float, float, float] = (150.0, 0.2, 10.0, 20.0),
    hard_gates: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    dense_relabel_output: bool = True,
    split_large_batches: bool = False,
    split_k: int = 0,
    split_max_k: int = 8,
    split_min_count: int = 200000,
    split_min_child_count: int = 500,
    split_sample_size: int = 12000,
    split_min_silhouette: float = 0.35,
    split_random_state: int = 1234,
) -> Tuple[np.ndarray, Dict[str, object], pd.DataFrame]:
    """Apply the production reduction path without truth-label parameter sweep."""
    if not 0 <= min_batch_fraction < 1:
        raise ValueError("min_batch_fraction must be in [0, 1).")
    if len(pdw) != len(sigidx):
        raise ValueError(f"Row count mismatch: pdw={len(pdw)}, sigidx={len(sigidx)}")

    out = sigidx.astype(np.int64).copy()
    positive_before = out[out > 0]
    summary: Dict[str, object] = {
        "mode": "fixed_no_truth",
        "min_cluster_size": int(min_cluster_size),
        "merge_thresh": float(merge_thresh),
        "min_batch_fraction": float(min_batch_fraction),
        "weights": [float(v) for v in weights],
        "tolerances": [float(v) for v in tolerances],
        "hard_gates": [float(v) for v in hard_gates],
        "positive_sigidx_before": int(len(np.unique(positive_before))) if len(positive_before) else 0,
        "noise_or_zero_before": int(np.sum(out <= 0)),
    }

    small_ids = np.array([], dtype=np.int64)
    if min_cluster_size > 1:
        positive = out[out > 0]
        ids, counts = np.unique(positive, return_counts=True)
        small_ids = ids[counts < min_cluster_size]
        if len(small_ids) > 0:
            out[np.isin(out, small_ids)] = 0
    summary["small_sigidx_mapped_to_zero"] = int(len(small_ids))

    centers = build_track_centers(pdw, out)
    mapping = merge_similar_sigidx(
        centers=centers,
        merge_thresh=merge_thresh,
        weights=weights,
        tolerances=tolerances,
        hard_gates=hard_gates,
    )
    center_merge_count = 0
    if mapping:
        mapped = out.copy()
        for old, new in mapping.items():
            if old != new:
                mapped[out == old] = new
                center_merge_count += 1
        out = mapped
    summary["center_merge_count"] = int(center_merge_count)

    centers_after_merge = build_track_centers(pdw, out)
    out, fraction_merge_count, min_batch_count, fallback_merge_count = merge_small_batches_by_fraction(
        sigidx=out,
        centers=centers_after_merge,
        min_fraction=min_batch_fraction,
        weights=weights,
        tolerances=tolerances,
        hard_gates=hard_gates,
    )
    summary.update(
        {
            "fraction_merge_count": int(fraction_merge_count),
            "fraction_min_batch_count": int(min_batch_count),
            "fraction_fallback_merge_count": int(fallback_merge_count),
        }
    )

    split_report = pd.DataFrame()
    if split_large_batches:
        out, split_report = split_large_batches_by_kmeans(
            pdw=pdw,
            sigidx=out,
            tolerances=tolerances,
            split_k=split_k,
            max_k=split_max_k,
            min_count=split_min_count,
            min_child_count=split_min_child_count,
            sample_size=split_sample_size,
            min_silhouette=split_min_silhouette,
            random_state=split_random_state,
        )
    split_count = int(split_report["split"].sum()) if len(split_report) > 0 and "split" in split_report.columns else 0
    summary["split_large_batches"] = bool(split_large_batches)
    summary["split_batches"] = int(split_count)

    if dense_relabel_output:
        out = relabel_dense(out)

    positive_after = out[out > 0]
    summary["positive_sigidx_after"] = int(len(np.unique(positive_after))) if len(positive_after) else 0
    summary["noise_or_zero_after"] = int(np.sum(out <= 0))
    return out.astype(np.int64), summary, split_report


def summarize(sigidx: np.ndarray, top_k: int = 20) -> str:
    uniq, counts = np.unique(sigidx, return_counts=True)
    valid = uniq > 0
    uniq_valid = uniq[valid]
    counts_valid = counts[valid]
    lines = [
        f"total_pulses={len(sigidx)}",
        f"positive_sigidx_count={len(uniq_valid)}",
        f"noise_or_unassigned_count={int(counts[uniq == 0][0]) if np.any(uniq == 0) else 0}",
    ]
    if len(uniq_valid) == 0:
        return "\n".join(lines)

    order = np.argsort(-counts_valid)
    lines.append(f"top_{min(top_k, len(order))}_sigidx:")
    for pos in order[:top_k]:
        lines.append(f"  SigIdx={int(uniq_valid[pos])}, count={int(counts_valid[pos])}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge similar SigIdx tracks after PDW sorting.")
    parser.add_argument("--pdw_file", type=Path, required=True, help="Original PDW file with TOA and Param columns.")
    parser.add_argument("--sorted_file", type=Path, required=True, help="Sorting output with TOA(s) and SigIdx columns.")
    parser.add_argument("--output_file", type=Path, required=True, help="Reduced sorting output path.")
    parser.add_argument("--min_cluster_size", type=int, default=20, help="Map tracks smaller than this to SigIdx=0.")
    parser.add_argument("--min_batch_fraction", type=float, default=0.0, help="Optional. After merging, merge final batches smaller than this fraction of all pulses into the nearest remaining SigIdx. Default 0 disables this rule.")
    parser.add_argument("--merge_thresh", type=float, default=2.8, help="Weighted center distance threshold. Larger merges more.")
    parser.add_argument("--no_dense_relabel", action="store_true", help="Keep original representative SigIdx values.")
    parser.add_argument("--validate_toa", action="store_true", help="Check that PDW and sorted TOA columns match closely.")
    parser.add_argument("--split_large_batches", action="store_true", help="After merging, split large mixed-looking SigIdx batches in PDW feature space.")
    parser.add_argument("--split_k", type=int, default=8, help="Fixed K for large-batch splitting. Use 0 to choose K automatically.")
    parser.add_argument("--split_max_k", type=int, default=8, help="Maximum K when --split_k is 0.")
    parser.add_argument("--split_min_count", type=int, default=200000, help="Only split SigIdx batches with at least this many pulses.")
    parser.add_argument("--split_min_child_count", type=int, default=500, help="After splitting, each child batch must contain at least this many pulses.")
    parser.add_argument("--split_sample_size", type=int, default=12000, help="Sample size used to fit each large-batch split.")
    parser.add_argument("--split_min_silhouette", type=float, default=0.35, help="Skip a split unless KMeans silhouette is at least this value.")
    parser.add_argument("--split_random_state", type=int, default=1234)

    parser.add_argument("--w_p1", type=float, default=1.4)
    parser.add_argument("--w_p2", type=float, default=1.0)
    parser.add_argument("--w_p4", type=float, default=0.8)
    parser.add_argument("--w_p5", type=float, default=1.2)

    parser.add_argument("--tol_p1", type=float, default=150.0)
    parser.add_argument("--tol_p2", type=float, default=0.2)
    parser.add_argument("--tol_p4", type=float, default=10.0)
    parser.add_argument("--tol_p5_deg", type=float, default=20.0)

    # hard_gate = 1.5 -> 27批
    # hard_gate = 1.8 -> 22批
    # hard_gate = 2.0 -> 18批
    # hard_gate = 2.1 -> 15批
    # hard_gate = 2.2 -> 10批
    # hard_gate >= 2.3 -> 9批
    parser.add_argument("--hard_gate_p1", type=float, default=1.5)
    parser.add_argument("--hard_gate_p2", type=float, default=1.5)
    parser.add_argument("--hard_gate_p4", type=float, default=1.5)
    parser.add_argument("--hard_gate_p5", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.min_batch_fraction < 1:
        raise ValueError("--min_batch_fraction must be in [0, 1).")

    pdw = read_pdw(args.pdw_file)
    sorted_df = read_sorted(args.sorted_file)
    if len(pdw) != len(sorted_df):
        raise ValueError(f"Row count mismatch: pdw={len(pdw)}, sorted={len(sorted_df)}")

    if args.validate_toa and "TOA(s)" in sorted_df.columns:
        pdw_toa = pdw["TOA(s)"].to_numpy(dtype=np.float64)
        sorted_toa = pd.to_numeric(sorted_df["TOA(s)"], errors="coerce").to_numpy(dtype=np.float64)
        max_abs_err = float(np.nanmax(np.abs(pdw_toa - sorted_toa)))
        if max_abs_err > 1e-7:
            raise ValueError(f"TOA mismatch is too large: max_abs_err={max_abs_err}")

    original_sigidx = sorted_df["SigIdx"].to_numpy(dtype=np.int64)
    reduced_sigidx = original_sigidx.copy()

    print("Before reduction:")
    print(summarize(original_sigidx))

    if args.min_cluster_size > 1:
        positive = reduced_sigidx[reduced_sigidx > 0]
        ids, counts = np.unique(positive, return_counts=True)
        small_ids = ids[counts < args.min_cluster_size]
        if len(small_ids) > 0:
            reduced_sigidx[np.isin(reduced_sigidx, small_ids)] = 0
        print(f"Small-track filter: mapped {len(small_ids)} SigIdx values to 0")

    centers = build_track_centers(pdw, reduced_sigidx)
    mapping = merge_similar_sigidx(
        centers=centers,
        merge_thresh=args.merge_thresh,
        weights=(args.w_p1, args.w_p2, args.w_p4, args.w_p5),
        tolerances=(args.tol_p1, args.tol_p2, args.tol_p4, args.tol_p5_deg),
        hard_gates=(args.hard_gate_p1, args.hard_gate_p2, args.hard_gate_p4, args.hard_gate_p5),
    )

    if mapping:
        mapped = reduced_sigidx.copy()
        for old, new in mapping.items():
            if old != new:
                mapped[reduced_sigidx == old] = new
        reduced_sigidx = mapped

    centers_after_merge = build_track_centers(pdw, reduced_sigidx)
    reduced_sigidx, fraction_merge_count, min_batch_count, fallback_merge_count = merge_small_batches_by_fraction(
        sigidx=reduced_sigidx,
        centers=centers_after_merge,
        min_fraction=args.min_batch_fraction,
        weights=(args.w_p1, args.w_p2, args.w_p4, args.w_p5),
        tolerances=(args.tol_p1, args.tol_p2, args.tol_p4, args.tol_p5_deg),
        hard_gates=(args.hard_gate_p1, args.hard_gate_p2, args.hard_gate_p4, args.hard_gate_p5),
    )
    if args.min_batch_fraction > 0:
        print(
            "Sample-fraction merge: "
            f"min_count={min_batch_count} ({args.min_batch_fraction:.2%}), "
            f"merged {fraction_merge_count} small SigIdx values into nearest batches "
            f"({fallback_merge_count} without a hard-gate match)"
        )

    if args.split_large_batches:
        reduced_sigidx, split_report = split_large_batches_by_kmeans(
            pdw=pdw,
            sigidx=reduced_sigidx,
            tolerances=(args.tol_p1, args.tol_p2, args.tol_p4, args.tol_p5_deg),
            split_k=args.split_k,
            max_k=args.split_max_k,
            min_count=args.split_min_count,
            min_child_count=args.split_min_child_count,
            sample_size=args.split_sample_size,
            min_silhouette=args.split_min_silhouette,
            random_state=args.split_random_state,
        )
        split_count = int(split_report["split"].sum()) if len(split_report) > 0 and "split" in split_report.columns else 0
        print(f"Large-batch split: split {split_count} SigIdx values")
        if len(split_report) > 0:
            print(split_report.to_string(index=False))

    if not args.no_dense_relabel:
        reduced_sigidx = relabel_dense(reduced_sigidx)

    sorted_df["SigIdx"] = reduced_sigidx.astype(np.int64)
    sorted_df.to_csv(args.output_file, sep=" ", index=False)

    print("\nAfter reduction:")
    print(summarize(reduced_sigidx))
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()

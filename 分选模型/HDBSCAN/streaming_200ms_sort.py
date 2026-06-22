#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming-style 200 ms PDW sorting with cross-beat SigIdx association.

Input is one whole PDW txt file. The script splits it into fixed 200 ms beats,
runs the existing sorter independently on each beat, maps beat-local SigIdx
values to stable global SigIdx values, and writes one output txt per beat with
the original columns plus a SigIdx column.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import Best


FEATURE_COLUMNS = ["p1_median", "p2_median", "p4_median", "p5_mean_deg", "pri_median_us"]


@dataclass
class TrackState:
    global_sigidx: int
    last_seen_beat: int
    num_seen_beats: int
    total_pulses: int
    features: Dict[str, float]
    feature_iqr: Dict[str, float] = field(default_factory=dict)
    prototypes: List[Dict[str, float]] = field(default_factory=list)
    prototype_iqrs: List[Dict[str, float]] = field(default_factory=list)


def read_pdw(path: Path, max_rows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    if max_rows > 0:
        df = df.iloc[:max_rows].reset_index(drop=True)
    return df


def normalize_pdw_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TOA(s)" not in out.columns:
        raise ValueError("Input file must contain a TOA(s) column.")
    if len(out.columns) < 8:
        raise ValueError("Input file must contain TOA(s) plus at least Param1..Param7 columns.")
    if "Param1" not in out.columns:
        rename = {out.columns[i]: f"Param{i}" for i in range(1, min(8, len(out.columns)))}
        out = out.rename(columns=rename)
    return out


def robust_iqr(values: Iterable[float], floor: float) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return float(floor)
    q25, q75 = np.percentile(arr, [25, 75])
    return max(float(q75 - q25), float(floor))


def circular_mean_deg(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    rad = np.deg2rad(np.mod(arr, 360.0))
    return float(np.rad2deg(math.atan2(float(np.sin(rad).mean()), float(np.cos(rad).mean()))) % 360.0)


def angle_delta_deg(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 360.0
    return float(min(delta, 360.0 - delta))


def robust_pri_us(toa_s: np.ndarray, gap_multiplier: float = 5.0, gap_quantile: float = 0.90) -> np.ndarray:
    toa = np.sort(np.asarray(toa_s, dtype=np.float64))
    dtoa = np.diff(toa) * 1e6
    dtoa = dtoa[np.isfinite(dtoa) & (dtoa > 0.0)]
    if len(dtoa) < 3:
        return dtoa
    median = float(np.median(dtoa))
    cap = max(float(gap_multiplier) * max(median, 1e-9), float(np.quantile(dtoa, gap_quantile)))
    return dtoa[dtoa <= cap]


def summarize_local_batches(pdw: pd.DataFrame, local_sigidx: np.ndarray, min_pulses: int) -> pd.DataFrame:
    rows = []
    for local_id in sorted(int(v) for v in np.unique(local_sigidx) if int(v) > 0):
        mask = local_sigidx == local_id
        sub = pdw.loc[mask]
        if len(sub) < int(min_pulses):
            continue
        pri = robust_pri_us(sub["TOA(s)"].to_numpy(dtype=np.float64))
        rows.append(
            {
                "local_sigidx": int(local_id),
                "num_pulses": int(len(sub)),
                "start_toa": float(sub["TOA(s)"].min()),
                "end_toa": float(sub["TOA(s)"].max()),
                "p1_median": float(sub["Param1"].median()),
                "p1_iqr": robust_iqr(sub["Param1"], 1.0),
                "p2_median": float(sub["Param2"].median()),
                "p2_iqr": robust_iqr(sub["Param2"], 0.01),
                "p4_median": float(sub["Param4"].median()),
                "p4_iqr": robust_iqr(sub["Param4"], 0.1),
                "p5_mean_deg": circular_mean_deg(sub["Param5"].to_numpy(dtype=np.float64)),
                "pri_median_us": float(np.median(pri)) if len(pri) else 0.0,
                "pri_iqr_us": robust_iqr(pri, 0.5),
            }
        )
    return pd.DataFrame(rows)


def row_features(row: pd.Series) -> Dict[str, float]:
    return {name: float(row[name]) for name in FEATURE_COLUMNS}


def row_iqr(row: pd.Series) -> Dict[str, float]:
    return {
        "p1_median": max(float(row.get("p1_iqr", 1.0)), 1.0),
        "p2_median": max(float(row.get("p2_iqr", 0.01)), 0.01),
        "p4_median": max(float(row.get("p4_iqr", 0.1)), 0.1),
        "p5_mean_deg": 1.0,
        "pri_median_us": max(float(row.get("pri_iqr_us", 0.5)), 0.5),
    }


def feature_distance(
    batch_features: Dict[str, float],
    track: TrackState,
    beat_gap: int,
    args: argparse.Namespace,
) -> float:
    scales = {
        "p1_median": float(args.scale_param1),
        "p2_median": float(args.scale_param2),
        "p4_median": float(args.scale_param4),
        "p5_mean_deg": float(args.scale_param5_deg),
        "pri_median_us": float(args.scale_pri_us),
    }
    weights = {
        "p1_median": float(args.weight_param1),
        "p2_median": float(args.weight_param2),
        "p4_median": float(args.weight_param4),
        "p5_mean_deg": float(args.weight_param5),
        "pri_median_us": float(args.weight_pri),
    }
    total = 0.0
    for name in FEATURE_COLUMNS:
        if name == "p5_mean_deg":
            delta = angle_delta_deg(batch_features[name], track.features[name])
        else:
            delta = abs(batch_features[name] - track.features[name])
        scale = max(scales[name], track.feature_iqr.get(name, 0.0), 1e-9)
        total += weights[name] * (delta / scale) ** 2
    total += float(args.weight_time_gap) * max(int(beat_gap) - 1, 0)
    return float(math.sqrt(total))


def track_from_features(features: Dict[str, float], iqrs: Dict[str, float] | None = None) -> TrackState:
    return TrackState(
        global_sigidx=-1,
        last_seen_beat=0,
        num_seen_beats=0,
        total_pulses=0,
        features=dict(features),
        feature_iqr=dict(iqrs or {}),
    )


def prototype_distance(
    batch_features: Dict[str, float],
    track: TrackState,
    beat_gap: int,
    args: argparse.Namespace,
) -> float:
    candidates = [(track.features, track.feature_iqr)]
    candidates.extend(zip(track.prototypes, track.prototype_iqrs))
    distances = [
        feature_distance(batch_features, track_from_features(features, iqrs), beat_gap, args)
        for features, iqrs in candidates
    ]
    base = min(distances) if distances else feature_distance(batch_features, track, beat_gap, args)
    if beat_gap > int(args.max_track_gap_beats):
        base += float(args.archive_time_decay) * math.log1p(max(int(beat_gap), 0))
    return float(base)


def passes_archive_gate(row: pd.Series, track: TrackState, args: argparse.Namespace) -> bool:
    feature_sets = [track.features] + list(track.prototypes)
    for features in feature_sets:
        p5_ok = angle_delta_deg(float(row["p5_mean_deg"]), features["p5_mean_deg"]) <= float(args.archive_gate_param5_deg)
        p4_ok = abs(float(row["p4_median"]) - features["p4_median"]) <= float(args.archive_gate_param4)
        pri_ok = abs(float(row["pri_median_us"]) - features["pri_median_us"]) <= float(args.archive_gate_pri_us)
        if p5_ok and p4_ok and pri_ok:
            return True
    return False


def local_feature_distance(a: pd.Series, b: pd.Series, args: argparse.Namespace) -> float:
    return feature_distance(row_features(a), track_from_features(row_features(b), row_iqr(b)), 1, args)


def update_track_prototypes(track: TrackState, row: pd.Series, args: argparse.Namespace) -> None:
    features = row_features(row)
    iqrs = row_iqr(row)
    if not track.prototypes:
        track.prototypes.append(dict(features))
        track.prototype_iqrs.append(dict(iqrs))
        return

    distances = [
        feature_distance(features, track_from_features(proto, proto_iqr), 1, args)
        for proto, proto_iqr in zip(track.prototypes, track.prototype_iqrs)
    ]
    best_index = int(np.argmin(distances))
    if float(distances[best_index]) > float(args.prototype_add_threshold) and len(track.prototypes) < int(args.max_prototypes_per_track):
        track.prototypes.append(dict(features))
        track.prototype_iqrs.append(dict(iqrs))
        return

    alpha = float(args.prototype_update_alpha)
    proto = track.prototypes[best_index]
    proto_iqr = track.prototype_iqrs[best_index]
    for name, value in features.items():
        if name == "p5_mean_deg":
            delta = ((value - proto[name] + 180.0) % 360.0) - 180.0
            proto[name] = (proto[name] + alpha * delta) % 360.0
        else:
            proto[name] = (1.0 - alpha) * proto[name] + alpha * value
        proto_iqr[name] = max((1.0 - alpha) * proto_iqr.get(name, iqrs[name]) + alpha * iqrs[name], 1e-9)


def update_track(track: TrackState, row: pd.Series, beat_id: int, args: argparse.Namespace) -> None:
    alpha = float(args.update_alpha)
    features = row_features(row)
    iqrs = row_iqr(row)
    for name, value in features.items():
        if name == "p5_mean_deg":
            # For short-term smoothing this linearized angle update is enough;
            # association still uses circular distance.
            delta = ((value - track.features[name] + 180.0) % 360.0) - 180.0
            track.features[name] = (track.features[name] + alpha * delta) % 360.0
        else:
            track.features[name] = (1.0 - alpha) * track.features[name] + alpha * value
        track.feature_iqr[name] = max((1.0 - alpha) * track.feature_iqr.get(name, iqrs[name]) + alpha * iqrs[name], 1e-9)
    track.last_seen_beat = int(beat_id)
    track.num_seen_beats += 1
    track.total_pulses += int(row["num_pulses"])
    update_track_prototypes(track, row, args)


def associate_batches(
    beat_id: int,
    batch_df: pd.DataFrame,
    tracks: Dict[int, TrackState],
    next_global_id: int,
    args: argparse.Namespace,
) -> Tuple[Dict[int, int], int, List[Dict[str, object]]]:
    assignments: Dict[int, int] = {}
    rows: List[Dict[str, object]] = []
    if len(batch_df) == 0:
        return assignments, next_global_id, rows

    max_gap = int(args.max_track_gap_beats)
    active_tracks = {
        gid: track
        for gid, track in tracks.items()
        if max_gap <= 0 or int(beat_id) - int(track.last_seen_beat) <= max_gap
    }
    archive_tracks = {
        gid: track
        for gid, track in tracks.items()
        if bool(args.enable_archive_memory)
        and max_gap > 0
        and int(beat_id) - int(track.last_seen_beat) > max_gap
        and (int(args.archive_max_gap_beats) <= 0 or int(beat_id) - int(track.last_seen_beat) <= int(args.archive_max_gap_beats))
        and int(track.num_seen_beats) >= int(args.archive_min_seen_beats)
    }

    candidates = []
    for row_index, row in batch_df.iterrows():
        features = row_features(row)
        for gid, track in active_tracks.items():
            gap = int(beat_id) - int(track.last_seen_beat)
            dist = prototype_distance(features, track, gap, args)
            if dist <= float(args.link_threshold):
                candidates.append((dist, int(row_index), int(gid)))

    used_rows: set[int] = set()
    used_tracks: set[int] = set()
    for dist, row_index, gid in sorted(candidates, key=lambda item: item[0]):
        if row_index in used_rows or gid in used_tracks:
            continue
        local_id = int(batch_df.loc[row_index, "local_sigidx"])
        assignments[local_id] = gid
        used_rows.add(row_index)
        used_tracks.add(gid)
        rows.append(
            {
                "beat": int(beat_id),
                "local_sigidx": local_id,
                "global_sigidx": gid,
                "decision": "linked",
                "distance": float(dist),
                "memory": "active",
            }
        )

    archive_candidates = []
    if archive_tracks:
        per_row_candidates: Dict[int, List[Tuple[float, int, int]]] = {}
        for row_index, row in batch_df.iterrows():
            if int(row_index) in used_rows:
                continue
            features = row_features(row)
            row_candidates = []
            for gid, track in archive_tracks.items():
                if gid in used_tracks:
                    continue
                if not passes_archive_gate(row, track, args):
                    continue
                gap = int(beat_id) - int(track.last_seen_beat)
                dist = prototype_distance(features, track, gap, args)
                if dist <= float(args.archive_link_threshold):
                    row_candidates.append((dist, int(row_index), int(gid)))
            per_row_candidates[int(row_index)] = sorted(row_candidates, key=lambda item: item[0])[: int(args.archive_top_k)]
        for row_candidates in per_row_candidates.values():
            archive_candidates.extend(row_candidates)

    for dist, row_index, gid in sorted(archive_candidates, key=lambda item: item[0]):
        if row_index in used_rows or gid in used_tracks:
            continue
        local_id = int(batch_df.loc[row_index, "local_sigidx"])
        assignments[local_id] = gid
        used_rows.add(row_index)
        used_tracks.add(gid)
        rows.append(
            {
                "beat": int(beat_id),
                "local_sigidx": local_id,
                "global_sigidx": gid,
                "decision": "archive_restored",
                "distance": float(dist),
                "memory": "archive",
                "beat_gap": int(beat_id) - int(tracks[gid].last_seen_beat),
            }
        )

    if bool(args.allow_fragment_merge):
        linked_by_gid = {
            gid: int(batch_df.index[batch_df["local_sigidx"] == local_id][0])
            for local_id, gid in assignments.items()
        }
        for row_index, row in batch_df.iterrows():
            local_id = int(row["local_sigidx"])
            if local_id in assignments:
                continue
            best = None
            for gid, linked_index in linked_by_gid.items():
                if gid not in active_tracks:
                    continue
                track_dist = prototype_distance(row_features(row), active_tracks[gid], 1, args)
                pair_dist = local_feature_distance(row, batch_df.loc[linked_index], args)
                if track_dist <= float(args.fragment_link_threshold) and pair_dist <= float(args.fragment_pair_threshold):
                    score = track_dist + pair_dist
                    if best is None or score < best[0]:
                        best = (score, gid, track_dist, pair_dist)
            if best is not None:
                _, gid, track_dist, pair_dist = best
                assignments[local_id] = int(gid)
                used_rows.add(int(row_index))
                rows.append(
                    {
                        "beat": int(beat_id),
                        "local_sigidx": local_id,
                        "global_sigidx": int(gid),
                        "decision": "fragment_merged",
                        "distance": float(track_dist),
                        "pair_distance": float(pair_dist),
                        "memory": "active",
                    }
                )

    for row_index, row in batch_df.iterrows():
        local_id = int(row["local_sigidx"])
        if local_id in assignments:
            continue
        gid = int(next_global_id)
        next_global_id += 1
        assignments[local_id] = gid
        tracks[gid] = TrackState(
            global_sigidx=gid,
            last_seen_beat=int(beat_id),
            num_seen_beats=0,
            total_pulses=0,
            features=row_features(row),
            feature_iqr=row_iqr(row),
            prototypes=[row_features(row)],
            prototype_iqrs=[row_iqr(row)],
        )
        rows.append(
            {
                "beat": int(beat_id),
                "local_sigidx": local_id,
                "global_sigidx": gid,
                "decision": "new_track",
                "distance": np.nan,
                "memory": "new",
            }
        )

    for _, row in batch_df.iterrows():
        local_id = int(row["local_sigidx"])
        gid = int(assignments[local_id])
        update_track(tracks[gid], row, beat_id, args)

    return assignments, next_global_id, rows


def write_pdw_with_sigidx(path: Path, pdw: pd.DataFrame, sigidx: np.ndarray, sigidx_column: str) -> None:
    out = pdw.copy()
    out[sigidx_column] = sigidx.astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, sep=" ", index=False, float_format="%.9f")


def run_local_sort(beat_df: pd.DataFrame, beat_id: int, temp_dir: Path, args: argparse.Namespace) -> Tuple[np.ndarray, str, str]:
    if len(beat_df) < int(args.min_sort_pulses):
        return np.zeros((len(beat_df),), dtype=np.int64), "skipped_too_few_pulses", ""

    input_path = temp_dir / f"beat_{beat_id:06d}_input.txt"
    output_path = temp_dir / f"beat_{beat_id:06d}_local_sort.txt"
    raw_output_path = temp_dir / f"beat_{beat_id:06d}_raw_sort.txt"
    report_path = temp_dir / f"beat_{beat_id:06d}_summary.json"
    beat_df.to_csv(input_path, sep=" ", index=False, float_format="%.9f")

    backend = str(args.sort_backend)
    sorter, run_fn = Best.import_sort_backend(backend)
    sort_args = sorter.build_parser().parse_args([])
    sort_args.input_file = input_path
    sort_args.truth_file = Path("__missing_truth__.txt")
    sort_args.output_file = output_path
    sort_args.raw_output_file = raw_output_path
    sort_args.report_json = report_path
    sort_args.metrics_dir = temp_dir / f"beat_{beat_id:06d}_metrics"
    sort_args.window_report_csv = temp_dir / f"beat_{beat_id:06d}_windows.csv"
    sort_args.edge_report_csv = temp_dir / f"beat_{beat_id:06d}_edges.csv"
    sort_args.max_pulses = 0
    sort_args.skip_metrics = True
    sort_args.seed = int(args.seed) + int(beat_id)
    sort_args.n_jobs = int(args.n_jobs)
    sort_args.verbose = bool(args.verbose)
    if hasattr(sort_args, "emit_label99"):
        sort_args.emit_label99 = True
    if hasattr(sort_args, "hdbscan_backend"):
        sort_args.hdbscan_backend = "auto"
    if hasattr(sort_args, "allow_optics_fallback"):
        sort_args.allow_optics_fallback = True
    if backend == "pa_tsr":
        sort_args.tsr_band_split_auto = bool(args.tsr_band_split_auto)
        sort_args.tsr_reduce_merge_thresh = float(args.tsr_reduce_merge_thresh)
        sort_args.tsr_rescue_merge_thresh = float(args.tsr_rescue_merge_thresh)
    elif backend == "pa_tgr":
        sort_args.tgr_pre_reduce = True
        sort_args.tgr_noise_rescue = True
        sort_args.tgr_mutual_nearest = True
        sort_args.tgr_check_component = True
        sort_args.tgr_tail_cleanup = True
        sort_args.tgr_thin_cleanup = True

    try:
        run_fn(sort_args)
        sigidx = Best.read_sort_sigidx(output_path)
        positive = int(np.sum(sigidx > 0))
        status = "ok" if positive > 0 else "ok_all_zero"
        if positive == 0 and bool(args.use_raw_on_sort_error) and raw_output_path.exists():
            raw_sigidx = Best.read_sort_sigidx(raw_output_path)
            if int(np.sum(raw_sigidx > 0)) > 0:
                return raw_sigidx, "ok_all_zero_raw_fallback", "final output was all zero; used raw sort"
        return sigidx, status, ""
    except Exception as exc:
        if bool(args.use_raw_on_sort_error) and raw_output_path.exists():
            raw_sigidx = Best.read_sort_sigidx(raw_output_path)
            if int(np.sum(raw_sigidx > 0)) > 0:
                error = repr(exc)
                print(f"[warn] beat {beat_id}: final sorter failed, using raw sort. reason={error}")
                return raw_sigidx, "sort_failed_raw_fallback", error
        if bool(args.fail_on_sort_error):
            raise
        error = repr(exc)
        print(f"[warn] beat {beat_id}: local sorter failed, marking as 0. reason={error}")
        return np.zeros((len(beat_df),), dtype=np.int64), "sort_failed_zero_fallback", error


def iter_beats(df: pd.DataFrame, beat_seconds: float) -> Iterable[Tuple[int, pd.DataFrame]]:
    toa = pd.to_numeric(df["TOA(s)"], errors="raise").to_numpy(dtype=np.float64)
    t0 = float(np.min(toa)) if len(toa) else 0.0
    beat_ids = np.floor((toa - t0) / float(beat_seconds)).astype(np.int64)
    work = df.copy()
    work["_stream_beat"] = beat_ids
    for beat_id, sub in work.groupby("_stream_beat", sort=True):
        yield int(beat_id), sub.drop(columns=["_stream_beat"]).copy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split one PDW txt into 200 ms beats, sort each beat, and associate global SigIdx.")
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs_streaming_200ms"))
    parser.add_argument("--temp_dir", type=Path, default=Path("outputs_streaming_200ms/_tmp"))
    parser.add_argument("--beat_seconds", type=float, default=0.2)
    parser.add_argument("--sigidx_column", type=str, default="SigIdx")
    parser.add_argument("--sort_backend", choices=["pa_tsr", "pa_tgr"], default="pa_tsr")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--min_sort_pulses", type=int, default=50)
    parser.add_argument("--min_batch_pulses", type=int, default=20)
    parser.add_argument("--max_track_gap_beats", type=int, default=4, help="Only link to tracks seen within this many beats; use 0 to compare with all previous tracks.")
    parser.add_argument("--link_threshold", type=float, default=2.2)
    parser.add_argument("--enable_archive_memory", action="store_true", default=True)
    parser.add_argument("--no_enable_archive_memory", dest="enable_archive_memory", action="store_false")
    parser.add_argument("--archive_max_gap_beats", type=int, default=0, help="Archive recall maximum gap; 0 means no hard limit.")
    parser.add_argument("--archive_min_seen_beats", type=int, default=2)
    parser.add_argument("--archive_top_k", type=int, default=5)
    parser.add_argument("--archive_link_threshold", type=float, default=1.8)
    parser.add_argument("--archive_time_decay", type=float, default=0.08)
    parser.add_argument("--archive_gate_param5_deg", type=float, default=1.2)
    parser.add_argument("--archive_gate_param4", type=float, default=6.0)
    parser.add_argument("--archive_gate_pri_us", type=float, default=15.0)
    parser.add_argument("--max_prototypes_per_track", type=int, default=4)
    parser.add_argument("--prototype_add_threshold", type=float, default=1.1)
    parser.add_argument("--prototype_update_alpha", type=float, default=0.20)
    parser.add_argument("--fragment_link_threshold", type=float, default=2.8)
    parser.add_argument("--fragment_pair_threshold", type=float, default=1.2)
    parser.add_argument("--allow_fragment_merge", action="store_true", default=True)
    parser.add_argument("--no_allow_fragment_merge", dest="allow_fragment_merge", action="store_false")
    parser.add_argument("--update_alpha", type=float, default=0.25)
    parser.add_argument("--scale_param1", type=float, default=25.0)
    parser.add_argument("--scale_param2", type=float, default=0.12)
    parser.add_argument("--scale_param4", type=float, default=3.0)
    parser.add_argument("--scale_param5_deg", type=float, default=0.6)
    parser.add_argument("--scale_pri_us", type=float, default=8.0)
    parser.add_argument("--weight_param1", type=float, default=0.25)
    parser.add_argument("--weight_param2", type=float, default=0.45)
    parser.add_argument("--weight_param4", type=float, default=0.45)
    parser.add_argument("--weight_param5", type=float, default=1.20)
    parser.add_argument("--weight_pri", type=float, default=0.80)
    parser.add_argument("--weight_time_gap", type=float, default=0.15)
    parser.add_argument("--tsr_band_split_auto", action="store_true", default=False)
    parser.add_argument("--tsr_reduce_merge_thresh", type=float, default=0.2)
    parser.add_argument("--tsr_rescue_merge_thresh", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail_on_sort_error", action="store_true")
    parser.add_argument("--use_raw_on_sort_error", action="store_true", default=True)
    parser.add_argument("--no_use_raw_on_sort_error", dest="use_raw_on_sort_error", action="store_false")
    parser.add_argument("--keep_temp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.temp_dir.exists() and not bool(args.keep_temp):
        shutil.rmtree(args.temp_dir)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_pdw_columns(read_pdw(args.input_file, int(args.max_rows)))
    tracks: Dict[int, TrackState] = {}
    next_global_id = 1
    summary_rows: List[Dict[str, object]] = []
    association_rows: List[Dict[str, object]] = []

    for beat_id, beat_df in iter_beats(df, float(args.beat_seconds)):
        print(f"[beat] {beat_id}: pulses={len(beat_df)}")
        local_sigidx, local_sort_status, local_sort_error = run_local_sort(beat_df, beat_id, args.temp_dir, args)
        batch_df = summarize_local_batches(beat_df, local_sigidx, int(args.min_batch_pulses))
        assignments, next_global_id, assoc = associate_batches(beat_id, batch_df, tracks, next_global_id, args)
        association_rows.extend(assoc)

        global_sigidx = np.zeros((len(beat_df),), dtype=np.int64)
        for local_id, global_id in assignments.items():
            global_sigidx[local_sigidx == int(local_id)] = int(global_id)

        out_path = args.output_dir / f"beat_{beat_id:06d}.txt"
        write_pdw_with_sigidx(out_path, beat_df, global_sigidx, str(args.sigidx_column))
        summary_rows.append(
            {
                "beat": int(beat_id),
                "output_file": str(out_path),
                "num_pulses": int(len(beat_df)),
                "num_local_batches": int(len(batch_df)),
                "local_sort_status": local_sort_status,
                "local_sort_error": local_sort_error,
                "num_global_batches_in_beat": int(len(set(int(v) for v in global_sigidx if int(v) > 0))),
                "num_noise_or_unassigned": int(np.sum(global_sigidx <= 0)),
            }
        )
        print(
            "[beat_result] "
            + json.dumps(
                {
                    "beat": int(beat_id),
                    "output_file": str(out_path),
                    "num_pulses": int(len(beat_df)),
                    "num_global_batches_in_beat": int(len(set(int(v) for v in global_sigidx if int(v) > 0))),
                    "num_noise_or_unassigned": int(np.sum(global_sigidx <= 0)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "streaming_200ms_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(association_rows).to_csv(args.output_dir / "streaming_200ms_association.csv", index=False, encoding="utf-8-sig")
    final_beat = int(summary_rows[-1]["beat"]) if summary_rows else 0
    max_gap = int(args.max_track_gap_beats)
    track_rows = []
    for track in tracks.values():
        gap = final_beat - int(track.last_seen_beat)
        memory_state = "active" if max_gap <= 0 or gap <= max_gap else "archive"
        track_rows.append(
            {
                "global_sigidx": int(track.global_sigidx),
                "last_seen_beat": int(track.last_seen_beat),
                "beats_since_seen": int(gap),
                "num_seen_beats": int(track.num_seen_beats),
                "total_pulses": int(track.total_pulses),
                "memory_state": memory_state,
                "num_prototypes": int(len(track.prototypes)),
                **{f"feature_{k}": float(v) for k, v in track.features.items()},
            }
        )
    pd.DataFrame(track_rows).to_csv(args.output_dir / "streaming_200ms_tracks.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "streaming_200ms_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not bool(args.keep_temp):
        shutil.rmtree(args.temp_dir, ignore_errors=True)
    print(f"[done] wrote beat files and reports to {args.output_dir}")


if __name__ == "__main__":
    main()

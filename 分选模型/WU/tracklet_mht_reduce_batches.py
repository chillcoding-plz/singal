#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reduce over-fragmented front-end batches with tracklet-level MHT.

This is the right place to use MHT when a front-end sorter already exists.

Instead of doing MHT on every raw pulse, this script treats each existing
front-end batch ID inside each beat file as one measurement/tracklet:

    raw pulses -> front-end sorter -> many batch IDs -> this MHT reducer

The MHT state is a signal source represented by batch-level statistics:

    last_toa, pri_us, RF center, PW center, Param4 center, Param5 angle center

For every new tracklet, the reducer tries to associate it with an existing
source using:

    1. RF/PW/Param4/Param5 similarity
    2. PRI rhythm consistency across the time gap
    3. beat gap penalty

If no existing source can explain the tracklet, a new MHTId is created.

This reduces batch count while preserving the front-end sorter's local quality.
It is intentionally much faster and more stable than pulse-level MHT.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_PDW_COLUMNS = ["TOA(s)", "Param1", "Param2", "Param3", "Param4", "Param5", "Param6", "Param7", "SigIdx"]
DEFAULT_INPUT_DIR = Path("outputs_streaming_200ms_sample1_main5_stream")
DEFAULT_OUTPUT_DIR = Path("outputs_tracklet_mht_reduced_sample1_main5_stream")
DEFAULT_ID_COLUMN = "OurPredID"
OUTPUT_ID_COLUMN = "MHTId"


@dataclass
class Tracklet:
    """One front-end batch summarized as an MHT measurement."""

    beat: int
    local_sigidx: int
    row_indices: np.ndarray
    num_pulses: int
    start_toa: float
    end_toa: float
    pri_us: float
    p1: float
    p2: float
    p4: float
    p5_deg: float


@dataclass
class SourceTrack:
    """One reduced output source track."""

    output_sigidx: int
    last_beat: int
    last_toa: float
    pri_us: float
    p1: float
    p2: float
    p4: float
    p5_deg: float
    num_tracklets: int = 0
    total_pulses: int = 0
    score: float = 0.0
    history: List[Tuple[int, int]] = field(default_factory=list)


def angle_mean_deg(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    rad = np.deg2rad(np.mod(values, 360.0))
    return float((np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360.0) % 360.0)


def angle_delta_deg(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 360.0
    return float(min(delta, 360.0 - delta))


def robust_pri_us(toa: np.ndarray) -> float:
    toa = np.sort(np.asarray(toa, dtype=np.float64))
    dtoa = np.diff(toa) * 1e6
    dtoa = dtoa[np.isfinite(dtoa) & (dtoa > 0.0)]
    if len(dtoa) == 0:
        return float("nan")
    med = float(np.median(dtoa))
    if len(dtoa) >= 5:
        cap = max(5.0 * med, float(np.quantile(dtoa, 0.90)))
        dtoa = dtoa[dtoa <= cap]
    return float(np.median(dtoa)) if len(dtoa) else med


def parse_beat_id(path: Path, fallback: int) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else int(fallback)


def list_beat_files(input_dir: Path) -> List[Path]:
    """Return beat files in processing order."""

    files = sorted(input_dir.glob("beat_*.txt"))
    if not files:
        raise FileNotFoundError(f"No beat_*.txt files found in {input_dir}")
    return files


def read_one_beat_file(path: Path, fallback: int, global_offset: int) -> pd.DataFrame:
    """Read one beat file and add internal bookkeeping columns."""

    df = pd.read_csv(path, sep=r"\s+", engine="python")
    # Keep every input column. Some front-end outputs contain both the original
    # SigIdx and an additional prediction column such as OurPredID. The MHT
    # reducer appends MHTId instead of overwriting either one.
    missing = [col for col in BASE_PDW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df = df.copy()
    df["_beat"] = parse_beat_id(path, fallback)
    df["_source_file"] = path.name
    df["_row_in_file"] = np.arange(len(df), dtype=np.int64)
    df["_global_row"] = np.arange(global_offset, global_offset + len(df), dtype=np.int64)
    return df


def read_beat_files(input_dir: Path) -> pd.DataFrame:
    """Read all beat_*.txt files from an existing sorter output directory."""

    files = list_beat_files(input_dir)
    frames = []
    global_offset = 0
    for fallback, path in enumerate(files):
        df = read_one_beat_file(path, fallback, global_offset)
        global_offset += len(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_tracklets(df: pd.DataFrame, min_pulses: int, group_mode: str, id_column: str) -> List[Tracklet]:
    """Convert existing front-end batch IDs into tracklet measurements."""

    tracklets: List[Tracklet] = []
    if id_column not in df.columns:
        raise ValueError(f"Input beat files do not contain id column: {id_column}")

    # Sorting result 0 is a valid front-end result in this workflow, so it must
    # be included as a tracklet instead of being treated as background noise.
    # Negative IDs are still ignored as invalid sentinel values.
    valid = df[df[id_column] >= 0]
    if group_mode == "global":
        # Default mode: one global front-end ID is one tracklet, even if it spans
        # several beat files. This is the intended mode for OurPredID streams.
        groups = ((int(batch_id), sub) for batch_id, sub in valid.groupby(id_column, sort=True))
    else:
        # Optional mode: split the same front-end ID by beat before reduction.
        groups = (
            ((int(beat), int(batch_id)), sub)
            for (beat, batch_id), sub in valid.groupby(["_beat", id_column], sort=True)
        )

    for key, sub in groups:
        if group_mode == "global":
            sigidx = int(key)
            beat = int(sub["_beat"].min())
        else:
            beat, sigidx = key
        if len(sub) < int(min_pulses):
            continue
        toa = sub["TOA(s)"].to_numpy(dtype=np.float64)
        tracklets.append(
            Tracklet(
                beat=int(beat),
                local_sigidx=int(sigidx),
                row_indices=sub.index.to_numpy(dtype=np.int64),
                num_pulses=int(len(sub)),
                start_toa=float(np.min(toa)),
                end_toa=float(np.max(toa)),
                pri_us=robust_pri_us(toa),
                p1=float(sub["Param1"].median()),
                p2=float(sub["Param2"].median()),
                p4=float(sub["Param4"].median()),
                p5_deg=angle_mean_deg(sub["Param5"].to_numpy(dtype=np.float64)),
            )
        )
    tracklets.sort(key=lambda t: (t.beat, t.start_toa, t.local_sigidx))
    return tracklets


def feature_distance(track: SourceTrack, t: Tracklet, args: argparse.Namespace) -> float:
    """Normalized batch-to-track distance."""

    d1 = abs(t.p1 - track.p1) / max(float(args.scale_p1), 1e-9)
    d2 = abs(t.p2 - track.p2) / max(float(args.scale_p2), 1e-9)
    d4 = abs(t.p4 - track.p4) / max(float(args.scale_p4), 1e-9)
    d5 = angle_delta_deg(t.p5_deg, track.p5_deg) / max(float(args.scale_p5_deg), 1e-9)
    if np.isfinite(t.pri_us) and np.isfinite(track.pri_us) and track.pri_us > 0:
        dpri = abs(t.pri_us - track.pri_us) / max(float(args.scale_pri_us), 1e-9)
    else:
        dpri = 0.0
    return float(
        math.sqrt(
            float(args.weight_p1) * d1 * d1
            + float(args.weight_p2) * d2 * d2
            + float(args.weight_p4) * d4 * d4
            + float(args.weight_p5) * d5 * d5
            + float(args.weight_pri) * dpri * dpri
        )
    )


def rhythm_penalty(track: SourceTrack, t: Tracklet, args: argparse.Namespace) -> float:
    """Penalty for PRI rhythm inconsistency between a source track and a tracklet."""

    if not np.isfinite(track.pri_us) or track.pri_us <= 0:
        return 0.0
    gap_us = max((t.start_toa - track.last_toa) * 1e6, 0.0)
    if gap_us <= 0.0:
        return 0.0
    steps = max(int(round(gap_us / track.pri_us)), 1)
    if steps > int(args.max_pri_steps):
        return float("inf")
    residual = abs(gap_us - steps * track.pri_us)
    return float(residual / max(float(args.rhythm_gate_us), 1e-9))


def association_cost(track: SourceTrack, t: Tracklet, args: argparse.Namespace) -> float:
    fdist = feature_distance(track, t, args)
    if fdist > float(args.feature_gate):
        return float("inf")
    rpen = rhythm_penalty(track, t, args)
    if not np.isfinite(rpen) or rpen > float(args.rhythm_gate_sigma):
        return float("inf")
    beat_gap = max(int(t.beat) - int(track.last_beat), 0)
    if beat_gap > int(args.max_beat_gap):
        return float("inf")
    return float(fdist + float(args.weight_rhythm) * rpen + float(args.weight_beat_gap) * beat_gap)


def update_track(track: SourceTrack, t: Tracklet, args: argparse.Namespace) -> None:
    """Update a source track with an associated tracklet."""

    alpha = float(args.update_alpha)
    track.last_beat = int(t.beat)
    track.last_toa = float(t.end_toa)
    if np.isfinite(t.pri_us):
        if np.isfinite(track.pri_us) and track.pri_us > 0:
            track.pri_us = (1.0 - alpha) * track.pri_us + alpha * t.pri_us
        else:
            track.pri_us = float(t.pri_us)
    track.p1 = (1.0 - alpha) * track.p1 + alpha * t.p1
    track.p2 = (1.0 - alpha) * track.p2 + alpha * t.p2
    track.p4 = (1.0 - alpha) * track.p4 + alpha * t.p4
    delta = ((t.p5_deg - track.p5_deg + 180.0) % 360.0) - 180.0
    track.p5_deg = float((track.p5_deg + alpha * delta) % 360.0)
    track.num_tracklets += 1
    track.total_pulses += int(t.num_pulses)
    track.history.append((int(t.beat), int(t.local_sigidx)))


def new_track(output_id: int, t: Tracklet) -> SourceTrack:
    return SourceTrack(
        output_sigidx=int(output_id),
        last_beat=int(t.beat),
        last_toa=float(t.end_toa),
        pri_us=float(t.pri_us) if np.isfinite(t.pri_us) else float("nan"),
        p1=float(t.p1),
        p2=float(t.p2),
        p4=float(t.p4),
        p5_deg=float(t.p5_deg),
        num_tracklets=1,
        total_pulses=int(t.num_pulses),
        history=[(int(t.beat), int(t.local_sigidx))],
    )


def process_tracklets_into_tracks(
    tracklets: List[Tracklet],
    args: argparse.Namespace,
    tracks: List[SourceTrack],
    next_id: int,
    input_id_to_output_id: Optional[Dict[int, int]] = None,
    processed_offset: int = 0,
    total_tracklets: Optional[int] = None,
) -> Tuple[Dict[Tuple[int, int], int], List[Dict[str, object]], int]:
    """Associate one beat's tracklets into the persistent MHT track state.

    The important streaming behavior is that ``tracks`` is passed in from the
    previous beat and updated in place. Each beat receives a local mapping from
    (beat, input_id) to the reduced MHTId, so output can be written immediately.
    """

    mapping: Dict[Tuple[int, int], int] = {}
    decisions: List[Dict[str, object]] = []
    total = int(total_tracklets) if total_tracklets is not None else int(len(tracklets))

    for pos, t in enumerate(tracklets, start=1):
        global_pos = int(processed_offset) + int(pos)
        if global_pos == 1 or global_pos % 1000 == 0:
            total_label = str(total) if total_tracklets is not None else "?"
            print(f"[tracklet-mht] processing {global_pos}/{total_label} beat={t.beat}", flush=True)
        key = (int(t.beat), int(t.local_sigidx))

        # In beat-by-beat streaming mode, the same front-end ID can appear in
        # many beat files. It must keep the same MHTId; otherwise one original
        # OurPredID would be split into many new MHT tracks and the batch count
        # would grow instead of shrink.
        if input_id_to_output_id is not None and int(t.local_sigidx) in input_id_to_output_id:
            output_id = int(input_id_to_output_id[int(t.local_sigidx)])
            track = next((item for item in tracks if int(item.output_sigidx) == output_id), None)
            mapping[key] = output_id
            if track is not None:
                update_track(track, t, args)
            decisions.append(
                {
                    "beat": int(t.beat),
                    "local_sigidx": int(t.local_sigidx),
                    "output_sigidx": output_id,
                    "decision": "same_input_id",
                    "cost": 0.0,
                    "num_candidates": 1 if track is not None else 0,
                }
            )
            continue

        active_tracks = [
            track
            for track in tracks
            if int(t.beat) - int(track.last_beat) <= int(args.max_beat_gap)
        ]
        candidates = []
        for track in active_tracks:
            if abs(t.p1 - track.p1) > float(args.prefilter_p1):
                continue
            if abs(t.p2 - track.p2) > float(args.prefilter_p2):
                continue
            if abs(t.p4 - track.p4) > float(args.prefilter_p4):
                continue
            if angle_delta_deg(t.p5_deg, track.p5_deg) > float(args.prefilter_p5_deg):
                continue
            cost = association_cost(track, t, args)
            if np.isfinite(cost):
                candidates.append((cost, track))
        candidates.sort(key=lambda item: item[0])

        if candidates:
            cost, best = candidates[0]
            mapping[key] = int(best.output_sigidx)
            if input_id_to_output_id is not None:
                input_id_to_output_id[int(t.local_sigidx)] = int(best.output_sigidx)
            update_track(best, t, args)
            decisions.append(
                {
                    "beat": int(t.beat),
                    "local_sigidx": int(t.local_sigidx),
                    "output_sigidx": int(best.output_sigidx),
                    "decision": "merge",
                    "cost": float(cost),
                    "num_candidates": int(len(candidates)),
                }
            )
        else:
            track = new_track(next_id, t)
            tracks.append(track)
            mapping[key] = int(next_id)
            if input_id_to_output_id is not None:
                input_id_to_output_id[int(t.local_sigidx)] = int(next_id)
            decisions.append(
                {
                    "beat": int(t.beat),
                    "local_sigidx": int(t.local_sigidx),
                    "output_sigidx": int(next_id),
                    "decision": "new",
                    "cost": np.nan,
                    "num_candidates": 0,
                }
            )
            next_id += 1

    return mapping, decisions, next_id


def reduce_tracklets(tracklets: List[Tracklet], args: argparse.Namespace) -> Tuple[Dict[Tuple[int, int], int], List[SourceTrack], List[Dict[str, object]]]:
    """MHT-inspired greedy association over tracklet measurements.

    This is intentionally tracklet-level. The front-end sorter handles local
    pulse clustering; this reducer handles over-fragmentation.
    """

    tracks: List[SourceTrack] = []
    mapping: Dict[Tuple[int, int], int] = {}
    decisions: List[Dict[str, object]] = []
    next_id = 1

    mapping, decisions, next_id = process_tracklets_into_tracks(
        tracklets,
        args,
        tracks,
        next_id,
        processed_offset=0,
        total_tracklets=len(tracklets),
    )

    return mapping, tracks, decisions


def apply_mapping(df: pd.DataFrame, mapping: Dict[Tuple[int, int], int], id_column: str) -> np.ndarray:
    out = np.zeros((len(df),), dtype=np.int64)
    for (beat, sigidx), sub in df[df[id_column] >= 0].groupby(["_beat", id_column], sort=False):
        value = int(mapping.get((int(beat), int(sigidx)), mapping.get((-1, int(sigidx)), 0)))
        out[sub.index.to_numpy(dtype=np.int64)] = value
    return out


def write_reduced_beat_files(df: pd.DataFrame, mht_id: np.ndarray, output_dir: Path) -> None:
    """Write beat files with the new MHTId appended as the final column."""

    output_dir.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    if OUTPUT_ID_COLUMN in work.columns:
        work = work.drop(columns=[OUTPUT_ID_COLUMN])
    work[OUTPUT_ID_COLUMN] = mht_id.astype(np.int64)
    for beat, sub in work.groupby("_beat", sort=True):
        drop_cols = [col for col in ["_beat", "_source_file", "_row_in_file", "_global_row"] if col in sub.columns]
        out = sub.drop(columns=drop_cols).copy()
        out.to_csv(output_dir / f"beat_{int(beat):06d}.txt", sep=" ", index=False, float_format="%.9f")


def write_reduced_beat_file(df: pd.DataFrame, mht_id: np.ndarray, output_dir: Path) -> None:
    """Write the current beat immediately with MHTId as the final column."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        return
    work = df.copy()
    if OUTPUT_ID_COLUMN in work.columns:
        work = work.drop(columns=[OUTPUT_ID_COLUMN])
    work[OUTPUT_ID_COLUMN] = mht_id.astype(np.int64)
    beat = int(work["_beat"].iloc[0])
    drop_cols = [col for col in ["_beat", "_source_file", "_row_in_file", "_global_row"] if col in work.columns]
    out = work.drop(columns=drop_cols).copy()
    out.to_csv(output_dir / f"beat_{beat:06d}.txt", sep=" ", index=False, float_format="%.9f")


def write_reports(
    output_dir: Path,
    original_df: pd.DataFrame,
    mht_id: np.ndarray,
    tracks: List[SourceTrack],
    decisions: List[Dict[str, object]],
    args: argparse.Namespace,
) -> None:
    rows = []
    for track in tracks:
        rows.append(
            {
                "output_sigidx": int(track.output_sigidx),
                "num_tracklets": int(track.num_tracklets),
                "total_pulses": int(track.total_pulses),
                "last_beat": int(track.last_beat),
                "pri_us": float(track.pri_us) if np.isfinite(track.pri_us) else np.nan,
                "p1": float(track.p1),
                "p2": float(track.p2),
                "p4": float(track.p4),
                "p5_deg": float(track.p5_deg),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "tracklet_mht_tracks.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(decisions).to_csv(output_dir / "tracklet_mht_decisions.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for beat, sub in original_df.groupby("_beat", sort=True):
        idx = sub.index.to_numpy(dtype=np.int64)
        old_n = len(set(int(v) for v in sub[args.id_column].to_numpy(dtype=np.int64) if int(v) >= 0))
        new_n = len(set(int(v) for v in mht_id[idx] if int(v) >= 0))
        summary_rows.append(
            {
                "beat": int(beat),
                "num_pulses": int(len(sub)),
                "input_id_column": str(args.id_column),
                "old_batches": int(old_n),
                "new_batches": int(new_n),
                "reduction": int(old_n - new_n),
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "tracklet_mht_summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "tracklet_mht_config.json").write_text(
        json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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


def infer_truth_file(input_dir: Path) -> Optional[Path]:
    text = str(input_dir).replace("\\", "/").lower()
    if "sample2" in text or "sample_2" in text:
        return Path("edata/Test_Data/Sample_2/Sorted_PDW.txt")
    if "sample1" in text or "sample_1" in text:
        return Path("edata/Test_Data/Sample_1/Sorted_PDW.txt")
    return None


def evaluate_accuracy(
    output_dir: Path,
    df: pd.DataFrame,
    mht_id: np.ndarray,
    truth_file: Optional[Path],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, object]]:
    """Compute sorting metrics for SigIdx, OurPredID, and MHTId.

    All three prediction ID columns are compared against the same official
    sorted truth ID: ``Sorted_PDW.txt`` column ``SigIdx``.
    """

    if truth_file is None or not truth_file.exists():
        print(f"[metrics] skipped: truth file not found ({truth_file})")
        return {}

    import sort_metrics as metrics_mod

    truth = pd.read_csv(truth_file, sep=r"\s+", engine="python", nrows=len(df)).iloc[:, :3].copy()
    truth.columns = ["TOA(s)", "SigIdx", "LABEL"]
    if len(truth) != len(df):
        raise ValueError(f"Truth rows ({len(truth)}) and output rows ({len(df)}) do not match.")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_sources: List[Tuple[str, np.ndarray]] = []
    if "SigIdx" in df.columns:
        pred_sources.append(("SigIdx", df["SigIdx"].to_numpy(dtype=np.int64)))
    if str(args.id_column) in df.columns and str(args.id_column) != "SigIdx":
        pred_sources.append((str(args.id_column), df[str(args.id_column)].to_numpy(dtype=np.int64)))
    pred_sources.append((OUTPUT_ID_COLUMN, mht_id.astype(np.int64)))

    all_metrics: Dict[str, Dict[str, object]] = {}
    comparison_rows = []
    for pred_name, pred_ids in pred_sources:
        # The displayed batch count includes ID 0. The metric implementation
        # still follows sort_metrics.py's official rule internally.
        ids = np.unique(pred_ids[pred_ids >= 0])
        batch_stub = pd.DataFrame({"pred_sigidx": ids, "batch_pred_label": np.full(len(ids), 99)})
        batch_df, target_df, target_beat_df, beat_df, metrics = metrics_mod.compute_sort_metrics_by_beat(
            df["TOA(s)"].to_numpy(dtype=np.float64),
            truth["SigIdx"].to_numpy(dtype=np.int64),
            pred_ids.astype(np.int64),
            truth["LABEL"].to_numpy(dtype=np.int64),
            float(args.sort_purity_threshold),
            float(args.sort_min_target_fraction),
            int(args.sort_mix_fail_min_pulses),
            batch_stub,
            float(args.beat_seconds),
        )
        metrics = dict(metrics)
        all_metrics[pred_name] = metrics
        comparison_rows.append(
            {
                "pred_id_column": pred_name,
                "num_pred_batches": int(len(ids)),
                "sort_acc": metrics.get("sample_sort_acc"),
                "MR": metrics.get("sample_mr"),
                "MP": metrics.get("sample_mp"),
                "MIOU": metrics.get("sample_miou"),
                "wrong_batch_rate": metrics.get("sample_wrong_batch_rate"),
                "extra_batch_rate": metrics.get("sample_extra_batch_rate"),
                "tracking": metrics.get("sample_signal_tracking_stability"),
            }
        )

        safe_name = pred_name.lower()
        (output_dir / f"tracklet_mht_metrics_{safe_name}.json").write_text(
            json.dumps(json_safe(metrics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        beat_df.to_csv(output_dir / f"tracklet_mht_beat_accuracy_{safe_name}.csv", index=False, encoding="utf-8-sig")
        target_df.to_csv(output_dir / f"tracklet_mht_target_accuracy_{safe_name}.csv", index=False, encoding="utf-8-sig")
        if pred_name == OUTPUT_ID_COLUMN:
            # Keep the old filenames for compatibility with earlier runs.
            (output_dir / "tracklet_mht_metrics.json").write_text(
                json.dumps(json_safe(metrics), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            beat_df.to_csv(output_dir / "tracklet_mht_beat_accuracy.csv", index=False, encoding="utf-8-sig")
            target_df.to_csv(output_dir / "tracklet_mht_target_accuracy.csv", index=False, encoding="utf-8-sig")
        if bool(args.save_detailed_metrics):
            batch_df.to_csv(output_dir / f"tracklet_mht_batch_eval_{safe_name}.csv", index=False, encoding="utf-8-sig")
            target_beat_df.to_csv(output_dir / f"tracklet_mht_target_beat_eval_{safe_name}.csv", index=False, encoding="utf-8-sig")

        print(
            f"[metrics:{pred_name}] "
            f"sort_acc={float(metrics.get('sample_sort_acc', float('nan'))):.4f}, "
            f"MR={float(metrics.get('sample_mr', float('nan'))):.4f}, "
            f"MP={float(metrics.get('sample_mp', float('nan'))):.4f}, "
            f"MIOU={float(metrics.get('sample_miou', float('nan'))):.4f}"
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "tracklet_mht_id_accuracy_compare.csv", index=False, encoding="utf-8-sig")
    (output_dir / "tracklet_mht_id_accuracy_compare.json").write_text(
        json.dumps(json_safe(comparison_rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return all_metrics


def write_overall_summary(
    output_dir: Path,
    args: argparse.Namespace,
    input_pulses: int,
    tracklets: int,
    old_batches: int,
    new_batches: int,
    metrics: Dict[str, Dict[str, object]],
) -> None:
    """Store one compact whole-sample summary: accuracy plus batch reduction."""

    summary = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_id_column": str(args.id_column),
        "output_id_column": OUTPUT_ID_COLUMN,
        "input_pulses": int(input_pulses),
        "tracklets": int(tracklets),
        "old_batches": int(old_batches),
        "new_batches": int(new_batches),
        "reduction": int(old_batches - new_batches),
        "reduction_rate": float((old_batches - new_batches) / old_batches) if old_batches > 0 else 0.0,
    }
    for pred_name in ["SigIdx", str(args.id_column), OUTPUT_ID_COLUMN]:
        if pred_name not in metrics:
            continue
        prefix = pred_name.lower()
        pred_metrics = metrics[pred_name]
        summary[f"{prefix}_sort_acc"] = pred_metrics.get("sample_sort_acc")
        summary[f"{prefix}_mr"] = pred_metrics.get("sample_mr")
        summary[f"{prefix}_mp"] = pred_metrics.get("sample_mp")
        summary[f"{prefix}_miou"] = pred_metrics.get("sample_miou")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tracklet_mht_overall_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame([summary]).to_csv(output_dir / "tracklet_mht_overall_summary.csv", index=False, encoding="utf-8-sig")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduce over-fragmented front-end batch IDs with tracklet-level MHT.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing beat_*.txt from a front-end sorter.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id_column", type=str, default=DEFAULT_ID_COLUMN, help="Front-end batch id column to reduce, default: OurPredID.")
    parser.add_argument("--truth_file", type=Path, default=None)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--save_detailed_metrics", action="store_true")
    parser.add_argument("--beat_seconds", type=float, default=0.2)
    parser.add_argument("--sort_purity_threshold", type=float, default=0.90)
    parser.add_argument("--sort_min_target_fraction", type=float, default=0.10)
    parser.add_argument("--sort_mix_fail_min_pulses", type=int, default=1)
    parser.add_argument("--group_mode", choices=["global", "beat"], default="global", help="Use global front-end IDs by default; beat mode splits each ID per beat.")
    parser.add_argument("--min_tracklet_pulses", type=int, default=3)
    parser.add_argument("--max_beat_gap", type=int, default=8)
    parser.add_argument("--max_pri_steps", type=int, default=80)
    parser.add_argument("--feature_gate", type=float, default=3.0)
    parser.add_argument("--prefilter_p1", type=float, default=250.0)
    parser.add_argument("--prefilter_p2", type=float, default=1.0)
    parser.add_argument("--prefilter_p4", type=float, default=25.0)
    parser.add_argument("--prefilter_p5_deg", type=float, default=10.0)
    parser.add_argument("--rhythm_gate_us", type=float, default=8.0)
    parser.add_argument("--rhythm_gate_sigma", type=float, default=3.0)
    parser.add_argument("--scale_p1", type=float, default=80.0)
    parser.add_argument("--scale_p2", type=float, default=0.35)
    parser.add_argument("--scale_p4", type=float, default=8.0)
    parser.add_argument("--scale_p5_deg", type=float, default=3.0)
    parser.add_argument("--scale_pri_us", type=float, default=10.0)
    parser.add_argument("--weight_p1", type=float, default=0.4)
    parser.add_argument("--weight_p2", type=float, default=0.5)
    parser.add_argument("--weight_p4", type=float, default=0.9)
    parser.add_argument("--weight_p5", type=float, default=0.2)
    parser.add_argument("--weight_pri", type=float, default=0.5)
    parser.add_argument("--weight_rhythm", type=float, default=0.8)
    parser.add_argument("--weight_beat_gap", type=float, default=0.05)
    parser.add_argument("--update_alpha", type=float, default=0.25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = list_beat_files(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracks: List[SourceTrack] = []
    decisions: List[Dict[str, object]] = []
    frames: List[pd.DataFrame] = []
    mht_chunks: List[np.ndarray] = []
    old_batch_ids = set()
    new_batch_ids = set()
    input_id_to_output_id: Dict[int, int] = {}
    next_id = 1
    global_offset = 0
    total_tracklets = 0

    print(f"[tracklet-mht] streaming beats={len(files)}, id_column={args.id_column}", flush=True)
    for fallback, path in enumerate(files):
        df = read_one_beat_file(path, fallback, global_offset)
        if str(args.id_column) not in df.columns:
            raise ValueError(f"{path} does not contain id column: {args.id_column}")

        beat = int(df["_beat"].iloc[0]) if len(df) else int(fallback)
        tracklets = summarize_tracklets(df, int(args.min_tracklet_pulses), str(args.group_mode), str(args.id_column))
        print(
            f"[beat] {beat}: pulses={len(df)}, tracklets={len(tracklets)}, active_tracks={len(tracks)}",
            flush=True,
        )
        mapping, beat_decisions, next_id = process_tracklets_into_tracks(
            tracklets,
            args,
            tracks,
            next_id,
            input_id_to_output_id=input_id_to_output_id,
            processed_offset=total_tracklets,
            total_tracklets=None,
        )
        mht_id = apply_mapping(df, mapping, str(args.id_column))
        write_reduced_beat_file(df, mht_id, args.output_dir)

        old_batch_ids.update(int(v) for v in df[str(args.id_column)].to_numpy(dtype=np.int64) if int(v) >= 0)
        new_batch_ids.update(int(v) for v in mht_id if int(v) >= 0)
        decisions.extend(beat_decisions)
        frames.append(df)
        mht_chunks.append(mht_id)
        total_tracklets += len(tracklets)
        global_offset += len(df)

    df = pd.concat(frames, ignore_index=True)
    mht_id = np.concatenate(mht_chunks).astype(np.int64) if mht_chunks else np.zeros((0,), dtype=np.int64)
    old_batches = len(old_batch_ids)
    new_batches = len(new_batch_ids)
    print(f"[tracklet-mht] input_pulses={len(df)}, tracklets={total_tracklets}", flush=True)
    write_reports(args.output_dir, df, mht_id, tracks, decisions, args)
    metrics: Dict[str, Dict[str, object]] = {}
    if not bool(args.skip_metrics):
        truth_file = Path(args.truth_file) if args.truth_file is not None else infer_truth_file(args.input_dir)
        metrics = evaluate_accuracy(args.output_dir, df, mht_id, truth_file, args)
    write_overall_summary(args.output_dir, args, len(df), total_tracklets, old_batches, new_batches, metrics)
    print(f"[done] old_batches={old_batches}, new_batches={new_batches}, output={args.output_dir}")


if __name__ == "__main__":
    main()

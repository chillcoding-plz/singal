#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sorting-only metrics for PDW deinterleaving.

This module intentionally has no XGBoost or open-set recognition dependency.
It evaluates the official sorting-style metrics by 200 ms beats and adds a
tracking stability score based on target loss and SigIdx switching.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def majority_value(values: np.ndarray) -> int:
    uniq, counts = np.unique(values.astype(np.int64), return_counts=True)
    return int(uniq[int(np.argmax(counts))])


def _safe_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _safe_ratio(numer: int, denom: int, zero_value: float = 0.0) -> float:
    return float(numer / denom) if denom > 0 else float(zero_value)


def compute_signal_tracking_stability(target_beat_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute target-level tracking stability from beat-level representatives.

    For each true target, a beat is considered tracked when it has at least one
    successful sorting batch and therefore a representative predicted SigIdx.
    The final score follows the contest document:

        signal_tracking_stability = 1 - sigidx_switch_rate - target_loss_rate

    The detailed loss and switch rates are returned separately so the score is
    auditable rather than a hidden magic number.
    """
    empty_metrics = {
        "sample_signal_tracking_stability": float("nan"),
        "sample_target_loss_rate": float("nan"),
        "sample_sigidx_switch_rate": float("nan"),
        "sample_transition_break_rate": float("nan"),
        "tracking_active_target_beats": 0,
        "tracking_tracked_target_beats": 0,
        "tracking_lost_target_beats": 0,
        "tracking_sigidx_switches": 0,
        "tracking_comparable_success_pairs": 0,
        "tracking_transition_breaks": 0,
        "tracking_total_transitions": 0,
    }
    if len(target_beat_df) == 0:
        return pd.DataFrame(), empty_metrics

    rows = []
    for true_id, sub in target_beat_df.groupby("true_sigidx", sort=True):
        sub = sub.sort_values("beat", kind="mergesort").reset_index(drop=True)
        representative = pd.to_numeric(sub["representative_pred_sigidx"], errors="coerce").to_numpy(dtype=np.float64)
        tracked = sub["sort_success"].astype(bool).to_numpy() & np.isfinite(representative)

        active_beats = int(len(sub))
        tracked_beats = int(tracked.sum())
        lost_beats = int(active_beats - tracked_beats)
        target_loss_rate = float(lost_beats / active_beats) if active_beats else float("nan")
        tracked_fraction = float(tracked_beats / active_beats) if active_beats else float("nan")

        comparable_success_pairs = 0
        sigidx_switches = 0
        transition_breaks = 0
        total_transitions = max(active_beats - 1, 0)
        for pos in range(1, active_beats):
            prev_ok = bool(tracked[pos - 1])
            curr_ok = bool(tracked[pos])
            if prev_ok and curr_ok:
                comparable_success_pairs += 1
                if int(representative[pos - 1]) != int(representative[pos]):
                    sigidx_switches += 1
                    transition_breaks += 1
            else:
                transition_breaks += 1

        sigidx_switch_rate = float(sigidx_switches / active_beats) if active_beats else float("nan")
        transition_break_rate = (
            float(transition_breaks / total_transitions)
            if total_transitions > 0
            else 0.0
        )
        signal_tracking_stability = float(np.clip(1.0 - sigidx_switch_rate - target_loss_rate, 0.0, 1.0))

        rows.append(
            {
                "true_sigidx": int(true_id),
                "tracking_active_beats": active_beats,
                "tracking_tracked_beats": tracked_beats,
                "tracking_lost_beats": lost_beats,
                "target_loss_rate": target_loss_rate,
                "tracked_fraction": tracked_fraction,
                "sigidx_switches": int(sigidx_switches),
                "comparable_success_pairs": int(comparable_success_pairs),
                "sigidx_switch_rate": sigidx_switch_rate,
                "transition_breaks": int(transition_breaks),
                "total_transitions": int(total_transitions),
                "transition_break_rate": transition_break_rate,
                "signal_tracking_stability": signal_tracking_stability,
            }
        )

    tracking_df = pd.DataFrame(rows)
    metrics = {
        "sample_signal_tracking_stability": _safe_mean(tracking_df["signal_tracking_stability"]),
        "sample_target_loss_rate": _safe_mean(tracking_df["target_loss_rate"]),
        "sample_sigidx_switch_rate": _safe_mean(tracking_df["sigidx_switch_rate"]),
        "sample_transition_break_rate": _safe_mean(tracking_df["transition_break_rate"]),
        "tracking_active_target_beats": int(tracking_df["tracking_active_beats"].sum()),
        "tracking_tracked_target_beats": int(tracking_df["tracking_tracked_beats"].sum()),
        "tracking_lost_target_beats": int(tracking_df["tracking_lost_beats"].sum()),
        "tracking_sigidx_switches": int(tracking_df["sigidx_switches"].sum()),
        "tracking_comparable_success_pairs": int(tracking_df["comparable_success_pairs"].sum()),
        "tracking_transition_breaks": int(tracking_df["transition_breaks"].sum()),
        "tracking_total_transitions": int(tracking_df["total_transitions"].sum()),
    }
    return tracking_df, metrics


def compute_sort_metrics_by_beat(
    toa: np.ndarray,
    true_sigidx: np.ndarray,
    pred_sigidx: np.ndarray,
    true_labels: np.ndarray,
    purity_threshold: float,
    min_target_fraction: float,
    mix_fail_min_pulses: int,
    batch_df: pd.DataFrame,
    chunk_seconds: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    if len(toa) == 0:
        empty = pd.DataFrame()
        metrics = {
            "sample_sort_acc": float("nan"),
            "sample_extra_batch_rate": float("nan"),
            "sample_wrong_batch_rate": float("nan"),
            "sample_mr": float("nan"),
            "sample_mp": float("nan"),
            "sample_miou": float("nan"),
            "MR": float("nan"),
            "MP": float("nan"),
            "MIOU": float("nan"),
            "recognition_acc_on_success": float("nan"),
            "num_true_targets": 0,
            "num_pred_batches": 0,
            "num_beats": 0,
            "successful_target_beats": 0,
            "total_target_beats": 0,
            "wrong_batches": 0,
            "total_pred_batch_beats": 0,
            "extra_batch_total": 0,
            "purity_threshold": float(purity_threshold),
            "min_target_fraction": float(min_target_fraction),
            "mix_fail_min_pulses": int(mix_fail_min_pulses),
            "chunk_seconds": float(chunk_seconds),
        }
        metrics.update(compute_signal_tracking_stability(empty)[1])
        return empty, empty, empty, empty, metrics

    if chunk_seconds <= 0:
        beat_ids = np.zeros((len(toa),), dtype=np.int64)
    else:
        t0 = float(np.min(toa))
        beat_ids = np.floor((toa.astype(np.float64) - t0) / chunk_seconds).astype(np.int64)

    work = pd.DataFrame(
        {
            "beat": beat_ids,
            "true_sigidx": true_sigidx.astype(np.int64),
            "pred_sigidx": pred_sigidx.astype(np.int64),
            "true_label": true_labels.astype(np.int64),
        }
    )

    batch_lookup = {}
    recognition_available = (
        len(batch_df) > 0
        and "pred_sigidx" in batch_df.columns
        and "batch_pred_label" in batch_df.columns
    )
    if len(batch_df) > 0 and "pred_sigidx" in batch_df.columns:
        batch_lookup = batch_df.set_index("pred_sigidx").to_dict(orient="index")

    batch_rows = []
    target_beat_rows = []
    beat_rows = []

    for beat, beat_df in work.groupby("beat", sort=True):
        true_part = beat_df[beat_df["true_sigidx"] > 0]
        pred_part = beat_df[beat_df["pred_sigidx"] > 0]

        true_ids = sorted(int(v) for v in true_part["true_sigidx"].unique())
        pred_ids = sorted(int(v) for v in pred_part["pred_sigidx"].unique())

        true_counts = true_part["true_sigidx"].value_counts().to_dict()
        pred_counts = pred_part["pred_sigidx"].value_counts().to_dict()
        true_label_by_target = (
            true_part.groupby("true_sigidx")["true_label"]
            .agg(lambda s: majority_value(s.to_numpy(dtype=np.int64)))
            .to_dict()
            if len(true_part) > 0
            else {}
        )
        cross = (
            pred_part.groupby(["pred_sigidx", "true_sigidx"]).size()
            if len(pred_part) > 0
            else pd.Series(dtype=np.int64)
        )

        success_batches_by_target = {tid: [] for tid in true_ids}
        wrong_batch_count = 0

        for pred_id in pred_ids:
            n_i = int(pred_counts[pred_id])
            target_counts = {tid: int(cross.get((pred_id, tid), 0)) for tid in true_ids}
            if len(target_counts) == 0:
                continue

            matched_true_sigidx = max(target_counts, key=target_counts.get)
            matched_count = int(target_counts[matched_true_sigidx])
            purity = float(matched_count / max(n_i, 1))
            target_fraction = float(matched_count / max(int(true_counts[matched_true_sigidx]), 1))

            swallowed_targets = []
            for other_id, other_count in target_counts.items():
                other_id = int(other_id)
                other_count = int(other_count)
                if other_id == matched_true_sigidx:
                    continue
                if other_count == int(true_counts[other_id]) and int(true_counts[other_id]) > mix_fail_min_pulses:
                    swallowed_targets.append(other_id)

            wrong_batch = (purity < purity_threshold) or bool(swallowed_targets)
            if wrong_batch:
                wrong_batch_count += 1

            sort_success_for_matched_target = (
                purity >= purity_threshold
                and target_fraction >= min_target_fraction
                and not swallowed_targets
            )
            if sort_success_for_matched_target:
                batch_info = batch_lookup.get(pred_id, {})
                success_batches_by_target[matched_true_sigidx].append(
                    {
                        "pred_sigidx": pred_id,
                        "matched_count": matched_count,
                        "num_pulses": n_i,
                        "batch_pred_label": batch_info.get("batch_pred_label", np.nan),
                        "batch_max_prob": batch_info.get("batch_max_prob", np.nan),
                    }
                )

            batch_info = batch_lookup.get(pred_id, {})
            batch_rows.append(
                {
                    "beat": int(beat),
                    "pred_sigidx": int(pred_id),
                    "num_pulses": n_i,
                    "matched_true_sigidx": int(matched_true_sigidx),
                    "matched_true_label": int(true_label_by_target.get(matched_true_sigidx, -1)),
                    "matched_true_count": matched_count,
                    "purity_m_over_n": purity,
                    "target_fraction_m_over_N": target_fraction,
                    "sort_success_for_matched_target": bool(sort_success_for_matched_target),
                    "wrong_batch": bool(wrong_batch),
                    "fail_reason": (
                        "low_purity" if purity < purity_threshold
                        else "swallowed_other_target" if swallowed_targets
                        else "low_target_fraction" if target_fraction < min_target_fraction
                        else ""
                    ),
                    "swallowed_targets": ",".join(str(v) for v in swallowed_targets),
                    "batch_pred_label": batch_info.get("batch_pred_label", np.nan),
                    "batch_max_prob": batch_info.get("batch_max_prob", np.nan),
                    "is_batch_confident": batch_info.get("is_batch_confident", np.nan),
                }
            )

        for true_id in true_ids:
            success_batches = success_batches_by_target[true_id]
            sort_success = len(success_batches) > 0
            extra_batches = max(len(success_batches) - 1, 0)
            representative_pred_sigidx = np.nan
            representative_pred_label = np.nan
            representative_max_prob = np.nan
            recognition_success = False if recognition_available else np.nan
            if sort_success:
                representative = max(success_batches, key=lambda x: x["matched_count"])
                representative_pred_sigidx = int(representative["pred_sigidx"])
                representative_pred_label = representative["batch_pred_label"]
                representative_max_prob = representative["batch_max_prob"]
                recognition_success = (
                    False
                    if pd.isna(representative_pred_label)
                    else bool(representative_pred_label == int(true_label_by_target[true_id]))
                ) if recognition_available else np.nan

            target_beat_rows.append(
                {
                    "beat": int(beat),
                    "true_sigidx": int(true_id),
                    "target_label": int(true_label_by_target[true_id]),
                    "target_pulses_Nj": int(true_counts[true_id]),
                    "sort_success": bool(sort_success),
                    "num_success_batches": int(len(success_batches)),
                    "extra_batches": int(extra_batches),
                    "representative_pred_sigidx": representative_pred_sigidx,
                    "representative_pred_label": representative_pred_label,
                    "representative_max_prob": representative_max_prob,
                    "recognition_success": recognition_success,
                }
            )

        beat_rows.append(
            {
                "beat": int(beat),
                "num_true_targets": int(len(true_ids)),
                "num_pred_batches": int(len(pred_ids)),
                "wrong_batches": int(wrong_batch_count),
                "wrong_batch_rate": float(wrong_batch_count / len(pred_ids)) if len(pred_ids) > 0 else float("nan"),
            }
        )

    batch_eval_df = pd.DataFrame(batch_rows)
    target_beat_df = pd.DataFrame(target_beat_rows)
    beat_eval_df = pd.DataFrame(beat_rows)
    tracking_df, tracking_metrics = compute_signal_tracking_stability(target_beat_df)
    tracking_by_target = (
        tracking_df.set_index("true_sigidx").to_dict(orient="index")
        if len(tracking_df) > 0
        else {}
    )
    matched_batch_count_by_target = (
        batch_eval_df["matched_true_sigidx"].value_counts().to_dict()
        if len(batch_eval_df) > 0 and "matched_true_sigidx" in batch_eval_df.columns
        else {}
    )

    if len(target_beat_df) > 0:
        target_summary_rows = []
        for true_id, sub in target_beat_df.groupby("true_sigidx", sort=True):
            recognition_series = sub["recognition_success"] if recognition_available else pd.Series(dtype=bool)
            recognition_success_rows = sub.loc[sub["sort_success"], "recognition_success"] if recognition_available else pd.Series(dtype=bool)
            target_beats = int(len(sub))
            successful_beats = int(sub["sort_success"].sum())
            matched_pred_batches = int(matched_batch_count_by_target.get(int(true_id), 0))
            false_positive_batches = max(matched_pred_batches - successful_beats, 0)
            false_negative_beats = max(target_beats - successful_beats, 0)
            recall = _safe_ratio(successful_beats, target_beats)
            precision = _safe_ratio(successful_beats, matched_pred_batches)
            iou = _safe_ratio(successful_beats, successful_beats + false_positive_batches + false_negative_beats)
            row = {
                "true_sigidx": int(true_id),
                "target_label": majority_value(sub["target_label"].to_numpy(dtype=np.int64)),
                "target_beats_T": target_beats,
                "successful_beats": successful_beats,
                "sort_acc": recall,
                "extra_batch_rate": float(sub["extra_batches"].mean()),
                "extra_batches_total": int(sub["extra_batches"].sum()),
                "matched_pred_batches_P": matched_pred_batches,
                "tp_target_beats": successful_beats,
                "fp_extra_or_wrong_batches": false_positive_batches,
                "fn_missed_target_beats": false_negative_beats,
                "recall": recall,
                "precision": precision,
                "iou": iou,
                "recognition_acc": float(recognition_series.mean()) if len(recognition_series) > 0 else float("nan"),
                "recognition_acc_on_success": float(recognition_success_rows.mean())
                if len(recognition_success_rows) > 0 else float("nan"),
            }
            row.update(tracking_by_target.get(int(true_id), {}))
            target_summary_rows.append(row)
        target_df = pd.DataFrame(target_summary_rows)
    else:
        target_df = pd.DataFrame()

    sample_sort_acc = float(target_df["sort_acc"].mean()) if len(target_df) > 0 else float("nan")
    sample_extra_batch_rate = float(target_df["extra_batch_rate"].mean()) if len(target_df) > 0 else float("nan")
    sample_wrong_batch_rate = float(beat_eval_df["wrong_batch_rate"].mean()) if len(beat_eval_df) > 0 else float("nan")
    sample_mr = float(target_df["recall"].mean()) if len(target_df) > 0 else float("nan")
    sample_mp = float(target_df["precision"].mean()) if len(target_df) > 0 else float("nan")
    sample_miou = float(target_df["iou"].mean()) if len(target_df) > 0 else float("nan")
    recognition_values_all = (
        target_beat_df["recognition_success"]
        if recognition_available and len(target_beat_df) > 0
        else pd.Series(dtype=bool)
    )
    recognition_values = (
        target_beat_df.loc[target_beat_df["sort_success"], "recognition_success"]
        if recognition_available and len(target_beat_df) > 0
        else pd.Series(dtype=bool)
    )
    sample_recognition_acc = (
        float(target_df["recognition_acc"].mean())
        if recognition_available and len(target_df) > 0
        else float("nan")
    )

    metrics = {
        "sample_sort_acc": sample_sort_acc,
        "sample_extra_batch_rate": sample_extra_batch_rate,
        "sample_wrong_batch_rate": sample_wrong_batch_rate,
        "sample_mr": sample_mr,
        "sample_mp": sample_mp,
        "sample_miou": sample_miou,
        "MR": sample_mr,
        "MP": sample_mp,
        "MIOU": sample_miou,
        "sample_recognition_acc": sample_recognition_acc,
        "recognition_acc": sample_recognition_acc,
        "recognition_acc_on_success": float(recognition_values.mean()) if len(recognition_values) > 0 else float("nan"),
        "recognition_correct_target_beats": int(pd.Series(recognition_values_all).fillna(False).sum()) if len(recognition_values_all) > 0 else 0,
        "num_true_targets": int(len(target_df)),
        "num_pred_batches": int(len([v for v in np.unique(pred_sigidx) if int(v) > 0])),
        "num_beats": int(len(beat_eval_df)),
        "successful_target_beats": int(target_beat_df["sort_success"].sum()) if len(target_beat_df) > 0 else 0,
        "total_target_beats": int(len(target_beat_df)),
        "extra_batch_total": int(target_beat_df["extra_batches"].sum()) if len(target_beat_df) > 0 else 0,
        "wrong_batches": int(beat_eval_df["wrong_batches"].sum()) if len(beat_eval_df) > 0 else 0,
        "total_pred_batch_beats": int(beat_eval_df["num_pred_batches"].sum()) if len(beat_eval_df) > 0 else 0,
        "purity_threshold": float(purity_threshold),
        "min_target_fraction": float(min_target_fraction),
        "mix_fail_min_pulses": int(mix_fail_min_pulses),
        "chunk_seconds": float(chunk_seconds),
    }
    metrics.update(tracking_metrics)

    return batch_eval_df, target_df, target_beat_df, beat_eval_df, metrics

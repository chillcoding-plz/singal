#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep template matching parameters and diagnose known/unknown behavior.

This script does not change the recognizer.  It reuses the batch feature
extraction and template matching functions from template_match_recognition.py,
then evaluates several threshold settings on the same test batches.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

import template_match_recognition as rec


def parse_float_list(text: str) -> List[float]:
    values = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not values:
        raise ValueError(f"empty float list: {text!r}")
    return values


def parse_modes(text: str) -> List[str]:
    modes = [v.strip() for v in str(text).split(",") if v.strip()]
    valid = {"nearest", "nearest_with_rescue", "class_min_ratio"}
    bad = [v for v in modes if v not in valid]
    if bad:
        raise ValueError(f"unsupported matching mode(s): {bad}; valid={sorted(valid)}")
    return modes


def uniform_label_scales(templates: pd.DataFrame, scale: float) -> Dict[int, float]:
    labels = sorted(int(v) for v in templates["label"].astype(int).unique().tolist())
    return {label: float(scale) for label in labels}


def metric_value(metrics: Dict[str, object], name: str) -> float:
    value = metrics.get(name, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def per_class_reject_summary(metrics: Dict[str, object]) -> str:
    per_class = metrics.get("per_class_known", {})
    if not isinstance(per_class, dict):
        return ""
    parts = []
    for label, row in sorted(per_class.items(), key=lambda item: int(item[0])):
        if not isinstance(row, dict):
            continue
        reject = float(row.get("reject_as_unknown_rate", float("nan")))
        nearest_hit = float(row.get("nearest_label_hit_rate", float("nan")))
        acc = float(row.get("acc", float("nan")))
        parts.append(f"{label}:acc={acc:.4f},reject99={reject:.4f},nearest_hit={nearest_hit:.4f}")
    return "; ".join(parts)


def command_for_row(row: Dict[str, object]) -> str:
    return (
        "python template_match_recognition.py "
        f"--threshold_scale {row['threshold_scale']} "
        f"--label_threshold_scales \"{row['label_threshold_scales']}\" "
        f"--class_threshold_floor_scale {row['class_threshold_floor_scale']} "
        f"--matching_mode {row['matching_mode']}"
    )


def build_case_cache(sample: str, base_args: argparse.Namespace) -> Dict[str, object]:
    pdw_file, sort_file, truth_file, _output_dir = rec.resolve_input_paths(base_args, sample)
    rec.require_exists(pdw_file, "PDW file")
    rec.require_exists(sort_file, "sort file")
    if truth_file is None or not truth_file.exists():
        raise FileNotFoundError(f"truth file is required for diagnosis: {truth_file}")

    pdw = rec.read_pdw(pdw_file, int(base_args.max_pulses))
    pred_sigidx = rec.read_sort_sigidx(sort_file, int(base_args.max_pulses))
    truth = rec.read_truth(truth_file, int(base_args.max_pulses))
    if len(pdw) != len(pred_sigidx):
        raise ValueError(f"PDW/sort row mismatch: {len(pdw)} vs {len(pred_sigidx)}")
    if len(truth) != len(pred_sigidx):
        raise ValueError(f"truth/sort row mismatch: {len(truth)} vs {len(pred_sigidx)}")

    library_path = Path(base_args.template_library)
    if not library_path.is_absolute():
        library_path = rec.PROJECT_ROOT / library_path
    templates, metadata = rec.load_template_library(library_path)
    batches = rec.build_recognition_batches(
        pdw,
        pred_sigidx,
        min_batch_pulses=int(base_args.min_batch_pulses),
        gap_multiplier=float(base_args.pri_gap_multiplier),
        gap_quantile=float(base_args.pri_gap_quantile),
    )
    return {
        "sample": sample,
        "pdw": pdw,
        "pred_sigidx": pred_sigidx,
        "truth": truth,
        "templates": templates,
        "metadata": metadata,
        "batches": batches,
    }


def evaluate_case(
    case: Dict[str, object],
    base_args: argparse.Namespace,
    threshold_scale: float,
    label_scale: float,
    class_floor_scale: float,
    matching_mode: str,
) -> Dict[str, object]:
    templates = case["templates"]
    assert isinstance(templates, pd.DataFrame)
    label_scales = uniform_label_scales(templates, label_scale)
    label_scales_text = ",".join(f"{label}:{scale:g}" for label, scale in label_scales.items())
    pred_batches = rec.match_batches(
        case["batches"],
        templates,
        case["metadata"],
        threshold_scale=float(threshold_scale),
        label_threshold_scales=label_scales,
        class_threshold_floor_scale=float(class_floor_scale),
        min_margin=float(base_args.min_margin),
        matching_mode=str(matching_mode),
        class_ratio_margin=float(base_args.class_ratio_margin),
        enable_label2_rescue=bool(base_args.enable_label2_rescue),
        label2_rescue_ratio=float(base_args.label2_rescue_ratio),
        label2_feature_padding=float(base_args.label2_feature_padding),
        secondary_reject_ratio_caps=rec.parse_label_float_map(base_args.secondary_reject_ratio_caps),
        enable_topk_label_rescue=bool(base_args.enable_topk_label_rescue),
        topk_rescue_label=int(base_args.topk_rescue_label),
        topk_size=int(base_args.topk_size),
        topk_min_votes=int(base_args.topk_min_votes),
        topk_max_ratio=float(base_args.topk_max_ratio),
        topk_feature_padding=float(base_args.topk_feature_padding),
        enable_class_ratio_label_rescue=bool(base_args.enable_class_ratio_label_rescue),
        class_ratio_rescue_label=int(base_args.class_ratio_rescue_label),
        class_ratio_rescue_max_ratio=float(base_args.class_ratio_rescue_max_ratio),
        class_ratio_rescue_max_delta=float(base_args.class_ratio_rescue_max_delta),
        class_ratio_rescue_feature_padding=float(base_args.class_ratio_rescue_feature_padding),
    )
    pred_labels = rec.labels_from_batches(case["pred_sigidx"], pred_batches)
    pred_batches = rec.add_truth_to_batches(pred_batches, case["truth"], case["pred_sigidx"])
    metrics = rec.compute_recognition_metrics(pred_batches, pred_labels, case["truth"])

    row: Dict[str, object] = {
        "sample": str(case["sample"]),
        "threshold_scale": float(threshold_scale),
        "label_scale": float(label_scale),
        "label_threshold_scales": label_scales_text,
        "class_threshold_floor_scale": float(class_floor_scale),
        "matching_mode": str(matching_mode),
        "batch_acc": metric_value(metrics, "batch_acc"),
        "known_batch_acc": metric_value(metrics, "known_batch_acc"),
        "unknown_reject_rate": metric_value(metrics, "unknown_reject_rate"),
        "false_accept_rate": metric_value(metrics, "false_accept_rate"),
        "known_pulse_acc": metric_value(metrics, "known_pulse_acc"),
        "unknown_pulse_reject_rate": metric_value(metrics, "unknown_pulse_reject_rate"),
        "num_known_batches": int(metrics.get("num_known_batches", 0)),
        "num_unknown_batches_truth": int(metrics.get("num_unknown_batches_truth", 0)),
        "per_class_known": per_class_reject_summary(metrics),
    }
    row["recommended_command"] = command_for_row(row)
    return row


def sort_rows(rows: Iterable[Dict[str, object]], min_unknown_reject: float) -> List[Dict[str, object]]:
    def key(row: Dict[str, object]) -> tuple:
        unknown = float(row.get("unknown_reject_rate", float("nan")))
        known = float(row.get("known_batch_acc", float("nan")))
        batch = float(row.get("batch_acc", float("nan")))
        ok_unknown = 1 if math.isfinite(unknown) and unknown >= float(min_unknown_reject) else 0
        return (ok_unknown, known if math.isfinite(known) else -1.0, unknown if math.isfinite(unknown) else -1.0, batch if math.isfinite(batch) else -1.0)

    return sorted(rows, key=key, reverse=True)


def run_sweep(args: argparse.Namespace) -> List[Dict[str, object]]:
    base_args = rec.build_parser().parse_args([])
    for name, value in vars(args).items():
        if hasattr(base_args, name) and value is not None:
            setattr(base_args, name, value)

    samples = ["sample1", "sample2"] if args.sample == "all" else [args.sample]
    cases = [build_case_cache(sample, base_args) for sample in samples]
    rows = []
    combos = itertools.product(
        parse_float_list(args.threshold_scales),
        parse_float_list(args.label_scales),
        parse_float_list(args.class_floor_scales),
        parse_modes(args.matching_modes),
    )
    for threshold_scale, label_scale, floor_scale, mode in combos:
        for case in cases:
            rows.append(evaluate_case(case, base_args, threshold_scale, label_scale, floor_scale, mode))
    return sort_rows(rows, float(args.min_unknown_reject))


def save_outputs(rows: List[Dict[str, object]], output_dir: Path, top_k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "template_match_parameter_sweep.csv"
    json_path = output_dir / "template_match_parameter_sweep_top.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(rows[: int(top_k)], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] rows: {len(rows)}")
    print(f"[done] csv: {csv_path}")
    print(f"[done] top json: {json_path}")
    if rows:
        best = rows[0]
        print(
            "[best] "
            f"sample={best['sample']} known_batch_acc={best['known_batch_acc']:.4f} "
            f"unknown_reject_rate={best['unknown_reject_rate']:.4f} "
            f"batch_acc={best['batch_acc']:.4f}"
        )
        print(f"[best command] {best['recommended_command']}")


def run_self_test() -> None:
    templates = pd.DataFrame(
        {
            "template_id": ["class1_a", "class2_a"],
            "label": [1, 2],
            "f1": [0.0, 10.0],
            "local_distance_threshold": [2.0, 2.0],
            "ambiguous_template": [False, False],
        }
    )
    metadata = {
        "features": ["f1"],
        "feature_centers": {"f1": 5.0},
        "feature_scales": {"f1": 5.0},
        "feature_weights": {"f1": 1.0},
        "class_thresholds": {"1": 2.0, "2": 2.0},
        "class_summary": {},
    }
    batches = pd.DataFrame({"pred_sigidx": [1, 2, 3], "f1": [0.2, 9.8, 50.0]})
    pred_sigidx = np.array([1, 2, 3], dtype=np.int64)
    truth = pd.DataFrame({"SigIdx": [1, 2, 3], "LABEL": [1, 2, rec.UNKNOWN_LABEL]})
    case = {
        "sample": "self_test",
        "pred_sigidx": pred_sigidx,
        "truth": truth,
        "templates": templates,
        "metadata": metadata,
        "batches": batches,
    }
    base_args = rec.build_parser().parse_args([])
    row = evaluate_case(case, base_args, 1.0, 1.0, 0.0, "nearest")
    assert row["known_batch_acc"] == 1.0, row
    assert row["unknown_reject_rate"] == 1.0, row
    save_outputs([row], Path.cwd() / "outputs_template_match_parameter_diagnosis_selftest", top_k=1)
    print("[self_test] ok")


def build_parser() -> argparse.ArgumentParser:
    base = rec.build_parser()
    parser = copy.deepcopy(base)
    parser.description = "Diagnose and sweep template matching parameters."
    parser.add_argument("--threshold_scales", type=str, default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--label_scales", type=str, default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--class_floor_scales", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--matching_modes", type=str, default="nearest,class_min_ratio")
    parser.add_argument("--min_unknown_reject", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--self_test", action="store_true")
    parser.set_defaults(output_dir=Path("outputs_template_match_parameter_diagnosis"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    rows = run_sweep(args)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    save_outputs(rows, output_dir, int(args.top_k))


if __name__ == "__main__":
    main()

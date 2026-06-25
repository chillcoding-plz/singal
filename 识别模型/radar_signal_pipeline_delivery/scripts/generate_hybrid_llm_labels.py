"""
Generate hybrid LLM-style pseudo labels.

Policy:
  - Primary evidence: radar-individualized PDW files with PRED_LABEL.
  - Auxiliary evidence: original pre-sorting Train_Data/Class_*.txt profiles.
  - The original data only calibrates confidence and flags distribution mismatch;
    it must not overwrite the individualized radar timeline.

Output JSONL is compatible with app.run_pipeline --llm-labels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_pipeline.input_adapter import (  # noqa: E402
    WINDOW_DURATION,
    _check_window_quality,
    _compute_window_id,
    _validate_doa,
    load_and_window,
)
from radar_pipeline.quality import check_window  # noqa: E402
from radar_pipeline.schemas import WindowFeatures, WindowRecord  # noqa: E402
from radar_pipeline.window_features import compute_features_batch  # noqa: E402


FEATURE_KEYS = [
    "pri_median",
    "pw_median",
    "pulse_density",
    "rf_iqr",
    "doa_unwrapped_range",
    "pa_cv",
    "pa_periodicity",
]

AUX_MIN_SCALE = {
    "pri_median": 5.0,
    "pw_median": 0.3,
    "pulse_density": 0.45,
    "rf_iqr": 0.35,
    "doa_unwrapped_range": 8.0,
    "pa_cv": 0.04,
    "pa_periodicity": 0.05,
}
AUX_LOG_FEATURES = {"pulse_density", "rf_iqr"}


def finite_median(values: list[float], default: float = 0.0) -> float:
    arr = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.median(arr)) if arr else default


def load_raw_class_windows(raw_root: Path) -> dict[str, list[WindowRecord]]:
    result: dict[str, list[WindowRecord]] = {}
    for path in sorted(raw_root.glob("Class_*.txt")):
        df = pd.read_csv(path, sep=r"\s+")
        if df.empty:
            continue
        df = df.sort_values("TOA(s)")
        class_name = path.stem
        toa = df["TOA(s)"].to_numpy(dtype=np.float64)
        signal = df[["Param1", "Param2", "Param4", "Param5", "Param6"]].to_numpy(
            dtype=np.float32
        )
        base_time = float(toa.min())
        window_ids = np.array([_compute_window_id(t, base_time) for t in toa])
        min_wid = int(window_ids.min())
        max_wid = int(window_ids.max())

        windows: list[WindowRecord] = []
        doa_valid = _validate_doa(signal[:, 3])
        for wid in range(min_wid, max_wid + 1):
            mask = window_ids == wid
            w_start = wid * WINDOW_DURATION
            w_end = w_start + WINDOW_DURATION
            if not mask.any():
                windows.append(
                    WindowRecord(
                        radar_id=class_name,
                        sample="raw",
                        window_id=wid,
                        start_time=w_start,
                        end_time=w_end,
                        pdw=None,
                        n_pulses=0,
                        is_empty=True,
                        quality_flags=["empty"],
                    )
                )
                continue

            pdw = np.column_stack([toa[mask], signal[mask]])
            qflags = _check_window_quality(pdw, int(mask.sum()))
            if not doa_valid:
                qflags.append("doa_invalid")
            windows.append(
                WindowRecord(
                    radar_id=class_name,
                    sample="raw",
                    window_id=wid,
                    start_time=w_start,
                    end_time=w_end,
                    pdw=pdw,
                    n_pulses=int(mask.sum()),
                    is_empty=False,
                    quality_flags=qflags,
                )
            )
        result[class_name] = windows
    return result


def summarize_one_second(features: list[WindowFeatures]) -> dict[str, Any]:
    valid = [f for f in features if not f.is_empty and f.n_pulses >= 5]
    total_pulses = sum(f.n_pulses for f in features)
    if not valid:
        return {
            "valid": False,
            "n_pulses": total_pulses,
            "valid_ratio": 0.0,
        }

    return {
        "valid": True,
        "n_pulses": total_pulses,
        "valid_ratio": len(valid) / max(len(features), 1),
        "pri_median": finite_median([f.pri_median for f in valid if f.pri_median > 0]),
        "pw_median": finite_median([f.pw_median for f in valid]),
        "pulse_density": finite_median([f.pulse_density for f in valid]),
        "rf_iqr": finite_median([f.rf_iqr for f in valid]),
        "doa_unwrapped_range": finite_median(
            [f.doa_unwrapped_range for f in valid if f.doa_valid]
        ),
        "pa_cv": finite_median([f.pa_cv for f in valid]),
        "pa_periodicity": finite_median([f.pa_periodicity for f in valid]),
    }


def one_second_summaries(windows: list[WindowRecord]) -> list[dict[str, Any]]:
    for w in windows:
        w.quality_flags = check_window(w)
    features = compute_features_batch(windows)
    rows = []
    for start in range(0, len(features), 5):
        chunk = features[start : start + 5]
        if not chunk:
            continue
        row = summarize_one_second(chunk)
        row["window_index"] = start // 5
        row["start_time"] = row["window_index"] * 1.0
        row["end_time"] = row["start_time"] + 1.0
        rows.append(row)
    return rows


def build_raw_profiles(raw_root: Path) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    raw_windows = load_raw_class_windows(raw_root)
    for class_name, windows in raw_windows.items():
        rows = [r for r in one_second_summaries(windows) if r.get("valid")]
        profile: dict[str, Any] = {
            "n_segments": len(rows),
            "n_pulses": sum(int(r.get("n_pulses", 0)) for r in rows),
        }
        for key in FEATURE_KEYS:
            values = [float(r.get(key, 0.0)) for r in rows if np.isfinite(r.get(key, 0.0))]
            profile[key] = finite_median(values)
            q25, q75 = np.percentile(values, [25, 75]) if values else (0.0, 1.0)
            profile[f"{key}_scale"] = max(float(q75 - q25), abs(profile[key]) * 0.1, 1e-6)
        profiles[class_name] = profile
    return profiles


def auxiliary_distance(row: dict[str, Any], profiles: dict[str, dict[str, Any]]) -> tuple[str, float]:
    if not row.get("valid") or not profiles:
        return "", float("inf")

    best_name = ""
    best_dist = float("inf")
    for name, profile in profiles.items():
        parts = []
        for key in FEATURE_KEYS:
            value = float(row.get(key, 0.0))
            center = float(profile.get(key, 0.0))
            scale = float(profile.get(f"{key}_scale", 1.0))
            if not np.isfinite(value):
                continue
            if key in AUX_LOG_FEATURES:
                raw_center = max(center, 0.0)
                raw_scale = max(scale, 0.0)
                value = math.log1p(max(value, 0.0))
                center = math.log1p(raw_center)
                upper = math.log1p(raw_center + raw_scale)
                lower = math.log1p(max(raw_center - raw_scale, 0.0))
                scale = max(abs(upper - center), abs(center - lower), AUX_MIN_SCALE[key])
            else:
                scale = max(scale, AUX_MIN_SCALE[key])
            parts.append(((value - center) / scale) ** 2)
        dist = math.sqrt(sum(parts) / max(len(parts), 1))
        if dist < best_dist:
            best_name = name
            best_dist = dist
    return best_name, best_dist


def infer_primary_label(row: dict[str, Any]) -> tuple[str, str, float, str]:
    if not row.get("valid") or row.get("n_pulses", 0) < 300:
        return "未知", "未知", 0.55, "有效脉冲不足"

    pri = float(row.get("pri_median", 0.0))
    pw = float(row.get("pw_median", 0.0))
    density = float(row.get("pulse_density", 0.0))
    rf_iqr = float(row.get("rf_iqr", 0.0))
    doa = float(row.get("doa_unwrapped_range", 0.0))
    pa_cv = float(row.get("pa_cv", 0.0))
    pa_per = float(row.get("pa_periodicity", 0.0))
    pulses = int(row.get("n_pulses", 0))

    if pri < 6.0 and pw < 1.6 and density > 100000 and rf_iqr > 50 and pa_cv < 0.12:
        return "制导", "火控", 0.88, "个体信号强证据: HPRF+窄脉宽+超高密度+稳定照射"

    if pw >= 6.0 and pa_cv < 0.08 and density > 8000:
        return "跟踪", "火控", 0.82, "个体信号强证据: 宽脉宽+稳定照射"

    if pw >= 6.0 and density > 5000 and pri > 20:
        return "搜索", "待定", 0.64, "个体信号: 宽脉冲但幅度稳定性不足, 不宜判为火控跟踪"

    if 20 <= pri <= 30 and pw < 1.5 and rf_iqr > 150 and pa_cv < 0.05 and density > 30000:
        return "搜索", "待定", 0.70, "个体信号: 中重频窄脉冲高捷变搜索"

    if 30 <= pri <= 60 and pw < 1.5 and rf_iqr > 50 and pa_cv < 0.05 and 15000 <= density <= 50000:
        return "未知", "未知", 0.68, "个体信号: 高捷变稳定窄脉冲样式未纳入已知属性"

    if rf_iqr > 1000 and density > 30000 and 12 <= doa <= 25:
        return "搜索", "对空搜索", 0.82, "个体信号: 超大RF捷变+中等DOA覆盖"

    if rf_iqr > 100 and density > 50000 and 12 <= doa <= 25:
        return "搜索", "对空搜索", 0.78, "个体信号: 大RF捷变+中等DOA覆盖"

    if rf_iqr < 10 and density > 50000 and 1.5 <= pw <= 5.0 and 8 <= doa <= 18:
        return "搜索", "对海搜索", 0.76, "个体信号: 固定RF+高密度+中脉宽"

    if rf_iqr < 10 and density < 3000 and pri > 20 and 1.0 <= pw <= 2.8 and pulses > 5000 and doa <= 12:
        return "搜索", "对海搜索", 0.70, "个体信号: 固定RF+低密度周期扫描+低DOA覆盖, 符合对海搜索段"

    if density < 3000 and pri > 20 and pa_per > 0.12:
        if pulses < 500:
            return "未知", "未知", 0.58, "个体信号: 低脉冲低密度扫描证据不足"
        return "搜索", "待定", 0.62, "个体信号: 低密度扫描但属性证据不足"

    if density > 8000:
        return "搜索", "待定", 0.60, "个体信号: 搜索倾向但属性证据不足"

    return "未知", "未知", 0.55, "个体信号: 已知模式证据不足"


def calibrate_with_auxiliary(
    mode: str,
    attr: str,
    confidence: float,
    aux_class: str,
    aux_dist: float,
) -> tuple[float, str]:
    note = f"raw_aux={aux_class or 'none'}, normalized_dist={aux_dist:.2f}, role=confidence_only"
    if not np.isfinite(aux_dist):
        return max(confidence - 0.02, 0.0), note + ", no_aux_profile"
    if aux_dist <= 1.8:
        return min(confidence + 0.03, 0.95), note + ", raw_distribution_support"
    if aux_dist <= 3.5:
        return min(confidence + 0.01, 0.95), note + ", weak_raw_support"
    if aux_dist >= 7.5:
        return max(confidence - 0.03, 0.0), note + ", raw_distribution_mismatch_warn"
    if aux_dist >= 5.0:
        return max(confidence - 0.02, 0.0), note + ", weak_raw_mismatch_warn"
    return confidence, note + ", raw_aux_neutral"


def smooth_isolated_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only smooth low-confidence singletons between identical neighbors."""
    if len(rows) < 3:
        return rows
    out = [dict(r) for r in rows]
    for i in range(1, len(rows) - 1):
        prev_state = (out[i - 1]["llm_work_mode"], out[i - 1]["llm_func_attr"])
        cur_state = (out[i]["llm_work_mode"], out[i]["llm_func_attr"])
        next_state = (out[i + 1]["llm_work_mode"], out[i + 1]["llm_func_attr"])
        if (
            prev_state == next_state
            and cur_state != prev_state
            and float(out[i].get("llm_confidence", 0.0)) < 0.62
        ):
            out[i]["llm_work_mode"], out[i]["llm_func_attr"] = prev_state
            out[i]["llm_confidence"] = min(float(out[i]["llm_confidence"]) + 0.08, 0.75)
            out[i]["llm_reason"] += "; temporal_smoothing=isolated_low_confidence"
    return out


def generate_labels(
    individual_inputs: list[str],
    raw_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles = build_raw_profiles(raw_root)
    all_radars = load_and_window(individual_inputs, min_pulses=100)

    labels: list[dict[str, Any]] = []
    per_radar: dict[str, list[dict[str, Any]]] = {}
    for radar_key, windows in sorted(all_radars.items()):
        sample, radar_id = radar_key.split("/")
        rows = one_second_summaries(windows)
        radar_labels = []
        for row in rows:
            mode, attr, conf, reason = infer_primary_label(row)
            aux_class, aux_dist = auxiliary_distance(row, profiles)
            conf, aux_note = calibrate_with_auxiliary(mode, attr, conf, aux_class, aux_dist)
            record = {
                "record_id": f"{sample}/{radar_id}/w{int(row['window_index']):03d}",
                "llm_work_mode": mode,
                "llm_func_attr": attr,
                "llm_confidence": round(conf, 3),
                "llm_reason": f"{reason}; {aux_note}; primary_weight=0.85; raw_aux_weight=0.15",
                "primary_source": "individualized_radar_pdw",
                "aux_source": "raw_train_class_profile",
                "evidence_policy": "individualized_signal_decides_label_raw_signal_confidence_only",
                "auxiliary_role": "confidence_only_no_label_override",
                "aux_nearest_class": aux_class,
                "aux_distance": round(aux_dist, 3) if np.isfinite(aux_dist) else None,
                "n_pulses": int(row.get("n_pulses", 0)),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
            }
            radar_labels.append(record)
        per_radar[radar_key] = smooth_isolated_labels(radar_labels)
        labels.extend(per_radar[radar_key])

    metadata = {
        "policy": "individualized_signal_primary_raw_signal_auxiliary",
        "primary_weight": 0.85,
        "raw_auxiliary_weight": 0.15,
        "raw_auxiliary_role": "confidence_only_no_label_override",
        "raw_profiles": profiles,
        "n_labels": len(labels),
        "n_radars": len(all_radars),
    }
    return labels, metadata


def write_outputs(labels: list[dict[str, Any]], metadata: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "hybrid_llm_labels.jsonl"
    with label_path.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "hybrid_label_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts: dict[str, dict[str, int]] = {}
    for row in labels:
        key = "/".join(row["record_id"].split("/")[:2])
        state = f"{row['llm_work_mode']}/{row['llm_func_attr']}"
        counts.setdefault(key, {})
        counts[key][state] = counts[key].get(state, 0) + 1

    lines = [
        "# Hybrid LLM Pseudo Labels",
        "",
        "This label set uses individualized radar PDW as the primary evidence and raw Train_Data/Class profiles only as auxiliary confidence calibration.",
        "Raw profiles never overwrite the individualized radar timeline labels.",
        "",
        f"- labels: {len(labels)}",
        f"- primary_weight: {metadata['primary_weight']}",
        f"- raw_auxiliary_weight: {metadata['raw_auxiliary_weight']}",
        f"- raw_auxiliary_role: {metadata['raw_auxiliary_role']}",
        "",
        "| radar | state distribution |",
        "|---|---|",
    ]
    for key in sorted(counts):
        dist = ", ".join(f"{state}: {count}" for state, count in sorted(counts[key].items()))
        lines.append(f"| {key} | {dist} |")
    (output_dir / "hybrid_label_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--individual-input",
        nargs="+",
        default=[
            "F:/signals/sample1_template_match_pdw_with_pred_label.txt",
            "F:/signals/sample2_template_match_pdw_with_pred_label.txt",
        ],
    )
    parser.add_argument(
        "--raw-root",
        default="F:/信号分选数据/KJ比赛/科目1/Data/data/Train_Data",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/llm_eval/hybrid_v1",
    )
    args = parser.parse_args()

    labels, metadata = generate_labels(args.individual_input, Path(args.raw_root))
    write_outputs(labels, metadata, Path(args.output_dir))
    print(json.dumps({"output_dir": args.output_dir, "n_labels": len(labels)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

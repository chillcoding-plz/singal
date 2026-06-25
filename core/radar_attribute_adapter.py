from __future__ import annotations

import json
import os
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RADAR_DELIVERY_ROOT = PROJECT_ROOT / "识别模型" / "radar_signal_pipeline_delivery"


@dataclass
class RadarAttributeDashboard:
    status: str
    run_dir: str
    input_file: str
    block_duration: float
    display_interval: float
    change_method: str
    radars: dict
    manifest: dict
    summary: dict
    display_frames: dict
    segments: pd.DataFrame
    report_path: str
    error: Optional[str] = None


@contextmanager
def _pipeline_import_path():
    root = str(RADAR_DELIVERY_ROOT)
    added = root not in sys.path
    if added:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def _toa_seconds(series: pd.Series) -> pd.Series:
    toa = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if len(toa) and float(toa.max()) > 1000.0:
        return toa / 1_000_000.0
    return toa


def _numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _coerce_label_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) > 0:
        return numeric.fillna(0).astype(int)
    extracted = series.astype(str).str.extract(r"(-?\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").fillna(0).astype(int)


def _track_labels(df: pd.DataFrame) -> pd.Series:
    label_column = next(
        (column for column in ("LABEL", "Predicted_Label", "PRED_LABEL", "Track_ID", "Display_Track_ID") if column in df.columns),
        None,
    )
    if label_column is None:
        raise ValueError("Radar attribute analysis requires LABEL, Predicted_Label, or PRED_LABEL.")
    labels = _coerce_label_series(df[label_column])
    labels = labels.where(labels > 0, 0)
    if int((labels > 0).sum()) == 0:
        raise ValueError(f"Radar attribute label column has no positive labels: {label_column}")
    return labels

def prepare_radar_pipeline_input(df: pd.DataFrame, directory: str | os.PathLike[str]) -> str:
    """Write current PDW data as the external pipeline's 200 ms beat zip package."""
    if df is None or df.empty:
        raise ValueError("输入数据为空，无法准备雷达管线输入文件")

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapted = pd.DataFrame(
        {
            "TOA(s)": _toa_seconds(df["TOA"] if "TOA" in df.columns else pd.Series(0, index=df.index)),
            "Param1": _numeric_column(df, "RF"),
            "Param2": _numeric_column(df, "PW"),
            "Param4": _numeric_column(df, "PA"),
            "Param5": _numeric_column(df, "DOA"),
            "Param6": _numeric_column(df, "Param6"),
            "PRED_LABEL": _track_labels(df),
        }
    )
    adapted = adapted[adapted["PRED_LABEL"] > 0].sort_values(["PRED_LABEL", "TOA(s)"])
    if adapted.empty:
        raise ValueError("预处理后无有效数据，请检查 PDW 表中是否包含有效的脉冲和 PRED_LABEL")
    path = output_dir / "sample1_template_match_pdw_with_pred_label.zip"
    beat_ids = (adapted["TOA(s)"] // 0.2).astype(int)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for beat_id in sorted(beat_ids.unique()):
            beat = adapted.loc[beat_ids == beat_id].sort_values(["TOA(s)", "PRED_LABEL"])
            if beat.empty:
                continue
            archive.writestr(
                f"beat_{int(beat_id):06d}.txt",
                beat.to_csv(sep=" ", index=False, float_format="%.9f"),
            )
    return str(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _read_segments(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "segment_tables" / "all_radars.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _summaries_to_radars(summaries: dict) -> dict:
    radars = {}
    for key, summary in (summaries or {}).items():
        diag = summary.get("diagnostics", {}) if isinstance(summary, dict) else {}
        conservation = summary.get("conservation", {}) if isinstance(summary, dict) else {}
        radars[key] = {
            "radar_id": summary.get("radar_id", ""),
            "sample": summary.get("sample", ""),
            "total_pulses": summary.get("total_pulses", 0),
            "total_windows": summary.get("total_windows", 0),
            "n_empty_windows": summary.get("n_empty_windows", 0),
            "n_valid_windows": summary.get("n_valid_windows", 0),
            "dominant_mode": summary.get("dominant_mode", "未知"),
            "mode_distribution": summary.get("mode_distribution", {}),
            "global_func_attr": summary.get("global_func_attr", "未知"),
            "global_func_attr_conf": summary.get("global_func_attr_conf", 0.0),
            "known_coverage": diag.get("known_coverage", 0.0),
            "unknown_ratio": diag.get("unknown_ratio", 0.0),
            "n_segments": summary.get("n_segments", 0),
            "n_blocks": summary.get("n_blocks", 0),
            "conservation": conservation,
            "quality": summary.get("quality", {}),
        }
    return radars


def _read_partial_segments(run_dir: Path) -> pd.DataFrame:
    rows = []
    timelines = run_dir / "timelines"
    if not timelines.exists():
        return pd.DataFrame()
    index = 1
    for block_path in sorted(timelines.glob("*/*/block_*.json")):
        try:
            block = _read_json(block_path)
        except Exception:
            continue
        sample = block.get("sample", block_path.parent.parent.name)
        radar_id = block.get("radar_id", block_path.parent.name)
        radar_key = f"{sample}/{radar_id}"
        attrs = block.get("attribute_timeline", []) or []
        modes = block.get("mode_timeline", []) or []
        for mode in modes:
            start = float(mode.get("start_time", block.get("time_start", 0.0)) or 0.0)
            end = float(mode.get("end_time", block.get("time_end", start + 0.2)) or start + 0.2)
            attr_label = "未知"
            attr_score = 0.0
            for attr in attrs:
                a0 = float(attr.get("start_time", start) or start)
                a1 = float(attr.get("end_time", end) or end)
                if a1 > start and a0 < end:
                    attr_label = str(attr.get("attr", "未知"))
                    attr_score = 0.7 if str(attr.get("decision", "")) == "known" else 0.45
                    break
            mode_score = float(mode.get("evidence_score", 0.0) or 0.0)
            rows.append({
                "index": index,
                "radar_key": radar_key,
                "time": f"{start:.3f}~{end:.3f}",
                "start_time": start,
                "end_time": end,
                "n_pulses": int(mode.get("n_pulses", block.get("n_pulses", 0)) or 0),
                "mode": str(mode.get("mode", "未知")),
                "attribute": attr_label,
                "accuracy": max(mode_score, attr_score),
                "mode_accuracy": mode_score,
                "attr_accuracy": attr_score,
                "joint_accuracy": min(mode_score, attr_score) if attr_score else mode_score,
                "source": "partial_block",
            })
            index += 1
    return pd.DataFrame(rows)


def _default_llm_labels_path() -> Optional[str]:
    path = RADAR_DELIVERY_ROOT / "artifacts" / "llm_eval" / "hybrid_v1" / "hybrid_llm_labels.jsonl"
    return str(path) if path.exists() else None


def run_radar_attribute_pipeline(
    df: pd.DataFrame,
    output_base_dir: str | os.PathLike[str],
    block_duration: float = 5.0,
    display_interval: float = 30.0,
    change_method: str = "auto",
    progress_callback: Optional[Callable[[int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    stream_callback: Optional[Callable[[RadarAttributeDashboard], None]] = None,
) -> RadarAttributeDashboard:
    if not RADAR_DELIVERY_ROOT.exists():
        raise FileNotFoundError(f"Radar attribute delivery root not found: {RADAR_DELIVERY_ROOT}")
    if should_cancel and should_cancel():
        raise RuntimeError("Radar attribute pipeline cancelled")
    if progress_callback:
        progress_callback(10, "准备工作模式/功能属性识别输入")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_dir = Path(output_base_dir)
    input_dir = base_dir / "radar_attribute_inputs" / timestamp
    runs_dir = base_dir / "radar_attribute_runs"
    input_file = prepare_radar_pipeline_input(df, input_dir)

    if should_cancel and should_cancel():
        raise RuntimeError("Radar attribute pipeline cancelled")
    if progress_callback:
        progress_callback(30, "运行工作模式/功能属性流水线")

    def emit_partial(payload: dict):
        if stream_callback is None:
            return
        run_dir = Path(payload.get("run_dir") or "")
        summaries = payload.get("summaries", {})
        stream_callback(RadarAttributeDashboard(
            status="running",
            run_dir=str(run_dir),
            input_file=input_file,
            block_duration=block_duration,
            display_interval=display_interval,
            change_method=change_method,
            radars=_summaries_to_radars(summaries),
            manifest={},
            summary={},
            display_frames={},
            segments=_read_partial_segments(run_dir),
            report_path=str(run_dir / "report.md"),
        ))

    with _pipeline_import_path():
        from app.api import RadarAPI

        api = RadarAPI(output_dir=str(runs_dir))
        result = api.run(
            input_files=[input_file],
            block_duration=block_duration,
            change_method=change_method,
            display_interval=display_interval,
            llm_labels_path=_default_llm_labels_path(),
            pre_segmented=False,
            partial_callback=emit_partial,
        )

    if result.get("status") != "ok":
        raise RuntimeError(result.get("error") or "Radar attribute pipeline failed")

    if progress_callback:
        progress_callback(82, "读取识别输出摘要")
    run_dir = Path(result.get("run_dir") or "")
    segments = _read_segments(run_dir)
    summary = _read_json(run_dir / "global_summary.json")
    display_frames = _read_json(run_dir / "display_frames" / "index.json")
    if progress_callback:
        progress_callback(96, "构建雷达属性仪表板")
    return RadarAttributeDashboard(
        status="ok",
        run_dir=str(run_dir),
        input_file=input_file,
        block_duration=block_duration,
        display_interval=display_interval,
        change_method=change_method,
        radars=result.get("radars", {}),
        manifest=result.get("manifest", {}),
        summary=summary,
        display_frames=display_frames,
        segments=segments,
        report_path=str(run_dir / "report.md"),
    )

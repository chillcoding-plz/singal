import subprocess
import sys
import time
import uuid
import re
import os
import queue
import threading
import importlib.util
import json
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = PROJECT_ROOT / ".model_runs"


SORT_MODEL_DIR = PROJECT_ROOT / "分选模型" / "HDBSCAN"
CYCLE_PERIOD_MODEL_FILE = PROJECT_ROOT / "分选模型" / "WU" / "cycle_period_sort.py"
CYCLE_PERIOD_BEAT_MODEL_FILE = PROJECT_ROOT / "分选模型" / "WU" / "main5_200ms_sort.py"
RECOGNITION_MODEL_DIR = PROJECT_ROOT / "识别模型" / "zeng"
ZENG_TEMPLATE_LIBRARY = RECOGNITION_MODEL_DIR / "outputs_expanded_template_library" / "template_library.json"


def _run_dir(prefix: str) -> Path:
    path = RUN_ROOT / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _truth_file_for_df(df: pd.DataFrame) -> Optional[Path]:
    for key in ("truth_path", "source_path"):
        value = df.attrs.get(key)
        if not value:
            continue
        path = Path(value)
        candidate = path if key == "truth_path" else path.parent / "Sorted_PDW.txt"
        if candidate.exists():
            return candidate.resolve()
    return None


def _require_modules(algorithm: str, modules: Dict[str, str], install_hint: str) -> None:
    missing = [package for module, package in modules.items() if find_spec(module) is None]
    if not missing:
        return
    names = ", ".join(missing)
    raise RuntimeError(
        f"{algorithm} 缺少依赖：{names}\n\n"
        f"请先在当前 VS Code/conda 环境中安装依赖：\n{install_hint}\n\n"
        f"当前 Python：{sys.executable}"
    )


def _series(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(np.full(len(df), default), index=df.index)


def _toa_seconds(df: pd.DataFrame) -> pd.Series:
    toa = _series(df, "TOA", 0.0).astype(float)
    # The bundled external-model samples use seconds in a column named TOA(s).
    # Some UI demo CSVs use microseconds. Treat only clearly large TOA values as
    # microseconds; otherwise preserve seconds so windowing matches direct runs.
    if len(toa) and float(toa.max()) > 1_000.0:
        return toa / 1_000_000.0
    return toa


def write_external_pdw(df: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "TOA(s)": _toa_seconds(df),
            "Param1": _series(df, "RF", 0.0),
            "Param2": _series(df, "PW", 0.0),
            "Param3": _series(df, "PRI", 0.0),
            "Param4": _series(df, "PA", 0.0),
            "Param5": _series(df, "DOA", 0.0),
            "Param6": np.zeros(len(df), dtype=int),
            "Param7": np.zeros(len(df), dtype=float),
        }
    )
    out.to_csv(path, sep=" ", index=False, float_format="%.9f")


def zeng_template_library_path() -> Path:
    return ZENG_TEMPLATE_LIBRARY


def zeng_template_library_exists() -> bool:
    return ZENG_TEMPLATE_LIBRARY.exists()


def write_external_sort(df: pd.DataFrame, path: Path) -> None:
    if "Track_ID" not in df.columns:
        raise ValueError("zeng 识别需要先完成分选，缺少 Track_ID 字段")
    out = pd.DataFrame(
        {
            "TOA(s)": _toa_seconds(df),
            "SigIdx": pd.to_numeric(df["Track_ID"], errors="coerce").fillna(0).astype(int),
        }
    )
    out.to_csv(path, sep=" ", index=False, float_format="%.9f")


def _add_display_track_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "Track_ID" not in df.columns:
        return df
    out = df.copy()
    tracks = pd.to_numeric(out["Track_ID"], errors="coerce").fillna(0).astype(int)
    mapping: Dict[int, int] = {}
    next_display_id = 1
    display_ids = []
    for track_id in tracks:
        if int(track_id) <= 0:
            display_ids.append(0)
            continue
        if int(track_id) not in mapping:
            mapping[int(track_id)] = next_display_id
            next_display_id += 1
        display_ids.append(mapping[int(track_id)])
    out["Display_Track_ID"] = display_ids
    return out


def _positive_cycle_track_ids(cycle_labels, source_sigidx) -> np.ndarray:
    labels = pd.to_numeric(pd.Series(cycle_labels), errors="coerce")
    source = pd.to_numeric(pd.Series(source_sigidx), errors="coerce").fillna(0).astype(int)
    valid = (source > 0) & labels.notna() & (labels >= 0)
    mapped = np.zeros(len(labels), dtype=int)
    mapped[valid.to_numpy()] = labels[valid].astype(int).to_numpy() + 1
    return mapped


def _raw_cycle_pred_ids(cycle_labels) -> np.ndarray:
    labels = pd.to_numeric(pd.Series(cycle_labels), errors="coerce").fillna(0).astype(int)
    return labels.to_numpy()


def _interactive_window_seconds(df: pd.DataFrame, target_pulses_per_window: int = 8000) -> float:
    if len(df) <= target_pulses_per_window:
        return 0.1
    toa = _toa_seconds(df).astype(float)
    span = float(toa.max() - toa.min()) if len(toa) else 0.0
    if not np.isfinite(span) or span <= 0:
        return 0.1
    windows = max(1, int(np.ceil(len(df) / max(target_pulses_per_window, 1))))
    return max(span / windows, 1e-6)


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        except Exception:
            pass
    process.kill()


ProgressCallback = Optional[Callable[[int, str], None]]
CancelCallback = Optional[Callable[[], bool]]
StreamCallback = Optional[Callable[[pd.DataFrame], None]]
LineCallback = Optional[Callable[[str], None]]


def _emit(progress_callback: ProgressCallback, value: int, text: str) -> None:
    if progress_callback is not None:
        progress_callback(int(max(0, min(100, value))), text)


def _check_cancelled(should_cancel: CancelCallback) -> None:
    if should_cancel is not None and should_cancel():
        raise RuntimeError("任务已取消")


def _hdbscan_progress_from_line(line: str) -> Optional[Tuple[int, str]]:
    if line.startswith("[beat_result]"):
        return None
    if line.startswith("[beat]"):
        return 35, line.strip()
    if line.startswith("[initial_sort]"):
        return 10, "初始HDBSCAN分选"
    if line.startswith("[tracklet_graph]"):
        return 84, "轨迹图修正"
    if line.startswith("[best_fusion]"):
        return 90, "Best融合"
    if line.startswith("[done]"):
        return 98, "结果写入"

    match = re.search(r"\[pa-features\]\s+([\d.]+)%\s*(.*)", line)
    if match:
        pct = max(0.0, min(100.0, float(match.group(1)))) / 100.0
        detail = match.group(2).strip()
        suffix = f"：{detail}" if detail else ""
        return int(15 + pct * 10), f"HDBSCAN物理特征构建{suffix}"

    match = re.search(r"\[pa-hdbscan\]\s+windows\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", line)
    if match:
        pos = int(match.group(1))
        total = max(int(match.group(2)), 1)
        pct = pos / total
        return int(25 + pct * 55), f"HDBSCAN窗口聚类：{pos}/{total}"

    if line.startswith("Features:"):
        return 25, "HDBSCAN物理特征构建完成"

    match = re.search(r"\[pa-hdbscan\]\s+windows\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", line)
    if match:
        pos = int(match.group(1))
        total = max(int(match.group(2)), 1)
        pct = pos / total
        return int(20 + pct * 60), f"HDBSCAN窗口聚类：{pos}/{total}"
    if line.startswith("Input PDW:"):
        return 8, "HDBSCAN读取输入数据"
    if line.startswith("Pulses:"):
        return 12, line.strip()
    if line.startswith("HDBSCAN backend:"):
        return 14, line.strip()
    if line.startswith("Features:"):
        return 20, "HDBSCAN物理特征构建完成"
    if "[pa-hdbscan] merging tracklets" in line:
        return 82, "HDBSCAN轨迹片段合并"
    if "Summary:" in line:
        return 95, "HDBSCAN生成结果摘要"
    if "Saved final sort file:" in line:
        return 98, "HDBSCAN写入分选结果"
    return None


def _run_python_script(
    command: list[str],
    cwd: Path,
    log_file: Path,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
    timeout_seconds: Optional[int] = None,
    line_callback: LineCallback = None,
    parse_progress: bool = True,
) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.with_name("command.txt").write_text(" ".join(f'"{part}"' for part in command), encoding="utf-8")
    start = time.perf_counter()
    return_code = -1
    try:
        with open(log_file, "w", encoding="utf-8", errors="replace") as log:
            log.write(f"Command: {' '.join(command)}\n")
            log.write(f"Working directory: {cwd}\n")
            log.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            log.flush()
            _emit(progress_callback, 5, "外部算法进程启动")
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            line_queue: queue.Queue[str] = queue.Queue()

            def read_stdout() -> None:
                if process.stdout is None:
                    return
                for output_line in process.stdout:
                    line_queue.put(output_line)

            reader = threading.Thread(target=read_stdout, daemon=True)
            reader.start()

            def drain_lines() -> None:
                while True:
                    try:
                        output_line = line_queue.get_nowait()
                    except queue.Empty:
                        break
                    log.write(output_line)
                    if line_callback is not None:
                        try:
                            line_callback(output_line)
                        except Exception as exc:
                            log.write(f"[warn] line callback failed: {exc!r}\n")
                    if parse_progress:
                        parsed = _hdbscan_progress_from_line(output_line)
                        if parsed is not None:
                            _emit(progress_callback, parsed[0], parsed[1])
                log.flush()

            while True:
                drain_lines()
                if should_cancel is not None and should_cancel():
                    _terminate_process_tree(process)
                    log.write("\nExternal model cancelled by user.\n")
                    log.flush()
                    raise RuntimeError("外部算法已停止")
                return_code = process.poll()
                if return_code is not None:
                    reader.join(timeout=1.0)
                    drain_lines()
                    break
                if timeout_seconds is not None and time.perf_counter() - start > timeout_seconds:
                    _terminate_process_tree(process)
                    log.write(f"\nExternal model timed out after {timeout_seconds} seconds.\n")
                    log.flush()
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                time.sleep(0.2)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"外部算法运行超过 {timeout_seconds // 60 if timeout_seconds else 0} 分钟，已停止等待。\n"
            "建议先用小数据测试，或在命令行单独运行模型脚本查看进度。\n\n"
            f"日志文件：{log_file}"
        )
    if return_code != 0:
        output = log_file.read_text(encoding="utf-8", errors="replace") if log_file.exists() else ""
        if "ModuleNotFoundError" in output or "No module named" in output:
            output = "外部算法启动失败，当前 Python 环境缺少依赖。\n\n" + output
        raise RuntimeError((output.strip() or f"External model failed with exit code {return_code}") + f"\n\n日志文件：{log_file}")


def run_hdbscan_sorting(
    df: pd.DataFrame,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
    stream_callback: StreamCallback = None,
) -> pd.DataFrame:
    if not SORT_MODEL_DIR.exists():
        raise FileNotFoundError(f"分选模型目录不存在：{SORT_MODEL_DIR}")
    _require_modules(
        "HDBSCAN 分选",
        {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn"},
        f"python -m pip install -r {SORT_MODEL_DIR / 'requirements.txt'}",
    )

    run_dir = _run_dir("hdbscan_sort")
    print(f"[HDBSCAN] run_dir={run_dir}", flush=True)
    _emit(progress_callback, 3, f"HDBSCAN运行目录：{run_dir}")
    pdw_file = run_dir / "input_pdw.txt"
    output_dir = run_dir / "outputs_streaming_200ms"
    temp_dir = output_dir / "_tmp"
    summary_file = output_dir / "streaming_200ms_summary.csv"
    cycle_output_dir = run_dir / "outputs_cycle_period_200ms"
    total_beats = max(_beat_count_for_dataframe(df), 1)
    _emit(progress_callback, 6, "HDBSCAN准备200ms流式分选输入")
    write_external_pdw(df, pdw_file)
    cycle_module = _load_cycle_period_beat_module()
    cycle_cfg = cycle_module.MainSortConfig(
        input_dir=str(output_dir),
        truth_path="",
        output_dir=str(run_dir / "main5_reports"),
        stream_output_per_beat=True,
        stream_output_dir=str(cycle_output_dir),
        stream_watch_forever=False,
        stream_idle_timeout_seconds=0.0,
        stream_skip_existing_outputs=False,
        write_annotated_outputs=False,
        show_progress=False,
    )
    cycle_mcfg = cycle_module.build_candidate_cfg(cycle_cfg)
    cycle_output_dir.mkdir(parents=True, exist_ok=True)
    streaming_out = df.copy()
    streaming_out["Track_ID"] = 0
    streaming_out["HDBSCAN_Input_Track_ID"] = 0
    streaming_out["HDBSCAN_Track_ID"] = 0
    streaming_out["HDBSCAN_Assigned"] = False
    streaming_out["HDBSCAN_Sorting_Method"] = "HDBSCAN-Streaming200ms"
    streaming_out["CyclePeriod_Track_ID"] = 0
    streaming_out["CyclePeriod_OurPredID"] = 0
    streaming_out["CyclePeriod_Assigned"] = False
    streaming_out["CyclePeriod_Sorting_Method"] = "cycle_period-200ms"
    streaming_out["Assigned"] = False
    streaming_out["Sorting_Method"] = "HDBSCAN+cycle_period-200ms"
    streaming_out["HDBSCAN_Run_Dir"] = str(run_dir)
    streaming_out["HDBSCAN_Beat_Output_Dir"] = str(output_dir)
    streaming_out["CyclePeriod_Run_Dir"] = str(run_dir)
    streaming_out["CyclePeriod_Beat_Output_Dir"] = str(cycle_output_dir)
    stream_state = {"next_row": 0}

    def on_stream_line(line: str) -> None:
        if not line.startswith("[beat_result]"):
            return
        event = json.loads(line.split(" ", 1)[1])
        beat_file = Path(str(event["output_file"]))
        if not beat_file.is_absolute():
            beat_file = output_dir / beat_file
        beat_result = pd.read_csv(beat_file, sep=r"\s+", engine="python")
        if "SigIdx" not in beat_result.columns:
            return
        cycle_result = cycle_module.process_one_streaming_beat(beat_file, cycle_cfg, cycle_mcfg, cycle_output_dir)
        if cycle_result is not None:
            cycle_module.persist_stream_summary(cycle_result, cycle_output_dir)
        cycle_file = cycle_output_dir / beat_file.name
        if cycle_file.exists():
            cycle_beat = pd.read_csv(cycle_file, sep=r"\s+", engine="python")
        else:
            cycle_beat = beat_result.copy()
            cycle_beat[cycle_cfg.annotated_label_col] = pd.to_numeric(beat_result["SigIdx"], errors="coerce").fillna(0).astype(int)
        start = int(stream_state["next_row"])
        end = min(start + len(beat_result), len(streaming_out))
        if end <= start:
            return
        hdbscan_sigidx = pd.to_numeric(beat_result["SigIdx"], errors="coerce").fillna(0).astype(int).to_numpy()
        hdbscan_sigidx = hdbscan_sigidx[: end - start]
        raw_cycle_sigidx = pd.to_numeric(cycle_beat[cycle_cfg.annotated_label_col], errors="coerce").to_numpy()[: end - start]
        cycle_raw_pred = _raw_cycle_pred_ids(raw_cycle_sigidx)[: end - start]
        cycle_track_ids = _positive_cycle_track_ids(cycle_raw_pred, hdbscan_sigidx)
        streaming_out.iloc[start:end, streaming_out.columns.get_loc("HDBSCAN_Input_Track_ID")] = hdbscan_sigidx
        streaming_out.iloc[start:end, streaming_out.columns.get_loc("HDBSCAN_Track_ID")] = hdbscan_sigidx
        streaming_out.iloc[start:end, streaming_out.columns.get_loc("HDBSCAN_Assigned")] = hdbscan_sigidx > 0
        streaming_out.iloc[start:end, streaming_out.columns.get_loc("CyclePeriod_Track_ID")] = cycle_track_ids
        streaming_out.iloc[start:end, streaming_out.columns.get_loc("CyclePeriod_OurPredID")] = cycle_raw_pred
        streaming_out.iloc[start:end, streaming_out.columns.get_loc("CyclePeriod_Assigned")] = cycle_track_ids > 0
        streaming_out.iloc[start:end, streaming_out.columns.get_loc("Track_ID")] = cycle_track_ids
        streaming_out.iloc[start:end, streaming_out.columns.get_loc("Assigned")] = cycle_track_ids > 0
        stream_state["next_row"] = end
        completed_beats = min(int(event.get("beat", 0)) + 1, total_beats)
        pct = int(100 * completed_beats / total_beats)
        _emit(
            progress_callback,
            pct,
            (
                f"200ms节拍流水线：beat {completed_beats}/{total_beats} 完成 "
                f"(HDBSCAN+cycle_period)，进度 {pct}%"
            ),
        )
        if stream_callback is not None:
            stream_callback(_add_display_track_ids(streaming_out))

    script = SORT_MODEL_DIR / "streaming_200ms_sort.py"
    if not script.exists():
        raise FileNotFoundError(f"HDBSCAN 200ms流式分选脚本不存在：{script}")

    command = [
        sys.executable,
        "-u",
        str(script),
        "--input_file",
        str(pdw_file),
        "--output_dir",
        str(output_dir),
        "--temp_dir",
        str(temp_dir),
        "--sort_backend",
        "pa_tsr",
        "--n_jobs",
        "1",
    ]
    _run_python_script(
        command,
        SORT_MODEL_DIR,
        run_dir / "run.log",
        progress_callback,
        should_cancel,
        timeout_seconds=None,
        line_callback=on_stream_line,
        parse_progress=False,
    )

    if not summary_file.exists():
        raise RuntimeError(f"HDBSCAN 200ms流式分选已结束，但没有生成汇总文件：{summary_file}\n日志文件：{run_dir / 'run.log'}")
    _emit(progress_callback, 100, "200ms节拍流水线全部完成，读取HDBSCAN汇总结果")

    summary = pd.read_csv(summary_file)
    if "output_file" not in summary.columns:
        raise ValueError(f"HDBSCAN 200ms流式分选汇总缺少 output_file 字段：{summary_file}")

    beat_files = []
    beat_results = []
    for output_file in summary["output_file"].dropna().astype(str):
        beat_file = Path(output_file)
        if not beat_file.is_absolute():
            beat_file = output_dir / beat_file
        if not beat_file.exists():
            raise RuntimeError(f"HDBSCAN 200ms流式分选缺少beat结果文件：{beat_file}\n日志文件：{run_dir / 'run.log'}")
        beat_files.append(beat_file)
        beat_results.append(pd.read_csv(beat_file, sep=r"\s+", engine="python"))

    hdbscan_result = pd.concat(beat_results, ignore_index=True) if beat_results else pd.DataFrame()
    if "SigIdx" not in hdbscan_result.columns:
        raise ValueError(f"HDBSCAN 200ms流式分选输出缺少 SigIdx 字段：{summary_file}")
    if len(hdbscan_result) != len(df):
        raise ValueError(f"HDBSCAN 200ms流式分选输出行数不匹配：{len(hdbscan_result)} != {len(df)}")

    cycle_result = _read_main5_annotated_outputs(beat_files, cycle_output_dir, pred_col=cycle_cfg.annotated_label_col)
    if len(cycle_result) != len(df):
        raise ValueError(f"cycle_period 200ms主分选输出行数不匹配：{len(cycle_result)} != {len(df)}")

    out = df.copy()
    hdbscan_track_ids = pd.to_numeric(hdbscan_result["SigIdx"], errors="coerce").fillna(0).astype(int).to_numpy()
    cycle_raw_pred = _raw_cycle_pred_ids(cycle_result[cycle_cfg.annotated_label_col])
    cycle_track_ids = _positive_cycle_track_ids(cycle_raw_pred, hdbscan_track_ids)
    out["HDBSCAN_Input_Track_ID"] = hdbscan_track_ids
    out["HDBSCAN_Track_ID"] = hdbscan_track_ids
    out["HDBSCAN_Assigned"] = hdbscan_track_ids > 0
    out["HDBSCAN_Sorting_Method"] = "HDBSCAN-Streaming200ms"
    out["CyclePeriod_Track_ID"] = cycle_track_ids
    out["CyclePeriod_OurPredID"] = cycle_raw_pred
    out["CyclePeriod_Assigned"] = cycle_track_ids > 0
    out["CyclePeriod_Sorting_Method"] = "cycle_period-200ms"
    out["Track_ID"] = cycle_track_ids
    out["Assigned"] = cycle_track_ids > 0
    out["Sorting_Method"] = "HDBSCAN+cycle_period-200ms"
    out["HDBSCAN_Run_Dir"] = str(run_dir)
    out["HDBSCAN_Beat_Output_Dir"] = str(output_dir)
    out["CyclePeriod_Run_Dir"] = str(run_dir)
    out["CyclePeriod_Beat_Output_Dir"] = str(cycle_output_dir)
    return _add_display_track_ids(out)


def _load_cycle_period_module():
    if not CYCLE_PERIOD_MODEL_FILE.exists():
        raise FileNotFoundError(f"cycle_period 模型文件不存在：{CYCLE_PERIOD_MODEL_FILE}")
    module_name = "cycle_period_sort_model"
    spec = importlib.util.spec_from_file_location(module_name, CYCLE_PERIOD_MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 cycle_period 模型文件：{CYCLE_PERIOD_MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_cycle_period_beat_module():
    if not CYCLE_PERIOD_BEAT_MODEL_FILE.exists():
        raise FileNotFoundError(f"200ms cycle_period 主分选脚本不存在：{CYCLE_PERIOD_BEAT_MODEL_FILE}")
    module_name = "cycle_period_200ms_main5_model"
    spec = importlib.util.spec_from_file_location(module_name, CYCLE_PERIOD_BEAT_MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 200ms cycle_period 主分选脚本：{CYCLE_PERIOD_BEAT_MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _constant_path_from_column(df: pd.DataFrame, column: str) -> Optional[Path]:
    if column not in df.columns:
        return None
    values = [str(value) for value in df[column].dropna().unique() if str(value).strip()]
    if not values:
        return None
    path = Path(values[0])
    return path if path.exists() else None


def _write_presort_beats_from_dataframe(df: pd.DataFrame, output_dir: Path, beat_seconds: float = 0.2) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    toa = _toa_seconds(df).astype(float).to_numpy()
    t0 = float(np.min(toa)) if len(toa) else 0.0
    beat_ids = np.floor((toa - t0) / float(beat_seconds)).astype(np.int64)
    source_tracks = pd.to_numeric(df["Track_ID"], errors="coerce").fillna(0).astype(int).to_numpy()
    beat_files: list[Path] = []
    for beat_id in sorted(set(int(value) for value in beat_ids)):
        mask = beat_ids == beat_id
        beat = pd.DataFrame(
            {
                "TOA(s)": toa[mask],
                "Param1": _series(df.loc[mask], "RF", 0.0).to_numpy(),
                "Param2": _series(df.loc[mask], "PW", 0.0).to_numpy(),
                "Param3": _series(df.loc[mask], "PRI", 0.0).to_numpy(),
                "Param4": _series(df.loc[mask], "PA", 0.0).to_numpy(),
                "Param5": _series(df.loc[mask], "DOA", 0.0).to_numpy(),
                "Param6": np.zeros(int(np.sum(mask)), dtype=int),
                "Param7": np.zeros(int(np.sum(mask)), dtype=float),
                "SigIdx": source_tracks[mask],
            }
        )
        path = output_dir / f"beat_{beat_id:06d}.txt"
        beat.to_csv(path, sep=" ", index=False, float_format="%.9f")
        beat_files.append(path)
    return beat_files


def _beat_count_for_dataframe(df: pd.DataFrame, beat_seconds: float = 0.2) -> int:
    toa = _toa_seconds(df).astype(float).to_numpy()
    if len(toa) == 0:
        return 0
    t0 = float(np.min(toa))
    beat_ids = np.floor((toa - t0) / float(beat_seconds)).astype(np.int64)
    return int(len(set(int(value) for value in beat_ids)))


def _beat_files_for_cycle_period(df: pd.DataFrame, run_dir: Path) -> tuple[Path, list[Path]]:
    beat_dir = _constant_path_from_column(df, "HDBSCAN_Beat_Output_Dir")
    if beat_dir is not None:
        beat_files = sorted(beat_dir.glob("beat_*.txt"))
        if beat_files:
            return beat_dir, beat_files
    generated_dir = run_dir / "input_hdbscan_200ms_beats"
    return generated_dir, _write_presort_beats_from_dataframe(df, generated_dir)


def _read_main5_annotated_outputs(beat_files: list[Path], output_dir: Path, pred_col: str = "OurPredID") -> pd.DataFrame:
    parts = []
    for source_file in beat_files:
        path = output_dir / source_file.name
        if not path.exists():
            raise RuntimeError(f"200ms cycle_period 主分选缺少beat输出文件：{path}")
        beat = pd.read_csv(path, sep=r"\s+", engine="python")
        if pred_col not in beat.columns:
            raise ValueError(f"200ms cycle_period 主分选输出缺少 {pred_col} 字段：{path}")
        parts.append(beat)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _apply_cycle_period_beat_result(
    df: pd.DataFrame,
    beat_files: list[Path],
    output_dir: Path,
    run_dir: Path,
    pred_col: str = "OurPredID",
) -> pd.DataFrame:
    result = _read_main5_annotated_outputs(beat_files, output_dir, pred_col=pred_col)
    if len(result) != len(df):
        raise ValueError(f"cycle_period 200ms 主分选输出行数不匹配：{len(result)} != {len(df)}")
    source_sigidx = result["SigIdx"] if "SigIdx" in result.columns else df.get("HDBSCAN_Input_Track_ID", df.get("Track_ID"))
    raw_pred = _raw_cycle_pred_ids(result[pred_col])
    pred_tracks = _positive_cycle_track_ids(raw_pred, source_sigidx)
    out = df.copy()
    if "HDBSCAN_Input_Track_ID" not in out.columns and "Track_ID" in out.columns:
        out["HDBSCAN_Input_Track_ID"] = pd.to_numeric(out["Track_ID"], errors="coerce").fillna(0).astype(int)
    out["Track_ID"] = pred_tracks
    out["CyclePeriod_Track_ID"] = pred_tracks
    out["CyclePeriod_OurPredID"] = raw_pred
    out["CyclePeriod_Assigned"] = pred_tracks > 0
    out["Assigned"] = pred_tracks > 0
    out["Sorting_Method"] = "cycle_period-200ms"
    out["CyclePeriod_Run_Dir"] = str(run_dir)
    out["CyclePeriod_Beat_Output_Dir"] = str(output_dir)
    return _add_display_track_ids(out)


def _cycle_period_config(module):
    cfg = module.Config()
    cfg.show_progress = False
    cfg.frame_T_min = 50e-6
    cfg.frame_T_max = 5e-3
    cfg.frame_candidate_bin = 0.5e-6
    cfg.num_base_samples = 80
    cfg.num_ref_per_base = 500
    cfg.top_k_frame_candidates = 10
    cfg.frame_toa_tol = 3e-6
    cfg.max_score_points = 8000
    cfg.min_pulses_for_tframe = 10
    cfg.min_hit_rate_for_valid = 0.45
    cfg.min_span_rate_for_valid = 0.40
    cfg.min_confidence_for_valid = 0.40
    cfg.merge_min_confidence = 0.55
    cfg.merge_min_hit_rate = 0.45
    cfg.merge_min_span_rate = 0.30
    cfg.merge_T_rel_tol = 0.022
    cfg.merge_abs_tol = 3.0e-6
    cfg.allow_harmonic_merge = False
    cfg.merge_verify = True
    cfg.merge_verify_min_hit_rate = 0.38
    cfg.merge_verify_min_span_rate = 0.25
    cfg.merge_verify_min_confidence = 0.34
    cfg.merge_min_size_ratio = 0.002
    return cfg


def _write_cycle_period_metrics(module, pred_df: pd.DataFrame, truth_file: Path, cfg, run_dir: Path) -> None:
    truth_raw = module.read_table_auto(truth_file)
    truth_df = module.normalize_truth_df(truth_raw, toa_col="TOA(s)", true_col="SigIdx")
    eval_df = module.align_pred_and_truth(pred_df, truth_df, cfg, prefer_pulse_id=False)
    metrics = module.evaluate_sorting_official_like(
        eval_df,
        cfg,
        pred_col="PredID",
        true_col="TrueLabel",
        toa_col="TOA",
        exp_name="cycle_period",
    )
    summary = {
        "truth_file": str(truth_file),
        "matched_rows": int(len(eval_df)),
        "Sort_ACC": float(metrics["Sort_ACC"]),
        "Add_Rate": float(metrics["Add_Rate"]),
        "Err_Rate": float(metrics["Err_Rate"]),
    }
    (run_dir / "cycle_period_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metrics["target_detail"].to_csv(run_dir / "cycle_period_target_detail.csv", index=False, encoding="utf-8-sig")
    metrics["beat_detail"].to_csv(run_dir / "cycle_period_beat_detail.csv", index=False, encoding="utf-8-sig")


def run_cycle_period_sorting(
    df: pd.DataFrame,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
) -> pd.DataFrame:
    if "Track_ID" not in df.columns:
        raise ValueError("cycle_period 需要先完成 HDBSCAN 分选，缺少 Track_ID 字段")
    _require_modules(
        "cycle_period 200ms 主分选",
        {"numpy": "numpy", "pandas": "pandas"},
        f"python -m pip install -r {CYCLE_PERIOD_BEAT_MODEL_FILE.parent / 'requirements.txt'}",
    )
    run_dir = _run_dir("cycle_period_sort")
    print(f"[cycle_period] run_dir={run_dir}", flush=True)
    _emit(progress_callback, 5, "cycle_period加载200ms主分选脚本")
    module = _load_cycle_period_beat_module()
    beat_input_dir, beat_files = _beat_files_for_cycle_period(df, run_dir)
    if not beat_files:
        raise RuntimeError(f"cycle_period 200ms 主分选没有可用beat输入：{beat_input_dir}")
    cached_output_dir = _constant_path_from_column(df, "CyclePeriod_Beat_Output_Dir")
    if cached_output_dir is not None:
        cached_files = sorted(cached_output_dir.glob("beat_*.txt"))
        if len(cached_files) >= len(beat_files):
            _emit(progress_callback, 100, f"cycle_period复用已完成的200ms节拍结果：{len(beat_files)} 个beat")
            out = _apply_cycle_period_beat_result(
                df,
                beat_files,
                cached_output_dir,
                run_dir,
                pred_col="OurPredID",
            )
            out["CyclePeriod_Beat_Input_Dir"] = str(beat_input_dir)
            return out

    _emit(progress_callback, 0, f"cycle_period 200ms主分选准备：总节拍数 {len(beat_files)}")
    output_dir = run_dir / "outputs_cycle_period_200ms"
    cfg = module.MainSortConfig(
        input_dir=str(beat_input_dir),
        truth_path="",
        output_dir=str(run_dir / "main5_reports"),
        stream_output_per_beat=True,
        stream_output_dir=str(output_dir),
        stream_watch_forever=False,
        stream_idle_timeout_seconds=0.0,
        stream_skip_existing_outputs=False,
        write_annotated_outputs=False,
        show_progress=False,
    )

    _check_cancelled(should_cancel)
    _emit(progress_callback, 1, "cycle_period 200ms主分选开始")
    output_dir.mkdir(parents=True, exist_ok=True)
    mcfg = module.build_candidate_cfg(cfg)
    summary_rows = []
    total_beats = max(len(beat_files), 1)
    for index, beat_file in enumerate(beat_files, start=1):
        _check_cancelled(should_cancel)
        start_pct = int(100 * (index - 1) / total_beats)
        _emit(progress_callback, start_pct, f"cycle_period 200ms主分选：beat {index}/{total_beats} 开始")
        result_row = module.process_one_streaming_beat(beat_file, cfg, mcfg, output_dir)
        if result_row is None:
            continue
        summary_rows.append(result_row)
        module.persist_stream_summary(result_row, output_dir)
        done_pct = int(100 * index / total_beats)
        _emit(
            progress_callback,
            done_pct,
            (
                f"cycle_period 200ms主分选：beat {index}/{total_beats} 完成，"
                f"进度 {done_pct}%，"
                f"{result_row.get('InputBatches', 0)}->{result_row.get('OutputBatches', 0)} 批"
            ),
        )
    summary = pd.DataFrame(summary_rows)
    _check_cancelled(should_cancel)

    _emit(progress_callback, 100, "cycle_period 200ms主分选全部beat完成，读取结果")
    result = _read_main5_annotated_outputs(beat_files, output_dir, pred_col=cfg.annotated_label_col)
    if len(result) != len(df):
        raise ValueError(f"cycle_period 200ms 主分选输出行数不匹配：{len(result)} != {len(df)}")

    source_sigidx = result["SigIdx"] if "SigIdx" in result.columns else df["Track_ID"]
    raw_pred = _raw_cycle_pred_ids(result[cfg.annotated_label_col])
    pred_tracks = _positive_cycle_track_ids(raw_pred, source_sigidx)

    out = df.copy()
    out["HDBSCAN_Input_Track_ID"] = pd.to_numeric(out["Track_ID"], errors="coerce").fillna(0).astype(int)
    out["Track_ID"] = pred_tracks
    out["CyclePeriod_Track_ID"] = pred_tracks
    out["CyclePeriod_OurPredID"] = raw_pred
    out["CyclePeriod_Assigned"] = pred_tracks > 0
    out["Assigned"] = pred_tracks > 0
    out["Sorting_Method"] = "cycle_period-200ms"
    out["CyclePeriod_Run_Dir"] = str(run_dir)
    out["CyclePeriod_Beat_Input_Dir"] = str(beat_input_dir)
    out["CyclePeriod_Beat_Output_Dir"] = str(output_dir)
    if summary is not None and len(summary) > 0:
        summary.to_csv(run_dir / "cycle_period_200ms_stream_summary.csv", index=False, encoding="utf-8-sig")
    _emit(progress_callback, 98, "cycle_period 200ms主分选结果已生成")
    return _add_display_track_ids(out)


def _confidence_by_track(batch_file: Path) -> Dict[int, float]:
    if not batch_file.exists():
        return {}
    batches = pd.read_csv(batch_file)
    confidence: Dict[int, float] = {}
    for _, row in batches.iterrows():
        track_id = int(row.get("pred_sigidx", 0))
        dist = float(row.get("template_distance", np.nan))
        threshold = float(row.get("template_distance_threshold", np.nan))
        if np.isfinite(dist) and np.isfinite(threshold) and threshold > 0:
            value = float(np.clip(1.0 - dist / threshold, 0.05, 0.99))
        else:
            value = 0.8 if int(row.get("batch_pred_label", 99)) != 99 else 0.5
        confidence[track_id] = value
    return confidence


def run_zeng_recognition(
    df: pd.DataFrame,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RECOGNITION_MODEL_DIR.exists():
        raise FileNotFoundError(f"识别模型目录不存在：{RECOGNITION_MODEL_DIR}")
    _require_modules(
        "zeng 识别",
        {"numpy": "numpy", "pandas": "pandas"},
        f"python -m pip install -r {RECOGNITION_MODEL_DIR / 'requirements.txt'}",
    )
    if "Track_ID" not in df.columns:
        raise ValueError("zeng 识别需要先完成分选，缺少 Track_ID 字段")

    template_library = zeng_template_library_path()
    if not template_library.exists():
        raise FileNotFoundError(f"zeng 识别需要训练生成模板库，当前缺少：{template_library}")

    run_dir = _run_dir("zeng_recognition")
    print(f"[zeng] run_dir={run_dir}", flush=True)
    _emit(progress_callback, 5, f"zeng运行目录：{run_dir}")
    pdw_file = run_dir / "input_pdw.txt"
    sort_file = run_dir / "input_sort.txt"
    output_dir = run_dir / "output"
    write_external_pdw(df, pdw_file)
    write_external_sort(df, sort_file)

    script = RECOGNITION_MODEL_DIR / "template_match_recognition.py"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--sample",
        "sample1",
        "--pdw_file",
        str(pdw_file),
        "--sort_file",
        str(sort_file),
        "--output_dir",
        str(output_dir),
        "--template_library",
        str(template_library),
    ]
    _run_python_script(command, RECOGNITION_MODEL_DIR, run_dir / "run.log", progress_callback, should_cancel)

    final_file = output_dir / "sample1_template_match_final.txt"
    batch_file = output_dir / "sample1_template_match_batches.csv"
    if not final_file.exists():
        raise RuntimeError(f"zeng 外部算法已结束，但没有生成识别结果：{final_file}\n日志文件：{run_dir / 'run.log'}")
    final = pd.read_csv(final_file, sep=r"\s+", engine="python")
    if "LABEL" not in final.columns:
        raise ValueError(f"zeng 识别输出缺少 LABEL 字段：{final_file}")
    if len(final) != len(df):
        raise ValueError(f"zeng 识别输出行数不匹配：{len(final)} != {len(df)}")

    out = df.copy()
    labels = pd.to_numeric(final["LABEL"], errors="coerce").fillna(99).astype(int)
    track_ids = pd.to_numeric(out["Track_ID"], errors="coerce").fillna(0).astype(int)
    confidences = _confidence_by_track(batch_file)
    out["Predicted_Label"] = labels.map(lambda value: "Unknown" if int(value) == 99 else f"Class_{int(value)}")
    out["Confidence"] = track_ids.map(lambda value: confidences.get(int(value), 0.5 if int(value) > 0 else 0.0))

    rows = []
    valid = out[track_ids > 0]
    for track_id, group in valid.groupby("Track_ID", sort=True):
        pulse_count = int(len(group))
        predicted_label = str(group["Predicted_Label"].mode().iloc[0]) if pulse_count else "Unknown"
        rows.append(
            {
                "Track_ID": int(track_id),
                "Pulse_Count": pulse_count,
                "Mean_RF": float(group["RF"].mean()) if "RF" in group else 0.0,
                "Mean_PW": float(group["PW"].mean()) if "PW" in group else 0.0,
                "Mean_PRI": float(group["PRI"].mean()) if "PRI" in group else 0.0,
                "Predicted_Label": predicted_label,
                "Confidence": float(group["Confidence"].mean()),
            }
        )
    return out, pd.DataFrame(rows)


def run_zeng_template_training(
    train_dir: str | Path,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
) -> Dict[str, object]:
    if not RECOGNITION_MODEL_DIR.exists():
        raise FileNotFoundError(f"识别模型目录不存在：{RECOGNITION_MODEL_DIR}")
    _require_modules(
        "zeng 模板库生成",
        {"numpy": "numpy", "pandas": "pandas"},
        f"python -m pip install -r {RECOGNITION_MODEL_DIR / 'requirements.txt'}",
    )

    train_path = Path(train_dir)
    if not train_path.exists() or not train_path.is_dir():
        raise FileNotFoundError(f"训练数据目录不存在：{train_path}")
    class_files = sorted(train_path.glob("Class_*.txt"))
    if not class_files:
        raise FileNotFoundError(f"训练数据目录缺少 Class_*.txt 文件：{train_path}")

    build_script = RECOGNITION_MODEL_DIR / "build_expanded_template_library.py"
    tune_script = RECOGNITION_MODEL_DIR / "tune_template_match_parameters.py"
    if not build_script.exists():
        raise FileNotFoundError(f"缺少模板库生成脚本：{build_script}")
    if not tune_script.exists():
        raise FileNotFoundError(f"缺少模板库调参脚本：{tune_script}")

    output_dir = RECOGNITION_MODEL_DIR / "outputs_expanded_template_library"
    run_dir = _run_dir("zeng_template_training")
    _emit(progress_callback, 5, f"zeng训练数据目录：{train_path}")

    _emit(progress_callback, 20, "生成 zeng 模板库")
    build_command = [
        sys.executable,
        "-u",
        str(build_script),
        "--train_dir",
        str(train_path),
        "--output_dir",
        str(output_dir),
    ]
    _run_python_script(build_command, RECOGNITION_MODEL_DIR, run_dir / "build_template_library.log", progress_callback, should_cancel)

    template_library = output_dir / "template_library.json"
    template_csv = output_dir / "template_library.csv"
    if not template_library.exists():
        raise RuntimeError(f"模板库生成结束，但没有找到：{template_library}")

    _emit(progress_callback, 70, "调优 zeng 模板匹配参数")
    tune_command = [
        sys.executable,
        "-u",
        str(tune_script),
        "--train_dir",
        str(train_path),
        "--output_dir",
        str(output_dir),
    ]
    _run_python_script(tune_command, RECOGNITION_MODEL_DIR, run_dir / "tune_template_parameters.log", progress_callback, should_cancel)

    tuned_parameters = output_dir / "tuned_match_parameters.json"
    if not tuned_parameters.exists():
        raise RuntimeError(f"模板参数调优结束，但没有找到：{tuned_parameters}")

    _emit(progress_callback, 98, "zeng 模板库更新完成")
    return {
        "train_dir": str(train_path),
        "class_count": len(class_files),
        "class_files": [path.name for path in class_files],
        "template_library": str(template_library),
        "template_csv": str(template_csv),
        "tuned_parameters": str(tuned_parameters),
        "run_dir": str(run_dir),
    }

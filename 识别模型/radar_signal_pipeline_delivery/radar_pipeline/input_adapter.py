"""
输入适配器

将上游固定 200ms 窗口数据转换为 WindowRecord 序列。
修复原 data_loader.py 的 float32 精度问题。

关键修复:
  - TOA 使用 float64, 信号参数保留 float32
  - 使用统一绝对基准 floor(TOA / 0.2) 作为窗口编号
  - 保留空窗口和窗口编号
  - 边界脉冲不丢失、不重复
"""
from __future__ import annotations
import re
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from .schemas import WindowRecord


WINDOW_DURATION = 0.200  # 固定 200ms
LABEL_COLUMNS = ("PRED_LABEL", "LABEL")


def _compute_window_id(toa: float, base_time: float = 0.0) -> int:
    """
    计算窗口编号: 使用统一绝对基准 floor((TOA - base_time) / 0.2)

    不能使用每部雷达第一条脉冲作为窗口起点，
    否则不同雷达的窗口编号不可比较。

    添加小 epsilon 处理浮点精度问题。
    """
    # 添加小 epsilon 处理浮点精度 (如 1.2-1.0 = 0.19999999999999996)
    epsilon = 1e-9
    return int(np.floor((toa - base_time + epsilon) / WINDOW_DURATION))


def load_and_window(
    file_paths: list[str],
    min_pulses: int = 100,
    base_time: float = 0.0,
) -> dict[str, list[WindowRecord]]:
    """
    加载数据文件并按固定 200ms 窗口组织。

    Args:
        file_paths: 输入文件路径列表
        min_pulses: 每部雷达最少脉冲数
        base_time: 窗口编号的统一基准时间 (默认 0.0, 即绝对时间)

    Returns:
        {radar_key: [WindowRecord, ...]} 按 window_id 排序
    """
    all_radars: dict[str, list[WindowRecord]] = {}

    for path in file_paths:
        sample_name = Path(path).stem.split("_")[0]
        radars = _load_single_file(path, sample_name, base_time)

        for rid, windows in radars.items():
            total_pulses = sum(w.n_pulses for w in windows)
            if total_pulses < min_pulses:
                continue
            key = f"{sample_name}/{rid}"
            all_radars[key] = windows

    return all_radars


def _load_single_file(
    path: str,
    sample_name: str,
    base_time: float,
) -> dict[str, list[WindowRecord]]:
    """加载单个文件，按 PRED_LABEL 拆分并按窗口组织"""
    df = pd.read_csv(path, sep=r'\s+')
    label_col = _detect_label_column(df)

    result: dict[str, list[WindowRecord]] = {}

    for label, group in df.groupby(label_col):
        group = group.sort_values('TOA(s)')
        rid = f"radar_{int(label)}"

        # ── 关键修复: TOA 使用 float64 ──
        toa = group['TOA(s)'].values.astype(np.float64)
        # 信号参数可以保留 float32
        signal = group[['Param1', 'Param2', 'Param4', 'Param5', 'Param6']].values.astype(np.float32)

        # 检查 DOA 有效性
        doa = signal[:, 3]  # Param5 = DOA
        doa_valid = _validate_doa(doa)

        # 计算每个脉冲的窗口编号
        window_ids = np.array([_compute_window_id(t, base_time) for t in toa])

        # 获取窗口编号范围
        if len(window_ids) == 0:
            result[rid] = []
            continue

        min_wid = int(window_ids.min())
        max_wid = int(window_ids.max())

        # 构建完整窗口序列 (包含空窗口)
        windows: list[WindowRecord] = []
        for wid in range(min_wid, max_wid + 1):
            w_start = base_time + wid * WINDOW_DURATION
            w_end = w_start + WINDOW_DURATION

            mask = window_ids == wid
            n_pulses = int(mask.sum())

            if n_pulses == 0:
                # 空窗口
                windows.append(WindowRecord(
                    radar_id=rid,
                    sample=sample_name,
                    window_id=wid,
                    start_time=w_start,
                    end_time=w_end,
                    pdw=None,
                    n_pulses=0,
                    is_empty=True,
                    quality_flags=["empty"],
                ))
            else:
                # 非空窗口: 构建 PDW 矩阵 (TOA float64 + signal float32)
                win_toa = toa[mask]
                win_signal = signal[mask]

                # 合并: [TOA(float64), RF, PW, PA, DOA, Param6]
                pdw = np.column_stack([win_toa, win_signal])

                # 质量检查
                qflags = _check_window_quality(pdw, n_pulses)

                windows.append(WindowRecord(
                    radar_id=rid,
                    sample=sample_name,
                    window_id=wid,
                    start_time=w_start,
                    end_time=w_end,
                    pdw=pdw,
                    n_pulses=n_pulses,
                    is_empty=False,
                    quality_flags=qflags,
                    upstream_label=int(label),
                ))

        # DOA 有效性标记传递到所有窗口
        if not doa_valid:
            for w in windows:
                w.quality_flags.append("doa_invalid")

        result[rid] = windows

    return result


def _validate_doa(doa: np.ndarray) -> bool:
    """检查 DOA 列是否包含有效数据"""
    if len(doa) == 0:
        return False
    doa_std = float(np.std(doa))
    doa_nonzero = float(np.sum(doa != 0) / max(len(doa), 1))
    doa_range = float(np.ptp(doa))
    return (
        doa_std > 0.01
        and doa_nonzero > 0.5
        and doa_range > 0.1
        and float(np.min(doa)) >= -10
        and float(np.max(doa)) <= 370
    )


def _check_window_quality(pdw: np.ndarray, n_pulses: int) -> list[str]:
    """检查单个窗口的数据质量"""
    flags: list[str] = []

    if n_pulses < 5:
        flags.append("low_pulse_count")

    toa = pdw[:, 0]  # float64
    rf = pdw[:, 1]
    pw = pdw[:, 2]
    pa = pdw[:, 3]
    doa = pdw[:, 4]

    # TOA 单调性
    if n_pulses > 1:
        toa_diffs = np.diff(toa)
        if np.any(toa_diffs < 0):
            flags.append("toa_non_monotonic")

    # RF 有效性
    if np.any(rf < 0) or np.any(rf > 50000):
        flags.append("invalid_rf")

    # PW 有效性
    if np.any(pw < 0) or np.any(pw > 1000):
        flags.append("invalid_pw")

    # PA 有效性
    if np.any(pa < -100) or np.any(pa > 100):
        flags.append("invalid_pa")

    # DOA 有效性
    if np.any(doa < 0) or np.any(doa > 360):
        flags.append("invalid_doa")

    return flags


def _detect_label_column(df: pd.DataFrame) -> str:
    for col in LABEL_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"输入数据缺少标签列，需要包含 {LABEL_COLUMNS}")


def _infer_sample_name(path: str) -> str:
    stem = Path(path).stem
    match = re.search(r"(sample\d+)", stem, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower()
    parts = stem.split("_")
    return parts[0] if parts else stem


def _beat_index(name: str, fallback: int) -> int:
    match = re.search(r"beat[_-]?(\d+)", Path(name).stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\d+", Path(name).stem)
    if numbers:
        return int(numbers[-1])
    return fallback


def _build_presegmented_window(
    df: pd.DataFrame,
    sample_name: str,
    window_id: int,
) -> dict[str, WindowRecord]:
    label_col = _detect_label_column(df)
    records: dict[str, WindowRecord] = {}
    for label, group in df.groupby(label_col):
        group = group.sort_values('TOA(s)')
        rid = f"radar_{int(label)}"

        toa = group['TOA(s)'].values.astype(np.float64)
        signal = group[['Param1', 'Param2', 'Param4', 'Param5', 'Param6']].values.astype(np.float32)
        n_pulses = len(toa)
        if n_pulses == 0:
            continue

        doa = signal[:, 3]
        doa_valid = _validate_doa(doa)
        pdw = np.column_stack([toa, signal])
        qflags = _check_window_quality(pdw, n_pulses)
        if not doa_valid:
            qflags.append("doa_invalid")

        w_start = window_id * WINDOW_DURATION
        w_end = w_start + WINDOW_DURATION
        records[rid] = WindowRecord(
            radar_id=rid,
            sample=sample_name,
            window_id=window_id,
            start_time=w_start,
            end_time=w_end,
            pdw=pdw,
            n_pulses=n_pulses,
            is_empty=False,
            quality_flags=qflags,
            upstream_label=int(label),
        )
    return records


def verify_conservation(
    all_radars: dict[str, list[WindowRecord]],
    original_pulse_counts: dict[str, int],
) -> dict[str, dict]:
    """
    验证脉冲守恒和窗口守恒。

    Returns:
        {radar_key: {input_pulses, output_pulses, match, window_count, empty_count}}
    """
    report = {}
    for key, windows in all_radars.items():
        output_pulses = sum(w.n_pulses for w in windows)
        empty_count = sum(1 for w in windows if w.is_empty)
        input_pulses = original_pulse_counts.get(key, output_pulses)
        report[key] = {
            "input_pulses": input_pulses,
            "output_pulses": output_pulses,
            "match": input_pulses == output_pulses,
            "window_count": len(windows),
            "empty_count": empty_count,
        }
    return report


# ── 200ms 预分段文件加载 ─────────────────────────────────────────────

def load_presegmented_files(
    file_paths: list[str],
    min_pulses: int = 100,
) -> dict[str, list[WindowRecord]]:
    """
    加载已按 200ms 切分的文件。

    每个文件是一个 200ms 窗口的脉冲数据，格式与标准输入相同
    (9列: TOA, RF, PW, dTOA, PA, DOA, Param6, Param7, PRED_LABEL)。
    文件按名称排序，依次分配窗口编号。

    Args:
        file_paths: 输入文件路径列表 (每个文件 = 一个 200ms 窗口)
        min_pulses: 每部雷达最少脉冲数

    Returns:
        {radar_key: [WindowRecord, ...]} 按 window_id 排序
    """
    # {sample_name: {radar_id: {window_id: WindowRecord}}}
    radar_windows_by_sample: dict[str, dict[str, dict[int, WindowRecord]]] = {}

    for path in sorted(file_paths):
        path_obj = Path(path)
        if path_obj.suffix.lower() == ".zip":
            _load_presegmented_zip(path, radar_windows_by_sample)
            continue

        sample_name = _infer_sample_name(path)
        try:
            df = pd.read_csv(path, sep=r'\s+')
        except Exception:
            continue

        if df.empty:
            continue

        window_id = _beat_index(path_obj.name, len(radar_windows_by_sample.get(sample_name, {})))
        records = _build_presegmented_window(df, sample_name, window_id)
        sample_map = radar_windows_by_sample.setdefault(sample_name, {})
        for rid, rec in records.items():
            sample_map.setdefault(rid, {})[window_id] = rec

    # 补全空窗口 + 按 window_id 排序
    all_radars: dict[str, list[WindowRecord]] = {}
    for sample_name, radar_windows in radar_windows_by_sample.items():
        for rid, wmap in radar_windows.items():
            if not wmap:
                continue
            min_wid = min(wmap.keys())
            max_wid = max(wmap.keys())
            key = f"{sample_name}/{rid}"

            windows = []
            for wid in range(min_wid, max_wid + 1):
                if wid in wmap:
                    windows.append(wmap[wid])
                else:
                    windows.append(WindowRecord(
                        radar_id=rid,
                        sample=sample_name,
                        window_id=wid,
                        start_time=wid * WINDOW_DURATION,
                        end_time=(wid + 1) * WINDOW_DURATION,
                        pdw=None,
                        n_pulses=0,
                        is_empty=True,
                        quality_flags=["empty"],
                    ))

            total_pulses = sum(w.n_pulses for w in windows)
            if total_pulses >= min_pulses:
                all_radars[key] = windows

    return all_radars


def _load_presegmented_zip(
    path: str,
    out: dict[str, dict[str, dict[int, WindowRecord]]],
) -> None:
    sample_name = _infer_sample_name(path)
    sample_map = out.setdefault(sample_name, {})
    with zipfile.ZipFile(path) as zf:
        members = [
            info for info in zf.infolist()
            if not info.is_dir() and info.filename.lower().endswith((".txt", ".csv"))
        ]
        members = sorted(members, key=lambda info: (_beat_index(info.filename, 0), info.filename))
        for fallback_idx, info in enumerate(members):
            window_id = _beat_index(info.filename, fallback_idx)
            with zf.open(info) as fp:
                try:
                    df = pd.read_csv(fp, sep=r'\s+')
                except Exception:
                    continue
            if df.empty:
                continue
            records = _build_presegmented_window(df, sample_name, window_id)
            for rid, rec in records.items():
                sample_map.setdefault(rid, {})[window_id] = rec

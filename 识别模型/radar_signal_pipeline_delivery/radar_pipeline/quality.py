"""
窗口质量检查

为每个窗口输出质量标志，不删除任何窗口。
空窗口和低脉冲窗口保留，下游通过 valid_mask 处理。
"""
from __future__ import annotations
import numpy as np
from .schemas import WindowRecord, WindowFeatures


# 质量标志定义
QUALITY_FLAGS = {
    "empty":                "窗口无脉冲",
    "low_pulse_count":      "脉冲数不足 (<5)",
    "toa_non_monotonic":    "TOA 非单调递增",
    "invalid_rf":           "RF 值超出物理范围",
    "invalid_pw":           "PW 值超出物理范围",
    "invalid_pa":           "PA 值超出物理范围",
    "invalid_doa":          "DOA 值超出物理范围",
    "doa_invalid":          "DOA 列整体无效",
    "upstream_low_confidence": "上游识别置信度低",
    "possible_mixed_emitter":  "可能包含混合辐射源脉冲",
}


def check_window(record: WindowRecord) -> list[str]:
    """
    对单个窗口执行质量检查，返回质量标志列表。

    不修改窗口数据，不删除窗口。
    """
    flags = list(record.quality_flags)  # 保留已有标志

    if record.is_empty:
        if "empty" not in flags:
            flags.append("empty")
        return flags

    pdw = record.pdw
    if pdw is None:
        return flags

    n = record.n_pulses
    toa = pdw[:, 0]  # float64
    rf = pdw[:, 1]
    pw = pdw[:, 2]
    pa = pdw[:, 3]
    doa = pdw[:, 4]

    # 低脉冲
    if n < 5 and "low_pulse_count" not in flags:
        flags.append("low_pulse_count")

    # TOA 单调性
    if n > 1:
        toa_diffs = np.diff(toa)
        if np.any(toa_diffs < 0) and "toa_non_monotonic" not in flags:
            flags.append("toa_non_monotonic")

    # RF 有效性
    if (np.any(rf < 0) or np.any(rf > 50000)) and "invalid_rf" not in flags:
        flags.append("invalid_rf")

    # PW 有效性
    if (np.any(pw < 0) or np.any(pw > 1000)) and "invalid_pw" not in flags:
        flags.append("invalid_pw")

    # PA 有效性
    if (np.any(pa < -100) or np.any(pa > 100)) and "invalid_pa" not in flags:
        flags.append("invalid_pa")

    # DOA 有效性
    if (np.any(doa < 0) or np.any(doa > 360)) and "invalid_doa" not in flags:
        flags.append("invalid_doa")

    return flags


def batch_check(windows: list[WindowRecord]) -> list[list[str]]:
    """批量质量检查"""
    return [check_window(w) for w in windows]


def compute_quality_summary(windows: list[WindowRecord]) -> dict:
    """
    计算窗口序列的质量汇总。

    Returns:
        {total, empty, low_pulse, valid, valid_ratio, flag_counts}
    """
    total = len(windows)
    empty = sum(1 for w in windows if w.is_empty)
    low_pulse = sum(
        1 for w in windows
        if not w.is_empty and w.n_pulses < 5
    )
    valid = sum(
        1 for w in windows
        if not w.is_empty and w.n_pulses >= 5
        and "toa_non_monotonic" not in w.quality_flags
    )

    # 统计各标志出现次数
    flag_counts: dict[str, int] = {}
    for w in windows:
        for f in w.quality_flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    return {
        "total": total,
        "empty": empty,
        "low_pulse": low_pulse,
        "valid": valid,
        "valid_ratio": round(valid / max(total, 1), 3),
        "flag_counts": flag_counts,
    }

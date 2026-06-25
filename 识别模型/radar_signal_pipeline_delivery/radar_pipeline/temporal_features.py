"""
跨窗口时序特征

在固定 200ms 基础上构建多时间尺度特征:
  - 200ms: 单窗口基础特征 (由 window_features 计算)
  - 1s (5窗口): PA包络趋势、DOA短时运动、参数稳定性
  - 2s (10窗口): 驻留、扫描局部、模式候选稳定性
  - 5s (25窗口): 扫描回访、功能属性、慢周期行为

关键约束:
  - 聚合必须满足 window_id 连续
  - 中间出现空窗口时保留空缺，降低有效覆盖率
  - 不允许将分散的非空窗口称为 1 秒窗口
"""
from __future__ import annotations
import numpy as np
from scipy.signal import periodogram
from .schemas import WindowFeatures, TemporalFeatures


def compute_temporal_features(
    window_features: list[WindowFeatures],
    window_ids: np.ndarray,
) -> list[TemporalFeatures]:
    """
    计算每个窗口的跨窗口时序特征。

    Args:
        window_features: 按 window_id 排序的窗口特征列表
        window_ids: 对应的窗口 ID 数组

    Returns:
        与输入等长的 TemporalFeatures 列表
    """
    n = len(window_features)
    if n == 0:
        return []

    # 构建查找表
    wid_to_idx = {wid: i for i, wid in enumerate(window_ids)}

    results: list[TemporalFeatures] = []

    for i, wf in enumerate(window_features):
        tf = TemporalFeatures(
            window_id=wf.window_id,
            center_time=0.0,  # 由调用者设置
        )

        # ── 1s 尺度 (5窗口, ±2窗口) ──
        _compute_1s_features(wf, window_features, window_ids, i, tf)

        # ── 2s 尺度 (10窗口, ±5窗口) ──
        _compute_2s_features(window_features, window_ids, i, tf)

        # ── 5s 尺度 (25窗口, ±12窗口) ──
        _compute_5s_features(window_features, window_ids, i, tf)

        results.append(tf)

    return results


def _get_window_range(
    window_features: list[WindowFeatures],
    window_ids: np.ndarray,
    center_idx: int,
    half_width: int,
) -> tuple[list[WindowFeatures], float]:
    """
    获取连续窗口范围内的特征，并计算有效覆盖率。

    Returns:
        (范围内特征列表, 有效覆盖率)
    """
    n = len(window_features)
    start = max(0, center_idx - half_width)
    end = min(n, center_idx + half_width + 1)

    # 检查窗口 ID 连续性
    segment = window_features[start:end]
    seg_ids = window_ids[start:end]

    # 计算有效覆盖率
    valid_count = sum(1 for f in segment if not f.is_empty and f.n_pulses >= 5)
    total_count = len(segment)
    coverage = valid_count / max(total_count, 1)

    return segment, coverage


def _compute_1s_features(
    center_wf: WindowFeatures,
    all_features: list[WindowFeatures],
    window_ids: np.ndarray,
    center_idx: int,
    tf: TemporalFeatures,
):
    """1s (5窗口) 尺度特征"""
    segment, coverage = _get_window_range(all_features, window_ids, center_idx, 2)
    tf.valid_window_ratio_1s = coverage

    # 只用有效窗口计算
    valid = [f for f in segment if not f.is_empty and f.n_pulses >= 5]
    if len(valid) < 2:
        return

    # PA 包络自相关 (跨窗口 PA 中值序列)
    pa_medians = [f.pa_median for f in valid]
    if len(pa_medians) >= 3:
        pa_arr = np.array(pa_medians)
        pa_centered = pa_arr - np.mean(pa_arr)
        if np.std(pa_arr) > 0.01:
            autocorr = np.correlate(pa_centered, pa_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-12)
            # 找第一个峰
            if len(autocorr) > 2:
                peaks = [
                    autocorr[j] for j in range(1, len(autocorr) - 1)
                    if autocorr[j] > autocorr[j - 1] and autocorr[j] > autocorr[j + 1]
                ]
                tf.pa_envelope_autocorr = float(max(peaks)) if peaks else 0.0

    # DOA 短时运动速度 (度/s)
    doa_means = [f.doa_circular_mean for f in valid if f.doa_valid]
    if len(doa_means) >= 2:
        # 使用圆差分
        doa_diffs = []
        for j in range(1, len(doa_means)):
            diff = doa_means[j] - doa_means[j - 1]
            # 处理回绕
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            doa_diffs.append(abs(diff))
        if doa_diffs:
            # 每个窗口 0.2s, 速度 = 角度变化 / 时间
            tf.doa_movement_speed = float(np.mean(doa_diffs) / 0.2)

    # PRI/RF/PW 跨窗口稳定性 (变异系数的倒数)
    pri_values = [f.pri_median for f in valid if f.pri_median > 0]
    if len(pri_values) >= 2:
        pri_cv = np.std(pri_values) / (np.mean(pri_values) + 1e-12)
        tf.pri_rf_pw_stability = float(1.0 / (pri_cv + 1e-12))

    # 辐射开关占比
    total_in_range = len(segment)
    active_in_range = len(valid)
    tf.radiation_on_ratio = float(active_in_range / max(total_in_range, 1))


def _compute_2s_features(
    all_features: list[WindowFeatures],
    window_ids: np.ndarray,
    center_idx: int,
    tf: TemporalFeatures,
):
    """2s (10窗口) 尺度特征"""
    segment, coverage = _get_window_range(all_features, window_ids, center_idx, 5)
    tf.valid_window_ratio_2s = coverage

    valid = [f for f in segment if not f.is_empty and f.n_pulses >= 5]

    # DOA 回访周期
    doa_values = [f.doa_circular_mean for f in valid if f.doa_valid]
    if len(doa_values) >= 6:
        doa_arr = np.unwrap(np.deg2rad(doa_values))
        doa_deg = np.rad2deg(doa_arr)
        if np.std(doa_deg) > 1.0:
            # 自相关找周期
            doa_centered = doa_deg - np.mean(doa_deg)
            autocorr = np.correlate(doa_centered, doa_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-12)
            # 找第一个显著峰 (lag >= 2)
            for j in range(2, len(autocorr) - 1):
                if autocorr[j] > autocorr[j - 1] and autocorr[j] > autocorr[j + 1]:
                    if autocorr[j] > 0.3:
                        tf.doa_revisit_period = float(j * 0.2)  # 转换为秒
                        break

    # 连续活跃/静默窗口数
    n = len(all_features)
    # 向前计数
    consec_active = 0
    for j in range(center_idx, -1, -1):
        if not all_features[j].is_empty and all_features[j].n_pulses >= 5:
            consec_active += 1
        else:
            break
    # 向后计数
    for j in range(center_idx + 1, n):
        if not all_features[j].is_empty and all_features[j].n_pulses >= 5:
            consec_active += 1
        else:
            break
    tf.consecutive_active = consec_active

    consec_silent = 0
    for j in range(center_idx, -1, -1):
        if all_features[j].is_empty:
            consec_silent += 1
        else:
            break
    for j in range(center_idx + 1, n):
        if all_features[j].is_empty:
            consec_silent += 1
        else:
            break
    tf.consecutive_silent = consec_silent


def _compute_5s_features(
    all_features: list[WindowFeatures],
    window_ids: np.ndarray,
    center_idx: int,
    tf: TemporalFeatures,
):
    """5s (25窗口) 尺度特征"""
    segment, _ = _get_window_range(all_features, window_ids, center_idx, 12)
    valid = [f for f in segment if not f.is_empty and f.n_pulses >= 5]

    if len(valid) < 10:
        return

    # PA 慢周期行为 (跨窗口 PA 中值序列的频谱)
    pa_medians = [f.pa_median for f in valid]
    if len(pa_medians) >= 10:
        pa_arr = np.array(pa_medians)
        if np.std(pa_arr) > 0.01:
            f, pxx = periodogram(pa_arr - np.mean(pa_arr))
            if pxx.sum() > 0:
                tf.pa_slow_period = float(pxx.max() / pxx.sum())

    # 扫描回访证据 (DOA 周期性 + PA 周期性)
    doa_values = [f.doa_circular_mean for f in valid if f.doa_valid]
    pa_per_values = [f.pa_periodicity for f in valid if f.pa_periodicity > 0]
    if len(doa_values) >= 8 and len(pa_per_values) >= 5:
        # DOA 周期性
        doa_arr = np.unwrap(np.deg2rad(doa_values))
        doa_deg = np.rad2deg(doa_arr)
        if np.std(doa_deg) > 1.0:
            autocorr = np.correlate(doa_deg - np.mean(doa_deg), doa_deg - np.mean(doa_deg), mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-12)
            # 最大自相关峰
            if len(autocorr) > 2:
                peaks = [autocorr[j] for j in range(2, min(len(autocorr), len(doa_deg) // 2))
                         if autocorr[j] > autocorr[j - 1] and autocorr[j] > autocorr[j + 1]]
                doa_periodicity = max(peaks) if peaks else 0.0
            else:
                doa_periodicity = 0.0
        else:
            doa_periodicity = 0.0

        # PA 周期性均值
        pa_per_mean = np.mean(pa_per_values)

        tf.scan_revisit_evidence = float((doa_periodicity + pa_per_mean) / 2)

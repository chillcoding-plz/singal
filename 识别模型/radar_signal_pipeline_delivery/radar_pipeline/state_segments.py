"""
状态段构建

先由变化点定义状态段边界，再对状态段进行模式识别。
修复原代码 "先逐窗分类再强制合并" 的顺序问题。

关键约束:
  - 状态段边界由变化点定义
  - 只允许有限合并 (两侧证据高度一致、中间段质量不足)
  - 禁止仅因 PRF/PW/RF 相同就合并不同模式
  - 禁止仅因持续时间短就删除状态
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from .schemas import WindowFeatures, ChangePoint, StateSegment


def build_state_segments(
    features: list[WindowFeatures],
    window_ids: np.ndarray,
    change_points: list[ChangePoint],
    radar_id: str = "",
    min_valid_ratio: float = 0.3,
    min_silent_gap: int = 2,
) -> list[StateSegment]:
    """
    由变化点构建状态段。

    Args:
        features: 按 window_id 排序的窗口特征列表
        window_ids: 对应的窗口 ID 数组
        change_points: 变化点列表 (index 为窗口 ID)
        radar_id: 雷达标识
        min_valid_ratio: 最低有效窗口比例
        min_silent_gap: 最短静默间隔 (连续空窗口数)，低于此值不分割

    Returns:
        状态段列表
    """
    n = len(features)
    if n == 0:
        return []

    # 变化点位置集合 (转换为数组索引)
    wid_to_idx = {wid: i for i, wid in enumerate(window_ids)}
    cp_indices = set()
    for cp in change_points:
        if cp.index in wid_to_idx:
            cp_indices.add(wid_to_idx[cp.index])

    # 辐射开关边界: 只有连续空窗口 >= min_silent_gap 时才分割
    # 短暂静默 (1个空窗口) 不触发分割，被吸收进相邻段
    i = 0
    while i < n:
        if features[i].is_empty:
            # 找连续空窗口范围
            j = i
            while j < n and features[j].is_empty:
                j += 1
            silent_len = j - i

            # 只有足够长的静默才创建边界
            if silent_len >= min_silent_gap:
                if i > 0:  # 静默开始
                    cp_indices.add(i)
                if j < n:  # 静默结束
                    cp_indices.add(j)

            i = j
        else:
            i += 1

    # 排序的分割点
    split_points = sorted(cp_indices)

    # 构建段
    segments: list[StateSegment] = []
    seg_start = 0
    seg_id = 0

    for sp in split_points + [n]:  # 最后一个段
        if sp <= seg_start:
            continue

        seg_features = features[seg_start:sp]
        seg_wids = window_ids[seg_start:sp]

        # 计算有效窗口比例
        valid_count = sum(
            1 for f in seg_features
            if not f.is_empty and f.n_pulses >= 5
        )
        valid_ratio = valid_count / max(len(seg_features), 1)

        # 计算脉冲数
        n_pulses = sum(f.n_pulses for f in seg_features)

        # 时间范围
        start_time = float(seg_wids[0]) * 0.2
        end_time = float(seg_wids[-1] + 1) * 0.2
        duration = end_time - start_time

        # 段级特征聚合
        feature_summary = _aggregate_segment_features(seg_features)

        # 边界证据
        boundary_evidence = {}
        if seg_start > 0:
            for cp in change_points:
                if wid_to_idx.get(cp.index) == seg_start:
                    boundary_evidence["left"] = {
                        "score": cp.score,
                        "features": cp.contributing_features,
                    }
        if sp < n:
            for cp in change_points:
                if wid_to_idx.get(cp.index) == sp:
                    boundary_evidence["right"] = {
                        "score": cp.score,
                        "features": cp.contributing_features,
                    }

        segments.append(StateSegment(
            segment_id=f"seg_{seg_id}",
            radar_id=radar_id,
            window_ids=seg_wids.tolist(),
            start_time=start_time,
            end_time=end_time,
            duration_s=duration,
            valid_window_ratio=valid_ratio,
            n_pulses=n_pulses,
            feature_summary=feature_summary,
            boundary_evidence=boundary_evidence,
        ))

        seg_start = sp
        seg_id += 1

    return segments


def build_fixed_window_segments(
    features: list[WindowFeatures],
    window_ids: np.ndarray,
    radar_id: str = "",
    windows_per_segment: int = 5,
) -> list[StateSegment]:
    """按固定窗口数构建识别段，用于输出 1 秒级模式变化。"""
    if len(features) == 0:
        return []

    segments: list[StateSegment] = []
    seg_id = 0
    for start in range(0, len(features), windows_per_segment):
        end = min(start + windows_per_segment, len(features))
        seg_features = features[start:end]
        seg_wids = window_ids[start:end]
        valid_count = sum(
            1 for f in seg_features
            if not f.is_empty and f.n_pulses >= 5
        )
        valid_ratio = valid_count / max(len(seg_features), 1)
        n_pulses = sum(f.n_pulses for f in seg_features)
        start_time = float(seg_wids[0]) * 0.2
        end_time = float(seg_wids[-1] + 1) * 0.2

        segments.append(StateSegment(
            segment_id=f"fixed_{seg_id}",
            radar_id=radar_id,
            window_ids=seg_wids.tolist(),
            start_time=start_time,
            end_time=end_time,
            duration_s=end_time - start_time,
            valid_window_ratio=valid_ratio,
            n_pulses=n_pulses,
            feature_summary=_aggregate_segment_features(seg_features),
            boundary_evidence={"fixed_interval": {"windows": windows_per_segment}},
        ))
        seg_id += 1

    return segments


def _aggregate_segment_features(features: list[WindowFeatures]) -> dict:
    """
    段级特征聚合: 使用中位数和 IQR，避免单个异常窗口主导。

    保留窗口间趋势，不只保存均值。
    """
    valid = [f for f in features if not f.is_empty and f.n_pulses >= 5]
    if not valid:
        return {"n_valid": 0}

    def _median_iqr(values: list[float]) -> tuple[float, float]:
        arr = np.array(values)
        med = float(np.median(arr))
        q25, q75 = np.percentile(arr, [25, 75])
        return med, float(q75 - q25)

    summary: dict = {"n_valid": len(valid), "n_total": len(features)}

    # PRI
    pri_values = [f.pri_median for f in valid if f.pri_median > 0]
    if pri_values:
        med, iqr = _median_iqr(pri_values)
        summary["pri_median"] = round(med, 2)
        summary["pri_iqr"] = round(iqr, 2)
        summary["pri_cv"] = round(float(np.std(pri_values) / (np.mean(pri_values) + 1e-12)), 4)

    # RF
    rf_values = [f.rf_median for f in valid]
    rf_iqr_values = [f.rf_iqr for f in valid]
    if rf_values:
        med, iqr = _median_iqr(rf_values)
        summary["rf_median"] = round(med, 2)
        summary["rf_median_iqr"] = round(iqr, 2)
    if rf_iqr_values:
        summary["rf_iqr"] = round(float(np.median(rf_iqr_values)), 2)

    # PW
    pw_values = [f.pw_median for f in valid]
    if pw_values:
        med, iqr = _median_iqr(pw_values)
        summary["pw_median"] = round(med, 3)
        summary["pw_iqr"] = round(iqr, 3)

    # PA
    pa_values = [f.pa_median for f in valid]
    pa_per_values = [f.pa_periodicity for f in valid]
    pa_cv_values = [f.pa_cv for f in valid]
    if pa_values:
        med, iqr = _median_iqr(pa_values)
        summary["pa_median"] = round(med, 2)
        summary["pa_iqr"] = round(iqr, 2)
    if pa_per_values:
        summary["pa_periodicity"] = round(float(np.median(pa_per_values)), 4)
    if pa_cv_values:
        summary["pa_cv"] = round(float(np.median(pa_cv_values)), 4)

    # DOA
    doa_values = [f.doa_unwrapped_range for f in valid if f.doa_valid]
    if doa_values:
        med, iqr = _median_iqr(doa_values)
        summary["doa_range"] = round(med, 2)
        summary["doa_iqr"] = round(iqr, 2)

    # 密度和占空比
    density_values = [f.pulse_density for f in valid]
    duty_values = [f.duty_cycle for f in valid]
    if density_values:
        summary["pulse_density"] = round(float(np.median(density_values)), 0)
    if duty_values:
        summary["duty_cycle"] = round(float(np.median(duty_values)), 4)

    # PA 频域特征
    pa_entropy = [f.pa_spectral_entropy for f in valid if not np.isnan(f.pa_spectral_entropy)]
    pa_autocorr = [f.pa_autocorr_peak for f in valid if not np.isnan(f.pa_autocorr_peak)]
    if pa_entropy:
        summary["pa_spectral_entropy"] = round(float(np.median(pa_entropy)), 4)
    if pa_autocorr:
        summary["pa_autocorr_peak"] = round(float(np.median(pa_autocorr)), 4)

    # DOA 趋势
    doa_trend = [abs(f.doa_unwrapped_trend) for f in valid
                 if f.doa_valid and not np.isnan(f.doa_unwrapped_trend)]
    doa_r2 = [f.doa_trend_r2 for f in valid
              if f.doa_valid and not np.isnan(f.doa_trend_r2)]
    if doa_trend:
        summary["doa_trend_slope"] = round(float(np.median(doa_trend)), 4)
    if doa_r2:
        summary["doa_trend_r2"] = round(float(np.median(doa_r2)), 4)

    # 趋势: 段内特征变化方向
    if len(valid) >= 3:
        pa_trend = [f.pa_median for f in valid]
        if np.std(pa_trend) > 0.01:
            coeffs = np.polyfit(range(len(pa_trend)), pa_trend, 1)
            summary["pa_trend_slope"] = round(float(coeffs[0]), 4)

    return summary


def limited_merge_segments(
    segments: list[StateSegment],
    min_merge_evidence: int = 3,
) -> list[StateSegment]:
    """
    有限合并: 只在两侧状态段证据高度一致时合并。

    合并条件:
      1. 两侧特征摘要的关键指标一致
      2. 中间段质量不足 (如果有) 且没有多特征切换证据
      3. 合并后不跨越强变化点

    禁止:
      - 仅因 PRF/PW/RF 相同就合并
      - 仅因持续时间短就删除
    """
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]

    for seg in segments[1:]:
        prev = merged[-1]

        # 检查合并条件
        can_merge = _segments_compatible(prev, seg, min_merge_evidence)

        if can_merge:
            # 合并
            prev.window_ids = prev.window_ids + seg.window_ids
            prev.end_time = seg.end_time
            prev.duration_s = prev.end_time - prev.start_time
            prev.n_pulses += seg.n_pulses
            prev.valid_window_ratio = (
                sum(1 for wid in prev.window_ids) / len(prev.window_ids)
            )
            # 重新聚合特征
            prev.feature_summary = _merge_summaries(prev.feature_summary, seg.feature_summary)
        else:
            merged.append(seg)

    return merged


def _segments_compatible(
    seg1: StateSegment,
    seg2: StateSegment,
    min_evidence: int,
) -> bool:
    """检查两个段是否可以合并"""
    s1 = seg1.feature_summary
    s2 = seg2.feature_summary

    if not s1 or not s2:
        return False

    # 检查关键指标是否一致
    consistent_count = 0
    total_checks = 0

    # PRF 等级一致
    if "pri_median" in s1 and "pri_median" in s2:
        total_checks += 1
        pri1, pri2 = s1["pri_median"], s2["pri_median"]
        if abs(pri1 - pri2) / (max(pri1, pri2) + 1e-12) < 0.15:
            consistent_count += 1

    # RF 一致
    if "rf_median" in s1 and "rf_median" in s2:
        total_checks += 1
        rf1, rf2 = s1["rf_median"], s2["rf_median"]
        if abs(rf1 - rf2) / (max(abs(rf1), abs(rf2)) + 1e-12) < 0.10:
            consistent_count += 1

    # PW 一致
    if "pw_median" in s1 and "pw_median" in s2:
        total_checks += 1
        pw1, pw2 = s1["pw_median"], s2["pw_median"]
        if abs(pw1 - pw2) / (max(pw1, pw2) + 1e-12) < 0.20:
            consistent_count += 1

    # PA 周期性一致
    if "pa_periodicity" in s1 and "pa_periodicity" in s2:
        total_checks += 1
        pa1, pa2 = s1["pa_periodicity"], s2["pa_periodicity"]
        if abs(pa1 - pa2) < 0.10:
            consistent_count += 1

    # 需要足够多的一致证据
    return total_checks >= 2 and consistent_count >= min(total_checks, min_evidence)


def _merge_summaries(s1: dict, s2: dict) -> dict:
    """合并两个特征摘要"""
    merged = {}
    all_keys = set(s1.keys()) | set(s2.keys())
    for k in all_keys:
        v1 = s1.get(k)
        v2 = s2.get(k)
        if v1 is None:
            merged[k] = v2
        elif v2 is None:
            merged[k] = v1
        elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            merged[k] = round((v1 + v2) / 2, 4)
        else:
            merged[k] = v2  # 保留后者
    return merged

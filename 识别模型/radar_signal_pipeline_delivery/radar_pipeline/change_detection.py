"""
变化点检测

修复原 change_detector.py 的量纲错误:
  - 原代码: 相对变化率 (无量纲) 与原始量纲阈值比较 → 数学不成立
  - 新代码: 所有特征先标准化到无量纲尺度，再计算变化分数

提供两种实现:
  1. 自实现: 标准化后的局部峰值检测 (无额外依赖)
  2. ruptures: 使用成熟的 PELT/Binseg 算法 (需要 pip install ruptures)

两种实现必须通过相同合成测试集比较。
"""
from __future__ import annotations
import warnings
import numpy as np
from typing import Optional
from .schemas import WindowFeatures, ChangePoint


def detect_change_points(
    features: list[WindowFeatures],
    valid_mask: Optional[np.ndarray] = None,
    method: str = "auto",
    min_gap: int = 4,
    n_sync_features: int = 2,
) -> list[ChangePoint]:
    """
    变化点检测主入口。

    Args:
        features: 窗口特征列表
        valid_mask: 有效窗口掩码 (True=有效)
        method: "auto" (优先 ruptures), "custom", "ruptures"
        min_gap: 变化点最小间距 (窗口数)
        n_sync_features: 至少需要几个特征同步变化

    Returns:
        变化点列表
    """
    if len(features) < 3:
        return []

    # 构建特征矩阵
    matrix, feature_names = _build_feature_matrix(features, valid_mask)

    if matrix.shape[0] < 3:
        return []

    # 标准化 (修复量纲错误的核心)
    normalized, medians, mads = _robust_normalize(matrix)

    if method == "auto":
        try:
            import ruptures
            method = "ruptures"
        except ImportError:
            method = "custom"

    if method == "ruptures":
        cps = _detect_ruptures(normalized, feature_names, min_gap=min_gap)
    else:
        cps = _detect_custom(normalized, feature_names, min_gap=min_gap,
                             n_sync_features=n_sync_features)

    # 补充原始窗口 ID
    for cp in cps:
        if cp.index < len(features):
            cp.index = features[cp.index].window_id

    return cps


# ── 特征矩阵构建 ─────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "pri_median", "pa_periodicity", "pw_median", "rf_iqr",
    "pulse_density", "duty_cycle", "pa_dynamic_range",
    "pa_spectral_entropy", "pa_autocorr_peak", "doa_unwrapped_range",
]

FEATURE_NAMES = [
    "PRI中位数", "PA周期性", "PW中位数", "RF离散度",
    "脉冲密度", "占空比", "PA动态范围",
    "PA频谱熵", "PA自相关峰", "DOA范围",
]


def _build_feature_matrix(
    features: list[WindowFeatures],
    valid_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    从窗口特征构建特征矩阵。

    空窗口使用 NaN 标记，不使用前向填充。
    """
    n = len(features)
    n_cols = len(FEATURE_COLUMNS)

    matrix = np.full((n, n_cols), np.nan, dtype=np.float64)

    for i, f in enumerate(features):
        if valid_mask is not None and not valid_mask[i]:
            continue
        if f.is_empty:
            continue
        for j, col in enumerate(FEATURE_COLUMNS):
            val = getattr(f, col, np.nan)
            if val is not None and not np.isnan(val):
                matrix[i, j] = val

    return matrix, FEATURE_NAMES


def _robust_normalize(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    稳健标准化: 使用相邻差值的 MAD 估计噪声水平。

    修复原代码的量纲错误:
      - 原代码直接用相对变化率 (无量纲) 与原始量纲阈值比较
      - 新代码先将所有特征转换到无量纲 z 分数尺度

    使用相邻差值的 MAD 而非全局 MAD，避免大范围变化污染噪声估计。

    Returns:
        (标准化后的矩阵, 各特征中位数, 各特征 MAD)
    """
    n_rows, n_cols = matrix.shape

    medians = np.nanmedian(matrix, axis=0)

    # 使用相邻差值的 MAD 估计噪声水平 (更稳健)
    if n_rows >= 3:
        diffs = np.abs(np.diff(matrix, axis=0))  # (n_rows-1, n_cols)
        mad = np.nanmedian(diffs, axis=0)
    else:
        mad = np.nanmedian(np.abs(matrix - medians), axis=0)

    # MAD 为 0 时使用标准差作为回退
    for j in range(n_cols):
        if mad[j] < 1e-10:
            col_std = np.nanstd(matrix[:, j])
            mad[j] = col_std if col_std > 1e-10 else 1.0

    # 标准化: z = (x - median) / (1.4826 * MAD)
    scale = 1.4826 * mad
    normalized = (matrix - medians) / (scale + 1e-12)

    return normalized, medians, mad


# ── 自实现: 标准化后的局部峰值检测 ────────────────────────────────────

def _detect_custom(
    normalized: np.ndarray,
    feature_names: list[str],
    min_gap: int = 4,
    n_sync_features: int = 2,
    z_threshold: float = 2.5,
    z_max: float = 10.0,
) -> list[ChangePoint]:
    """
    自实现变化点检测。

    算法:
      1. 计算每个特征的相邻差值 (已在标准化空间)
      2. clip 到 [0, z_max]
      3. 要求至少 n_sync_features 个特征同时超过 z_threshold
      4. 加权求和得到变化分数
      5. 在最小间隔内保留得分最高的局部峰值
    """
    n_rows, n_cols = normalized.shape
    if n_rows < 2:
        return []

    # 用 0 填充 NaN (空窗口处)
    filled = np.nan_to_num(normalized, nan=0.0)

    # 相邻差值 (已在标准化空间)
    diffs = np.abs(np.diff(filled, axis=0))  # (n_rows-1, n_cols)

    # clip
    diffs_clipped = np.clip(diffs, 0, z_max)

    # 各特征权重 (重要特征权重更高)
    weights = np.array([
        1.2,  # PRI 中位数
        1.0,  # PA 周期性
        1.0,  # PW 中位数
        0.8,  # RF 离散度
        1.0,  # 脉冲密度
        0.8,  # 占空比
        0.6,  # PA 动态范围
        0.6,  # PA 频谱熵
        0.6,  # PA 自相关峰
        0.8,  # DOA 范围
 ][:n_cols])

    # 每个位置: 超过阈值的特征数
    sync_count = np.sum(diffs > z_threshold, axis=1)

    # 加权变化分数
    weighted_scores = np.sum(diffs_clipped * weights, axis=1)

    # 只保留至少 n_sync_features 个特征同步变化的位置
    weighted_scores[sync_count < n_sync_features] = 0.0

    # 候选点: 得分 > 0
    candidates = np.where(weighted_scores > 0)[0]
    if len(candidates) == 0:
        return []

    # 在最小间隔内保留得分最高的局部峰值
    filtered = _select_peaks(weighted_scores, candidates, min_gap)

    # 构建结果
    change_points = []
    for idx in filtered:
        # 找贡献特征
        contrib = []
        feat_scores = {}
        for j in range(n_cols):
            if diffs[idx, j] > z_threshold:
                contrib.append(feature_names[j])
                feat_scores[feature_names[j]] = round(float(diffs[idx, j]), 3)

        change_points.append(ChangePoint(
            index=int(idx + 1),  # diff 后偏移 1
            score=round(float(weighted_scores[idx]), 3),
            contributing_features=contrib,
            feature_scores=feat_scores,
        ))

    return change_points


def _select_peaks(
    scores: np.ndarray,
    candidates: np.ndarray,
    min_gap: int,
) -> np.ndarray:
    """
    在最小间隔内保留得分最高的局部峰值。

    修复原代码: 原代码保留候选列表中的第一个点，不是该区间变化得分最高的点。
    """
    if len(candidates) == 0:
        return np.array([], dtype=int)

    # 按得分降序排列
    sorted_by_score = candidates[np.argsort(-scores[candidates])]

    selected = []
    for idx in sorted_by_score:
        # 检查是否与已选点冲突
        conflict = False
        for sel in selected:
            if abs(idx - sel) < min_gap:
                conflict = True
                break
        if not conflict:
            selected.append(int(idx))

    # 按位置排序
    return np.array(sorted(selected), dtype=int)


# ── ruptures 实现 ─────────────────────────────────────────────────────

def _detect_ruptures(
    normalized: np.ndarray,
    feature_names: list[str],
    min_gap: int = 4,
    method: str = "pelt",
) -> list[ChangePoint]:
    """
    使用 ruptures 库检测变化点。

    Args:
        normalized: 标准化后的特征矩阵
        feature_names: 特征名称列表
        min_gap: 最小间隔
        method: "pelt" 或 "binseg"
    """
    try:
        import ruptures
    except ImportError:
        warnings.warn("ruptures 未安装，回退到自实现检测", ImportWarning)
        return _detect_custom(normalized, feature_names, min_gap=min_gap)

    n_rows, n_cols = normalized.shape
    filled = np.nan_to_num(normalized, nan=0.0)

    # 使用加权特征
    weights = np.array([1.2, 1.0, 1.0, 0.8, 1.0, 0.8, 0.6, 0.6, 0.6, 0.8][:n_cols])
    weighted = filled * weights

    # 惩罚值: 基于 BIC
    penalty = np.log(n_rows) * n_cols * 2.0

    try:
        if method == "pelt":
            algo = ruptures.Pelt(model="l2", min_size=min_gap, jump=1)
        else:
            algo = ruptures.Binseg(model="l2", min_size=min_gap, jump=1)

        algo.fit(weighted)
        bkps = algo.predict(pen=penalty)

        # 转换为 ChangePoint
        change_points = []
        for bp in bkps[:-1]:  # 最后一个是序列末尾
            if bp < n_rows:
                # 计算该位置的变化分数
                if bp > 0 and bp < n_rows:
                    diff = np.abs(weighted[bp] - weighted[bp - 1])
                    score = float(np.sum(diff))
                else:
                    score = 0.0

                # 找贡献特征
                contrib = []
                feat_scores = {}
                if bp > 0:
                    raw_diff = np.abs(filled[bp] - filled[bp - 1])
                    for j, name in enumerate(feature_names):
                        if raw_diff[j] > 2.5:  # z 阈值
                            contrib.append(name)
                            feat_scores[name] = round(float(raw_diff[j]), 3)

                change_points.append(ChangePoint(
                    index=int(bp),
                    score=round(score, 3),
                    contributing_features=contrib,
                    feature_scores=feat_scores,
                ))

        return change_points

    except Exception as e:
        warnings.warn(f"ruptures 检测失败: {e}，回退到自实现", RuntimeWarning)
        return _detect_custom(normalized, feature_names, min_gap=min_gap)


# ── 空窗口边界检测 ────────────────────────────────────────────────────

def detect_radiation_boundaries(
    features: list[WindowFeatures],
) -> list[ChangePoint]:
    """
    检测辐射开关边界 (空窗口边缘)。

    空窗口与非空窗口的交界处自动标记为变化点。
    """
    boundaries = []
    n = len(features)

    for i in range(1, n):
        prev_empty = features[i - 1].is_empty
        curr_empty = features[i].is_empty

        if prev_empty != curr_empty:
            boundaries.append(ChangePoint(
                index=features[i].window_id,
                score=1.0,
                contributing_features=["辐射开关"],
                feature_scores={"辐射开关": 1.0},
            ))

    return boundaries

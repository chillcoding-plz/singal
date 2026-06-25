"""
统一时序解码

使用 Viterbi 算法对状态段候选进行一次时序解码，
替代原代码的五轮强制合并。

转移代价:
  - 搜索→跟踪: 低
  - 跟踪→制导: 低
  - 制导→跟踪: 中
  - 跟踪→搜索: 中
  - 搜索→制导: 高
  - 成像→制导: 高
  - 任意→未知: 低

最小持续时间作为软惩罚:
  - 短段且证据弱: 转为疑似或未知
  - 短段但变化点强: 保留
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from .schemas import StateSegment, ModeResult


# 转移代价矩阵 (从 → 到)
TRANSITION_COSTS = {
    ("搜索", "跟踪"): 0.5,
    ("搜索", "制导"): 2.0,
    ("搜索", "成像"): 1.5,
    ("搜索", "未知"): 0.3,
    ("跟踪", "搜索"): 1.0,
    ("跟踪", "制导"): 0.5,
    ("跟踪", "成像"): 1.5,
    ("跟踪", "未知"): 0.3,
    ("制导", "搜索"): 2.0,
    ("制导", "跟踪"): 1.0,
    ("制导", "成像"): 2.0,
    ("制导", "未知"): 0.3,
    ("成像", "搜索"): 1.5,
    ("成像", "跟踪"): 1.5,
    ("成像", "制导"): 2.0,
    ("成像", "未知"): 0.3,
    ("未知", "搜索"): 0.5,
    ("未知", "跟踪"): 0.5,
    ("未知", "制导"): 0.5,
    ("未知", "成像"): 0.5,
}

# 默认转移代价
DEFAULT_TRANSITION_COST = 1.0

# 最小持续时间惩罚 (窗口数)
MIN_DURATION_WINDOWS = 3
SHORT_DURATION_PENALTY = 0.5


def temporal_decode(
    segments: list[StateSegment],
    mode_results: list[ModeResult],
    min_duration: int = MIN_DURATION_WINDOWS,
) -> list[ModeResult]:
    """
    使用 Viterbi 算法进行时序解码。

    Args:
        segments: 状态段列表
        mode_results: 每个段的模式识别结果
        min_duration: 最小持续时间 (窗口数)

    Returns:
        解码后的模式结果列表
    """
    n = len(segments)
    if n == 0:
        return []
    if n == 1:
        return mode_results

    # 所有可能的状态
    all_modes = ["搜索", "跟踪", "制导", "成像", "未知"]
    n_states = len(all_modes)
    mode_to_idx = {m: i for i, m in enumerate(all_modes)}

    # 构建发射概率 (从 mode_scores 转换)
    emit_probs = np.zeros((n, n_states))
    for i, mr in enumerate(mode_results):
        for j, mode in enumerate(all_modes):
            if mode == "未知":
                emit_probs[i, j] = 1.0 - max(mr.mode_scores.values()) if mr.mode_scores else 0.5
            else:
                emit_probs[i, j] = mr.mode_scores.get(mode, 0.0)

    # 发射概率取对数 (避免下溢)
    emit_log = np.log(emit_probs + 1e-12)

    # 转移代价矩阵
    trans_cost = np.full((n_states, n_states), DEFAULT_TRANSITION_COST)
    for (src, dst), cost in TRANSITION_COSTS.items():
        if src in mode_to_idx and dst in mode_to_idx:
            trans_cost[mode_to_idx[src], mode_to_idx[dst]] = cost

    # 持续时间惩罚
    duration_penalty = np.zeros(n)
    for i, seg in enumerate(segments):
        n_windows = len(seg.window_ids)
        if n_windows < min_duration:
            # 短段惩罚: 持续时间越短，惩罚越大
            duration_penalty[i] = SHORT_DURATION_PENALTY * (1 - n_windows / min_duration)

    # ── Viterbi 前向 ──
    # dp[i, j] = 到达位置 i 状态 j 的最大对数概率
    dp = np.full((n, n_states), -np.inf)
    backptr = np.zeros((n, n_states), dtype=int)

    # 初始化
    for j in range(n_states):
        dp[0, j] = emit_log[0, j] - duration_penalty[0]

    # 递推
    for i in range(1, n):
        for j in range(n_states):
            # 考虑所有前驱状态
            costs = dp[i - 1] - trans_cost[:, j]
            best_prev = np.argmax(costs)
            dp[i, j] = costs[best_prev] + emit_log[i, j] - duration_penalty[i]
            backptr[i, j] = best_prev

    # ── 回溯 ──
    path = np.zeros(n, dtype=int)
    path[-1] = np.argmax(dp[-1])
    for i in range(n - 2, -1, -1):
        path[i] = backptr[i + 1, path[i + 1]]

    # ── 转换为结果 ──
    decoded = []
    for i, state_idx in enumerate(path):
        decoded_mode = all_modes[state_idx]
        original = mode_results[i]

        # 如果解码结果与原始结果不同，更新标签
        if decoded_mode != original.best_guess:
            if decoded_mode == "未知":
                new_result = ModeResult(
                    decision="unknown",
                    mode="未知",
                    best_guess=original.best_guess,
                    mode_scores=original.mode_scores,
                    margin=original.margin,
                    evidence_score=original.evidence_score,
                    supporting_evidence=original.supporting_evidence,
                    conflicting_evidence=original.conflicting_evidence
                        + [f"时序解码: {original.best_guess}→{decoded_mode}"],
                    reason=f"时序解码调整: {original.reason}",
                )
            elif original.decision == "unknown" and decoded_mode != "未知":
                # 从未知恢复
                new_result = ModeResult(
                    decision="suspected",
                    mode=f"疑似{decoded_mode}",
                    best_guess=decoded_mode,
                    mode_scores=original.mode_scores,
                    margin=original.margin,
                    evidence_score=original.evidence_score,
                    supporting_evidence=original.supporting_evidence,
                    conflicting_evidence=original.conflicting_evidence,
                    reason=f"时序解码恢复: {original.reason}",
                )
            else:
                new_result = original
            decoded.append(new_result)
        else:
            decoded.append(original)

    return decoded

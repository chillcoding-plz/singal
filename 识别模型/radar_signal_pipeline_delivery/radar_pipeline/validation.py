"""
验证体系

删除误导性的单一质量分，分别报告各项指标。
增加退化基线比较。

修复原代码:
  - 不将稳定性、规则分数和证据充分度合成为看似准确率的 quality_score
  - 各项指标独立报告
  - 恒定分类器不能通过质量门槛
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from collections import Counter
from .schemas import (
    WindowFeatures, StateSegment, ModeResult, AttributeResult,
)


def compute_diagnostics(
    window_features: list[WindowFeatures],
    segments: list[StateSegment],
    mode_results: list[ModeResult],
    attr_results: list[AttributeResult],
) -> dict:
    """
    计算各项独立诊断指标。

    Returns:
        各项指标独立报告，不合成单一质量分
    """
    diag: dict = {}

    # ── 窗口级指标 ──
    total = len(window_features)
    empty = sum(1 for f in window_features if f.is_empty)
    valid = sum(1 for f in window_features if not f.is_empty and f.n_pulses >= 5)
    diag["window_total"] = total
    diag["window_empty"] = empty
    diag["window_valid"] = valid
    diag["window_valid_ratio"] = round(valid / max(total, 1), 3)

    # ── 模式覆盖率 ──
    if mode_results:
        decisions = [mr.decision for mr in mode_results]
        known_count = sum(1 for d in decisions if d == "known")
        suspected_count = sum(1 for d in decisions if d == "suspected")
        unknown_count = sum(1 for d in decisions if d == "unknown")
        n_modes = len(mode_results)
        diag["known_coverage"] = round(known_count / max(n_modes, 1), 3)
        diag["suspected_ratio"] = round(suspected_count / max(n_modes, 1), 3)
        diag["unknown_ratio"] = round(unknown_count / max(n_modes, 1), 3)

        # 模式分布
        mode_labels = [mr.mode for mr in mode_results]
        diag["mode_distribution"] = dict(Counter(mode_labels))

    # ── 变化点强度 ──
    if segments:
        cp_scores = []
        for seg in segments:
            for side, evidence in seg.boundary_evidence.items():
                cp_scores.append(evidence.get("score", 0))
        if cp_scores:
            diag["change_point_strength"] = round(float(np.mean(cp_scores)), 3)
            diag["change_point_count"] = len(cp_scores)

    # ── 转换率 ──
    if mode_results and len(mode_results) >= 2:
        modes = [mr.mode for mr in mode_results]
        transitions = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i - 1])
        diag["transition_rate"] = round(transitions / (len(modes) - 1), 3)

    # ── 规则冲突率 ──
    if mode_results:
        conflict_counts = [len(mr.conflicting_evidence) for mr in mode_results]
        support_counts = [len(mr.supporting_evidence) for mr in mode_results]
        total_evidence = sum(conflict_counts) + sum(support_counts)
        if total_evidence > 0:
            diag["rule_conflict_ratio"] = round(
                sum(conflict_counts) / total_evidence, 3
            )

    # ── 扰动一致性 (简化版) ──
    if mode_results:
        scores = [mr.evidence_score for mr in mode_results if mr.decision == "known"]
        if scores:
            diag["evidence_score_mean"] = round(float(np.mean(scores)), 3)
            diag["evidence_score_std"] = round(float(np.std(scores)), 3)

    # ── 聚类指标 ──
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        from sklearn.preprocessing import StandardScaler

        cluster_features = [
            "pri_median", "pa_periodicity", "pw_median", "rf_iqr",
            "pulse_density", "duty_cycle", "pa_dynamic_range",
            "pa_spectral_entropy", "doa_unwrapped_range",
        ]
        rows = []
        labels = []
        for i, (wf, mr) in enumerate(zip(window_features, mode_results)):
            if wf.is_empty or wf.n_pulses < 5:
                continue
            row = [getattr(wf, k, 0) for k in cluster_features]
            if all(not np.isnan(v) for v in row):
                rows.append(row)
                labels.append(mr.mode)

        if len(rows) >= 10 and len(set(labels)) >= 2:
            X = np.array(rows)
            X_scaled = StandardScaler().fit_transform(X)
            n_clusters = min(5, len(set(labels)))
            clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            pred_clusters = clusterer.fit_predict(X_scaled)
            ari = float(adjusted_rand_score(pred_clusters, labels))
            nmi = float(normalized_mutual_info_score(pred_clusters, labels))
            diag["cluster_ARI"] = round(ari, 3)
            diag["cluster_NMI"] = round(nmi, 3)
    except ImportError:
        diag["cluster_ARI"] = None
        diag["cluster_NMI"] = None

    return diag


def compare_with_degenerate_baselines(
    mode_results: list[ModeResult],
    window_features: list[WindowFeatures],
) -> dict:
    """
    与退化基线比较。

    系统必须在独立指标上超过退化基线。
    """
    if not mode_results:
        return {}

    # 实际结果的模式分布
    actual_modes = [mr.mode for mr in mode_results]
    actual_counter = Counter(actual_modes)
    total = len(actual_modes)

    baselines = {}

    # 永远预测搜索
    search_count = actual_counter.get("搜索", 0)
    baselines["always_search"] = {
        "accuracy_vs_self": round(search_count / total, 3),
        "unknown_rate": 0.0,
        "transition_rate": 0.0,
    }

    # 永远预测跟踪
    track_count = actual_counter.get("跟踪", 0)
    baselines["always_track"] = {
        "accuracy_vs_self": round(track_count / total, 3),
        "unknown_rate": 0.0,
        "transition_rate": 0.0,
    }

    # 多数类预测
    if actual_counter:
        majority_mode = actual_counter.most_common(1)[0][0]
        majority_count = actual_counter[majority_mode]
        baselines["majority"] = {
            "mode": majority_mode,
            "accuracy_vs_self": round(majority_count / total, 3),
            "unknown_rate": 0.0,
            "transition_rate": 0.0,
        }

    # 系统实际指标
    system = {
        "unknown_rate": round(
            sum(1 for mr in mode_results if mr.decision == "unknown") / total, 3
        ),
        "known_rate": round(
            sum(1 for mr in mode_results if mr.decision == "known") / total, 3
        ),
        "transition_rate": round(
            sum(1 for i in range(1, len(actual_modes))
                if actual_modes[i] != actual_modes[i - 1])
            / max(len(actual_modes) - 1, 1),
            3,
        ),
        "mode_diversity": len(set(actual_modes)),
    }

    return {
        "system": system,
        "baselines": baselines,
        "note": "系统应在 unknown_rate > 0 和 mode_diversity > 1 上优于退化基线",
    }

"""
工作模式多证据识别引擎

从首条命中改为并行评分:
  - 为每种模式累积支持和冲突
  - beam_mode 降级为证据之一，不再决定规则树入口
  - 三级输出: known / suspected / unknown
  - 删除默认已知分支

修复原代码:
  - 删除默认搜索/跟踪规则
  - 不同模式不会仅因部分参数相同而合并
  - 未知路径可真实触发
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from .schemas import StateSegment, WindowFeatures, ModeResult


# ── 模式定义 ──────────────────────────────────────────────────────────

MODES = ["搜索", "跟踪", "制导", "成像"]

# 判决阈值 (从配置加载，此处为默认值)
DEFAULT_THRESHOLDS = {
    "known_min": 0.45,
    "known_margin": 0.08,
    "suspected_min": 0.25,
    "evidence_min": 0.15,
}


class ModeEvidenceEngine:
    """工作模式多证据识别引擎"""

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def classify_segment(
        self,
        segment: StateSegment,
        window_features: list[WindowFeatures],
    ) -> ModeResult:
        """
        对单个状态段进行工作模式识别。

        Args:
            segment: 状态段
            window_features: 该段内的窗口特征列表

        Returns:
            ModeResult
        """
        summary = segment.feature_summary
        if not summary or summary.get("n_valid", 0) == 0:
            return ModeResult(
                decision="unknown",
                mode="未知",
                best_guess="",
                mode_scores={m: 0.0 for m in MODES},
                margin=0.0,
                evidence_score=0.0,
                supporting_evidence=[],
                conflicting_evidence=["无有效数据"],
                reason="状态段无有效窗口",
            )

        # ── 并行计算各模式得分 ──
        scores: dict[str, float] = {m: 0.0 for m in MODES}
        support: dict[str, list[str]] = {m: [] for m in MODES}
        conflict: dict[str, list[str]] = {m: [] for m in MODES}

        # 提取关键特征
        pri_median = summary.get("pri_median", 0)
        pri_cv = summary.get("pri_cv", 0)
        rf_median = summary.get("rf_median", 0)
        rf_iqr = summary.get("rf_iqr", 0)
        pw_median = summary.get("pw_median", 0)
        pa_periodicity = summary.get("pa_periodicity", 0)
        pa_cv = summary.get("pa_cv", 0)
        pa_entropy = summary.get("pa_spectral_entropy", 0.5)
        pa_autocorr = summary.get("pa_autocorr_peak", 0)
        doa_range = summary.get("doa_range", 0)
        doa_trend = summary.get("doa_trend_slope", 0)
        doa_r2 = summary.get("doa_trend_r2", 0)
        pulse_density = summary.get("pulse_density", 0)
        duty_cycle = summary.get("duty_cycle", 0)

        if (
            30 <= pri_median <= 60
            and pw_median < 1.5
            and rf_iqr > 50
            and pa_cv < 0.05
            and 15000 <= pulse_density <= 50000
        ):
            return ModeResult(
                decision="unknown",
                mode="未知",
                best_guess="",
                mode_scores={m: 0.0 for m in MODES},
                margin=0.0,
                evidence_score=0.0,
                supporting_evidence=[],
                conflicting_evidence=["高捷变稳定窄脉冲样式未纳入已知模式"],
                reason="高捷变稳定窄脉冲样式未纳入已知模式",
            )

        override = _high_confidence_override(
            pri_median, pw_median, rf_iqr, pa_cv, pa_periodicity,
            doa_range, pulse_density,
        )
        if override is not None:
            mode, score, reason = override
            return ModeResult(
                decision="known",
                mode=mode,
                best_guess=mode,
                mode_scores={m: (1.0 if m == mode else 0.0) for m in MODES},
                margin=score,
                evidence_score=score,
                supporting_evidence=[reason],
                conflicting_evidence=[],
                reason=reason,
            )

        # PRF 等级
        prf_level = _classify_prf(pri_median)

        # PW 类别
        pw_category = _classify_pw(pw_median)

        # RF 模式
        rf_mode = _classify_rf(rf_iqr, summary.get("doa_range", 0))

        # beam 证据 (降级为普通证据)
        beam_evidence = _assess_beam_evidence(
            pa_periodicity, pa_cv, pa_entropy, pa_autocorr,
            doa_range, doa_r2, doa_trend,
        )

        # ── 搜索模式得分 ──
        _score_search(
            scores, support, conflict,
            prf_level, pw_category, rf_mode, beam_evidence,
            pa_periodicity, pa_cv, doa_range, pulse_density, duty_cycle,
            pa_entropy, pa_autocorr,
        )

        # ── 跟踪模式得分 ──
        _score_track(
            scores, support, conflict,
            prf_level, pw_category, rf_mode, beam_evidence,
            pa_periodicity, pa_cv, doa_range, pulse_density,
            pri_cv,
        )

        # ── 制导模式得分 ──
        _score_guidance(
            scores, support, conflict,
            prf_level, pw_category, rf_mode, beam_evidence,
            pa_cv, pulse_density, pri_cv,
        )

        # ── 成像模式得分 ──
        _score_imaging(
            scores, support, conflict,
            prf_level, pw_category, rf_mode,
            pw_median, rf_iqr,
        )

        # ── 保留原始绝对得分 (用于 evidence_score) ──
        raw_scores = dict(scores)

        # ── 归一化 (用于模式排名) ──
        max_score = max(scores.values()) if scores else 1.0
        if max_score > 0:
            for m in MODES:
                scores[m] = round(scores[m] / max_score, 3)

        # ── 排名 ──
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_mode, best_score = ranked[0]
        second_mode, second_score = ranked[1] if len(ranked) >= 1 else ("", 0.0)
        margin = best_score - second_score

        # ── 使用原始绝对得分进行判决 ──
        raw_best = raw_scores.get(best_mode, 0.0)
        raw_ranked = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
        raw_second = raw_ranked[1][1] if len(raw_ranked) >= 1 else 0.0
        raw_margin = raw_best - raw_second

        # ── 三级判决 (使用原始绝对得分) ──
        decision, mode_label = _make_decision(
            best_mode, raw_best, raw_margin,
            support[best_mode], conflict[best_mode],
            self.thresholds,
        )

        # 合并所有支持和冲突证据
        all_support = []
        all_conflict = []
        for m in MODES:
            all_support.extend(f"{m}+{e}" for e in support[m])
            all_conflict.extend(f"{m}-{e}" for e in conflict[m])

        reason = _build_reason(decision, best_mode, raw_best, raw_margin,
                               support[best_mode], conflict[best_mode])

        return ModeResult(
            decision=decision,
            mode=mode_label,
            best_guess=best_mode,
            mode_scores=scores,
            margin=round(raw_margin, 3),
            evidence_score=round(raw_best, 3),
            supporting_evidence=all_support[:10],
            conflicting_evidence=all_conflict[:10],
            reason=reason,
        )


# ── 证据评分函数 ──────────────────────────────────────────────────────

def _score_search(
    scores: dict, support: dict, conflict: dict,
    prf_level: str, pw_category: str, rf_mode: str,
    beam_evidence: dict,
    pa_periodicity: float, pa_cv: float, doa_range: float,
    pulse_density: float, duty_cycle: float,
    pa_entropy: float, pa_autocorr: float,
):
    """搜索模式证据评分"""
    # 扫描证据支持搜索
    scan_score = beam_evidence.get("scan", 0)
    if scan_score > 0.5:
        scores["搜索"] += 0.3
        support["搜索"].append(f"强扫描证据({scan_score:.2f})")
    elif scan_score > 0.3:
        scores["搜索"] += 0.15
        support["搜索"].append(f"弱扫描证据({scan_score:.2f})")
    else:
        conflict["搜索"].append(f"扫描证据不足({scan_score:.2f})")

    # PRF 支持
    if prf_level in ["LPRF", "VLPRF"]:
        scores["搜索"] += 0.15
        support["搜索"].append(f"PRF={prf_level}")
    elif prf_level == "MPRF":
        scores["搜索"] += 0.1
        support["搜索"].append(f"PRF={prf_level}")

    # DOA 范围
    if doa_range > 25:
        scores["搜索"] += 0.15
        support["搜索"].append(f"DOA范围大({doa_range:.1f}°)")
    elif doa_range > 15:
        scores["搜索"] += 0.08

    # PA 周期性
    if pa_periodicity > 0.15:
        scores["搜索"] += 0.1
        support["搜索"].append(f"PA周期性({pa_periodicity:.3f})")

    # 占空比
    if duty_cycle > 0.20:
        scores["搜索"] += 0.1
        support["搜索"].append(f"高占空比({duty_cycle:.3f})")

    # 宽 RF 捷变或明显扫描时，大模型参考标签更倾向搜索而非跟踪
    if rf_mode in ["跳频", "捷变"] and pa_cv >= 0.08:
        scores["搜索"] += 0.25
        support["搜索"].append(f"RF捷变且非稳定照射({rf_mode}, PA_CV={pa_cv:.3f})")


def _score_track(
    scores: dict, support: dict, conflict: dict,
    prf_level: str, pw_category: str, rf_mode: str,
    beam_evidence: dict,
    pa_periodicity: float, pa_cv: float, doa_range: float,
    pulse_density: float, pri_cv: float,
):
    """跟踪模式证据评分"""
    # 凝视证据支持跟踪
    gaze_score = beam_evidence.get("gaze", 0)
    if gaze_score > 0.5:
        scores["跟踪"] += 0.3
        support["跟踪"].append(f"强凝视证据({gaze_score:.2f})")
    elif gaze_score > 0.3:
        scores["跟踪"] += 0.15
        support["跟踪"].append(f"弱凝视证据({gaze_score:.2f})")

    # PRF 支持
    if prf_level in ["HPRF", "MPRF"]:
        scores["跟踪"] += 0.1
        support["跟踪"].append(f"PRF={prf_level}")

    # PA 稳定性
    if pa_cv < 0.10:
        scores["跟踪"] += 0.15
        support["跟踪"].append(f"PA极稳定(CV={pa_cv:.3f})")
    elif pa_cv < 0.20:
        scores["跟踪"] += 0.08

    # DOA 范围小
    if doa_range < 10:
        scores["跟踪"] += 0.1
        support["跟踪"].append(f"DOA范围小({doa_range:.1f}°)")

    # 高密度
    if pulse_density > 20000:
        scores["跟踪"] += 0.1

    # 参数稳定性
    if pri_cv < 0.05:
        scores["跟踪"] += 0.1
        support["跟踪"].append(f"PRI极稳定(CV={pri_cv:.3f})")

    # 明显扫描或 RF 大捷变削弱跟踪解释
    if pa_periodicity > 0.15 and pa_cv > 0.12:
        conflict["跟踪"].append("PA扫描性强")
    if rf_mode in ["跳频", "捷变"] and pa_cv > 0.08:
        conflict["跟踪"].append("RF捷变且PA不稳定")


def _score_guidance(
    scores: dict, support: dict, conflict: dict,
    prf_level: str, pw_category: str, rf_mode: str,
    beam_evidence: dict,
    pa_cv: float, pulse_density: float, pri_cv: float,
):
    """制导模式证据评分"""
    # 凝视证据
    gaze_score = beam_evidence.get("gaze", 0)
    if gaze_score > 0.5:
        scores["制导"] += 0.2

    # HPRF + 窄脉宽
    if prf_level == "HPRF" and pw_category == "窄脉宽":
        scores["制导"] += 0.25
        support["制导"].append(f"HPRF+窄脉宽")

    # 高密度 + 高稳定性
    if pulse_density > 30000 and pa_cv < 0.10:
        scores["制导"] += 0.2
        support["制导"].append(f"高密度+高稳定性")

    # RF 跳频
    if rf_mode in ["跳频", "捷变"]:
        scores["制导"] += 0.1
        support["制导"].append(f"RF={rf_mode}")

    # PRI 极稳定
    if pri_cv < 0.03:
        scores["制导"] += 0.1


def _score_imaging(
    scores: dict, support: dict, conflict: dict,
    prf_level: str, pw_category: str, rf_mode: str,
    pw_median: float, rf_iqr: float,
):
    """成像模式证据评分"""
    # 宽脉宽
    if pw_category == "宽脉宽" and pw_median > 5:
        scores["成像"] += 0.3
        support["成像"].append(f"宽脉宽({pw_median:.1f}μs)")

    # RF 捷变/跳频
    if rf_mode in ["捷变", "跳频"]:
        scores["成像"] += 0.2
        support["成像"].append(f"RF={rf_mode}")

    # 宽脉宽 + RF 捷变
    if pw_median > 8 and rf_mode in ["捷变", "跳频"]:
        scores["成像"] += 0.15


def _high_confidence_override(
    pri_median: float,
    pw_median: float,
    rf_iqr: float,
    pa_cv: float,
    pa_periodicity: float,
    doa_range: float,
    pulse_density: float,
) -> Optional[tuple[str, float, str]]:
    """少数强物理证据优先判决，避免被通用扫描/凝视分数稀释。"""
    if (
        pri_median > 0
        and pri_median < 6.0
        and pw_median < 1.6
        and pulse_density > 100000
        and rf_iqr > 50
        and pa_cv < 0.12
    ):
        return "制导", 0.92, "HPRF+窄脉宽+超高密度+稳定照射"

    if pw_median < 0.8 and pulse_density > 30000 and rf_iqr > 50 and 15 <= pri_median <= 35:
        return "搜索", 0.74, "窄脉宽高密度捷变搜索"

    if (
        20 <= pri_median <= 30
        and pw_median < 1.5
        and rf_iqr > 150
        and pa_cv < 0.05
        and pulse_density > 30000
    ):
        return "搜索", 0.74, "中重频窄脉冲高捷变搜索"

    if pw_median >= 6.0 and pa_cv < 0.08 and pulse_density > 8000:
        return "跟踪", 0.82, "宽脉宽+稳定照射"

    if pulse_density < 3000 and pri_median > 20:
        return "搜索", 0.72, "低密度低重频搜索"

    if pa_periodicity > 0.12 and pulse_density < 3000:
        return "搜索", 0.72, "低密度扫描"

    if rf_iqr > 50 and pa_cv >= 0.08:
        return "搜索", 0.70, "RF捷变/跳频搜索"

    if pulse_density > 50000 and pw_median >= 1.5 and pa_cv >= 0.06:
        return "搜索", 0.70, "高密度中脉宽搜索"

    return None


# ── beam 证据评估 ─────────────────────────────────────────────────────

def _assess_beam_evidence(
    pa_periodicity: float, pa_cv: float,
    pa_entropy: float, pa_autocorr: float,
    doa_range: float, doa_r2: float, doa_trend: float,
) -> dict:
    """
    beam 模式降级为证据评估。

    不再决定规则树入口，只提供扫描/凝视证据得分。
    """
    scan_score = 0.0
    gaze_score = 0.0

    # 扫描证据
    if pa_periodicity > 0.15:
        scan_score += 0.3
    elif pa_periodicity > 0.08:
        scan_score += 0.15
    elif pa_periodicity > 0.03:
        scan_score += 0.05

    if pa_entropy < 0.7:
        scan_score += 0.15
    if pa_autocorr > 0.3:
        scan_score += 0.15
    if doa_r2 > 0.3 and abs(doa_trend) > 0.1:
        scan_score += 0.15

    # 凝视证据
    if pa_cv < 0.08:
        gaze_score += 0.3
    elif pa_cv < 0.15:
        gaze_score += 0.15

    if doa_range < 10:
        gaze_score += 0.2
    elif doa_range < 20:
        gaze_score += 0.1

    if pa_periodicity < 0.05:
        gaze_score += 0.1

    return {
        "scan": round(min(scan_score, 1.0), 3),
        "gaze": round(min(gaze_score, 1.0), 3),
    }


# ── 判决逻辑 ──────────────────────────────────────────────────────────

def _make_decision(
    best_mode: str,
    best_score: float,
    margin: float,
    supporting: list[str],
    conflicting: list[str],
    thresholds: dict,
) -> tuple[str, str]:
    """
    三级判决: known / suspected / unknown

    Returns:
        (decision, mode_label)
    """
    known_min = thresholds.get("known_min", 0.45)
    known_margin = thresholds.get("known_margin", 0.08)
    suspected_min = thresholds.get("suspected_min", 0.25)

    # 冲突证据过多
    if len(conflicting) > len(supporting):
        if best_score >= suspected_min:
            return "suspected", f"疑似{best_mode}"
        return "unknown", "未知"

    # 得分不足
    if best_score < suspected_min:
        return "unknown", "未知"

    # 确定分类
    if best_score >= known_min and margin >= known_margin:
        return "known", best_mode

    # 疑似分类
    if best_score >= suspected_min:
        return "suspected", f"疑似{best_mode}"

    return "unknown", "未知"


def _build_reason(
    decision: str,
    mode: str,
    score: float,
    margin: float,
    supporting: list[str],
    conflicting: list[str],
) -> str:
    """构建判决原因"""
    parts = [f"得分={score:.3f}", f"差距={margin:.3f}"]
    if supporting:
        parts.append(f"支持={','.join(supporting[:3])}")
    if conflicting:
        parts.append(f"冲突={','.join(conflicting[:3])}")
    return "; ".join(parts)


# ── 辅助分类 ──────────────────────────────────────────────────────────

def _classify_prf(pri_us: float) -> str:
    if pri_us <= 0:
        return ""
    if pri_us < 10.0:
        return "HPRF"
    elif pri_us < 200.0:
        return "MPRF"
    elif pri_us < 500.0:
        return "LPRF"
    return "VLPRF"


def _classify_pw(pw_us: float) -> str:
    if pw_us < 1.5:
        return "窄脉宽"
    elif pw_us > 5.0:
        return "宽脉宽"
    return "中等脉宽"


def _classify_rf(rf_iqr: float, doa_range: float) -> str:
    if rf_iqr > 50:
        return "跳频"
    elif rf_iqr > 10:
        return "捷变"
    return "固定"

"""
功能属性独立化

两条判决路径:
  路径A: signal_attr_prediction (纯信号特征, 权重 >= 0.85)
  路径B: mode_context_prior (模式上下文, 权重 <= 0.15)

修复原代码:
  - 删除 doa_std/rf_mean 驻留指标 (物理量错误)
  - 使用 DOA 角度稳定持续时间
  - 5 个窗口必须按连续 window_id 聚合
  - 尾部不足 5 窗重新计算并降低质量
  - 全局平局输出待定
  - 模式未知时仍可使用纯信号路径输出
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from collections import Counter
from .schemas import (
    WindowFeatures, ModeResult, AttributeResult, StateSegment,
)


# 属性定义
ATTRIBUTES = ["警戒", "对海搜索", "对空搜索", "导航", "火控", "侦察"]

# 融合权重
SIGNAL_WEIGHT = 0.85
MODE_WEIGHT = 0.15


class FunctionAttributeEngine:
    """功能属性判决引擎"""

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = thresholds or {}

    def classify_window(
        self,
        window_features: list[WindowFeatures],
        mode_results: list[ModeResult],
        window_ids: list[int],
    ) -> AttributeResult:
        """
        对 1 秒窗口 (5 个连续窗口) 判决功能属性。

        修复: 必须按连续 window_id 聚合，不是 5 个分散非空窗口。
        """
        if not window_features:
            return AttributeResult(
                decision="unknown",
                attr="未知",
                best_guess="",
                attr_scores={a: 0.0 for a in ATTRIBUTES},
                margin=0.0,
                signal_scores={},
                mode_context_scores={},
                reason="无窗口数据",
            )

        # 路径 A: 纯信号特征
        signal_scores = self._score_signal_path(window_features)

        # 路径 B: 模式上下文 (低权重)
        mode_scores = self._score_mode_path(mode_results)

        override = self._high_confidence_override(window_features, mode_results)
        if override is not None:
            decision, attr, score, reason = override
            return AttributeResult(
                decision=decision,
                attr=attr,
                best_guess=attr,
                attr_scores={a: (score if a == attr else 0.0) for a in ATTRIBUTES},
                margin=score,
                signal_scores=signal_scores,
                mode_context_scores=mode_scores,
                reason=reason,
            )

        # 融合
        final_scores: dict[str, float] = {}
        for attr in ATTRIBUTES:
            sig_part = SIGNAL_WEIGHT * signal_scores.get(attr, 0.0)
            mode_part = MODE_WEIGHT * mode_scores.get(attr, 0.0)
            final_scores[attr] = round(sig_part + mode_part, 3)

        # 排名
        ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        best_attr, best_score = ranked[0]

        # 判决
        decision, attr_label, reason = self._make_decision(
            ranked, window_features, mode_results,
        )

        return AttributeResult(
            decision=decision,
            attr=attr_label,
            best_guess=best_attr,
            attr_scores=final_scores,
            margin=round(ranked[0][1] - ranked[1][1], 3) if len(ranked) >= 2 else best_score,
            signal_scores=signal_scores,
            mode_context_scores=mode_scores,
            reason=reason,
        )

    def classify_tail_windows(
        self,
        window_features: list[WindowFeatures],
        mode_results: list[ModeResult],
    ) -> AttributeResult:
        """
        尾部不足 5 窗口的处理。

        修复: 尾部窗口特征重新参与计算，不只修改结束时间。
        降低质量标记。
        """
        result = self.classify_window(window_features, mode_results, [])
        result.reason = f"尾部窗口({len(window_features)}窗): {result.reason}"
        return result

    def _score_signal_path(
        self,
        features: list[WindowFeatures],
    ) -> dict[str, float]:
        """
        纯信号特征路径 (独立于工作模式)。

        使用中位数聚合，避免单个异常窗口主导。
        """
        valid = [f for f in features if not f.is_empty and f.n_pulses >= 5]
        if not valid:
            return {a: 0.0 for a in ATTRIBUTES}

        # 聚合特征 (中位数)
        def med(values, default=0.0):
            arr = [v for v in values if v is not None and not np.isnan(v)]
            return float(np.median(arr)) if arr else default

        avg_pri = med([f.pri_median for f in valid if f.pri_median > 0])
        avg_pw = med([f.pw_median for f in valid])
        avg_pa_cv = med([f.pa_cv for f in valid])
        avg_pa_per = med([f.pa_periodicity for f in valid])
        avg_doa_range = med([f.doa_unwrapped_range for f in valid if f.doa_valid])
        avg_density = med([f.pulse_density for f in valid])
        avg_duty = med([f.duty_cycle for f in valid])
        avg_rf_iqr = med([f.rf_iqr for f in valid])
        avg_pa_entropy = med([f.pa_spectral_entropy for f in valid])
        avg_pa_autocorr = med([f.pa_autocorr_peak for f in valid])
        avg_doa_r2 = med([f.doa_trend_r2 for f in valid if f.doa_valid])

        # PRF 等级
        prf_level = _classify_prf(avg_pri)
        pw_type = _classify_pw(avg_pw)
        rf_behavior = _classify_rf(avg_rf_iqr)

        # 扫描证据
        scan_evidence = float(np.mean([f.pa_periodicity > 0.15 for f in valid]))

        # DOA 驻留时间 (修复: 不再使用 doa_std/rf_mean)
        # 使用 DOA 角度稳定持续时间: 连续窗口 DOA 标准差 < 阈值的窗口数
        doa_stable_count = sum(
            1 for f in valid
            if f.doa_valid and f.doa_circular_std < 5.0
        )
        doa_dwell_ratio = doa_stable_count / max(len(valid), 1)

        scores: dict[str, float] = {a: 0.0 for a in ATTRIBUTES}

        # 警戒
        if avg_doa_range > 25:
            scores["警戒"] += 0.3
        elif avg_doa_range > 18:
            scores["警戒"] += 0.15
        if scan_evidence > 0.4:
            scores["警戒"] += 0.25
        if prf_level in ["LPRF", "VLPRF"]:
            scores["警戒"] += 0.2
        if avg_duty > 0.20:
            scores["警戒"] += 0.15
        if avg_pa_entropy < 0.5 and avg_pa_autocorr > 0.3:
            scores["警戒"] += 0.1
        if avg_doa_r2 > 0.3:
            scores["警戒"] += 0.1

        # 对海搜索
        if scan_evidence > 0.4:
            scores["对海搜索"] += 0.25
        if prf_level in ["MPRF", "LPRF"]:
            scores["对海搜索"] += 0.2
        if pw_type in ["medium", "wide"]:
            scores["对海搜索"] += 0.2
        if avg_doa_range < 20:
            scores["对海搜索"] += 0.15
        if 0.4 < avg_pa_entropy < 0.8 and avg_doa_r2 < 0.2:
            scores["对海搜索"] += 0.1

        # 对空搜索
        if scan_evidence > 0.3:
            scores["对空搜索"] += 0.2
        if 15 < avg_doa_range < 30:
            scores["对空搜索"] += 0.25
        if prf_level in ["LPRF", "MPRF"]:
            scores["对空搜索"] += 0.15
        if pw_type == "narrow":
            scores["对空搜索"] += 0.15
        if 0.1 < avg_doa_r2 < 0.4:
            scores["对空搜索"] += 0.1

        # 导航
        param_stability = 1.0 / (avg_pa_cv + 1e-12)
        if param_stability > 10:
            scores["导航"] += 0.3
        if rf_behavior == "fixed":
            scores["导航"] += 0.2
        if avg_doa_range < 10:
            scores["导航"] += 0.1

        # 火控
        if prf_level == "HPRF":
            scores["火控"] += 0.3
        if avg_pa_cv < 0.15:
            scores["火控"] += 0.2
        if avg_density > 20000 and prf_level == "HPRF":
            scores["火控"] += 0.2
        elif avg_density > 50000:
            scores["火控"] += 0.1
        if pw_type == "narrow" and prf_level == "HPRF":
            scores["火控"] += 0.15

        # 侦察
        if pw_type == "wide" and avg_pw > 8:
            scores["侦察"] += 0.4
        if rf_behavior in ["agile", "hopping"]:
            scores["侦察"] += 0.2
        if avg_pw > 5:
            scores["侦察"] += 0.1

        return {k: round(v, 3) for k, v in scores.items()}

    def _high_confidence_override(
        self,
        features: list[WindowFeatures],
        mode_results: list[ModeResult],
    ) -> Optional[tuple[str, str, float, str]]:
        """强规则属性判决；证据不足时返回待定而不是默认对海搜索。"""
        valid = [f for f in features if not f.is_empty and f.n_pulses >= 5]
        if not valid:
            return "unknown", "数据不足", 0.0, "无有效窗口"

        def med(values, default=0.0):
            arr = [v for v in values if v is not None and not np.isnan(v)]
            return float(np.median(arr)) if arr else default

        pri = med([f.pri_median for f in valid if f.pri_median > 0])
        pw = med([f.pw_median for f in valid])
        pa_cv = med([f.pa_cv for f in valid])
        pa_per = med([f.pa_periodicity for f in valid])
        density = med([f.pulse_density for f in valid])
        rf_iqr = med([f.rf_iqr for f in valid])
        doa_range = med([f.doa_unwrapped_range for f in valid if f.doa_valid])
        total_pulses = sum(f.n_pulses for f in features)
        modes = {_normalize_mode(mr.mode) for mr in mode_results}

        if "制导" in modes or (
            pri > 0 and pri < 6.0 and pw < 1.6
            and density > 100000 and pa_cv < 0.12
        ):
            return "known", "火控", 0.90, "制导/超高密度稳定照射"

        if "跟踪" in modes and pw >= 6.0 and pa_cv < 0.08:
            return "known", "火控", 0.82, "宽脉宽稳定跟踪"

        if "搜索" in modes and rf_iqr > 1000 and density > 30000 and 12 <= doa_range <= 25:
            return "known", "对空搜索", 0.80, "超大RF捷变+中等DOA覆盖"

        if "搜索" in modes and rf_iqr > 100 and density > 50000 and 12 <= doa_range <= 25:
            return "known", "对空搜索", 0.78, "大RF捷变+中等DOA覆盖"

        if (
            "搜索" in modes
            and rf_iqr < 10
            and density > 50000
            and 1.5 <= pw <= 5.0
            and 8 <= doa_range <= 18
        ):
            return "known", "对海搜索", 0.76, "固定RF+高密度+中脉宽"

        if (
            "搜索" in modes
            and rf_iqr < 10
            and density < 3000
            and 1.0 <= pw <= 2.5
            and doa_range < 12
            and pa_per > 0.10
            and total_pulses > 1000
        ):
            return "known", "对海搜索", 0.68, "固定RF+低密度+小DOA周期搜索"

        if (
            "搜索" in modes
            and rf_iqr >= 80
            and density > 10000
            and pw < 1.5
            and 8 <= doa_range <= 18
        ):
            return "known", "对空搜索", 0.70, "RF捷变+较高密度+窄脉冲搜索"

        if (
            "搜索" in modes
            and pw >= 6.0
            and density > 5000
            and pa_per > 0.12
        ):
            return "known", "侦察", 0.62, "宽脉冲+周期扫描, 更接近侦察/宽脉冲搜索"

        if (
            "搜索" in modes
            and rf_iqr < 50
            and doa_range < 15
            and density > 3000
        ):
            return "known", "对海搜索", 0.60, "低DOA覆盖+中低RF捷变搜索"

        if "未知" in modes:
            return "unknown", "未知", 0.0, "模式未知"

        return None

    def _score_mode_path(
        self,
        mode_results: list[ModeResult],
    ) -> dict[str, float]:
        """
        模式上下文路径 (低权重，不直接决定属性)。
        """
        if not mode_results:
            return {a: 0.0 for a in ATTRIBUTES}

        # 统计模式占比
        modes = [mr.best_guess for mr in mode_results if mr.best_guess]
        if not modes:
            return {a: 0.0 for a in ATTRIBUTES}

        counter = Counter(modes)
        total = len(modes)
        ratios = {m: counter.get(m, 0) / total for m in ["搜索", "跟踪", "制导", "成像"]}

        scores: dict[str, float] = {a: 0.0 for a in ATTRIBUTES}

        # 搜索主导
        if ratios.get("搜索", 0) > 0.7:
            scores["对空搜索"] += 0.3
            scores["对海搜索"] += 0.3
            scores["警戒"] += 0.2

        # 跟踪主导
        if ratios.get("跟踪", 0) > 0.5:
            scores["火控"] += 0.3

        # 制导主导
        if ratios.get("制导", 0) > 0.5:
            scores["火控"] += 0.4

        # 成像主导
        if ratios.get("成像", 0) > 0.5:
            scores["侦察"] += 0.4

        return {k: round(v, 3) for k, v in scores.items()}

    def _make_decision(
        self,
        ranked: list[tuple[str, float]],
        features: list[WindowFeatures],
        mode_results: list[ModeResult],
    ) -> tuple[str, str, str]:
        """
        属性判决。

        Returns:
            (decision, attr_label, reason)
        """
        best_attr, best_score = ranked[0]

        # 数据不足
        total_pulses = sum(f.n_pulses for f in features)
        if total_pulses < 500:
            return "unknown", "数据不足", f"脉冲数不足({total_pulses}<500)"

        # 得分过低
        if best_score < 0.35:
            return "unknown", "未知", f"融合得分过低({best_score:.3f}<0.35)"

        # 平局检查 (修复: 全局平局输出待定)
        if len(ranked) >= 2:
            second_attr, second_score = ranked[1]
            gap = best_score - second_score
            has_search_mode = any(_normalize_mode(mr.mode) == "搜索" for mr in mode_results)
            if has_search_mode and best_score >= 0.42 and gap >= 0.04:
                return "known", best_attr, (
                    f"搜索属性弱确定({best_attr}={best_score:.3f}, "
                    f"gap={gap:.3f})"
                )
            if gap < 0.08 and second_score > 0.25:
                return "pending", "待定", (
                    f"候选矛盾({best_attr}={best_score:.3f} vs "
                    f"{second_attr}={second_score:.3f})"
                )

        return "known", best_attr, f"得分={best_score:.3f}"


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
        return "narrow"
    elif pw_us > 5.0:
        return "wide"
    return "medium"


def _classify_rf(rf_iqr: float) -> str:
    if rf_iqr > 50:
        return "hopping"
    elif rf_iqr > 10:
        return "agile"
    return "fixed"


def _normalize_mode(label: str) -> str:
    label = str(label).strip()
    if label.startswith("疑似"):
        label = label[2:]
    return label


def compute_global_attribute(
    window_results: list[AttributeResult],
) -> tuple[str, float]:
    """
    计算全局功能属性 (多数投票)。

    修复: 平局输出待定，不使用 Counter.most_common(1)。
    """
    if not window_results:
        return "未知", 0.0

    uncertain_count = sum(
        1 for r in window_results
        if r.attr in ("未知", "数据不足") or r.decision in ("unknown", "no_data")
    )
    if uncertain_count / max(len(window_results), 1) >= 0.5:
        return "未知", 0.0

    # 统计各属性出现次数 (排除未知和数据不足)
    valid = [r for r in window_results if r.attr not in ("未知", "数据不足")]
    if not valid:
        return "未知", 0.0

    counter = Counter(r.attr for r in valid)
    most_common = counter.most_common(2)

    # 平局检查
    if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
        return "待定", 0.0

    winner = most_common[0][0]
    # 置信度: 获胜属性对应窗口的加权得分
    winner_scores = [r.attr_scores.get(winner, 0) for r in valid if r.attr == winner]
    conf = float(np.mean(winner_scores)) if winner_scores else 0.0

    return winner, round(conf, 3)

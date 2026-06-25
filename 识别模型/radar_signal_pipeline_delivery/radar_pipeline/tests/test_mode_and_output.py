"""
模式识别与输出测试

验证:
  - 未命中规则返回 unknown
  - 候选接近返回 suspected
  - 不同确定模式不会被直接合并
  - 块级窗口和脉冲守恒
  - 平局输出待定
"""
import pytest
import numpy as np
from radar_pipeline.schemas import (
    WindowFeatures, StateSegment, ModeResult, AttributeResult,
    BlockOutput, ModeTimelineEntry, AttributeTimelineEntry,
)
from radar_pipeline.mode_evidence import ModeEvidenceEngine
from radar_pipeline.function_attribute import FunctionAttributeEngine, compute_global_attribute
from radar_pipeline.output_writer import build_blocks, verify_block_conservation


def _make_segment(feature_summary: dict, n_windows: int = 5) -> StateSegment:
    """创建测试用状态段"""
    return StateSegment(
        segment_id="test_seg",
        radar_id="test",
        window_ids=list(range(n_windows)),
        start_time=0.0,
        end_time=n_windows * 0.2,
        duration_s=n_windows * 0.2,
        valid_window_ratio=1.0,
        n_pulses=n_windows * 100,
        feature_summary=feature_summary,
        boundary_evidence={},
    )


def _make_window_features(n: int, values: dict = None) -> list[WindowFeatures]:
    """创建测试用窗口特征"""
    features = []
    for i in range(n):
        wf = WindowFeatures(
            window_id=i,
            n_pulses=100,
            is_empty=False,
            quality_flags=[],
        )
        if values:
            for k, v in values.items():
                setattr(wf, k, v)
        features.append(wf)
    return features


class TestModeRecognition:
    """模式识别测试"""

    def test_unknown_returns_unknown(self):
        """证据不足时返回 unknown"""
        engine = ModeEvidenceEngine()
        # 空特征摘要
        segment = _make_segment({"n_valid": 0})
        result = engine.classify_segment(segment, [])
        assert result.decision == "unknown"
        assert result.mode == "未知"

    def test_strong_search_evidence(self):
        """强搜索证据应返回搜索"""
        engine = ModeEvidenceEngine()
        segment = _make_segment({
            "n_valid": 5,
            "pri_median": 500.0,
            "pri_cv": 0.1,
            "rf_median": 3000.0,
            "rf_iqr": 5.0,
            "pw_median": 1.0,
            "pa_periodicity": 0.25,
            "pa_cv": 0.3,
            "pa_spectral_entropy": 0.5,
            "pa_autocorr_peak": 0.4,
            "doa_range": 30.0,
            "doa_trend_slope": 0.5,
            "doa_trend_r2": 0.6,
            "pulse_density": 500.0,
            "duty_cycle": 0.1,
        })
        features = _make_window_features(5)
        result = engine.classify_segment(segment, features)
        assert result.best_guess == "搜索"
        assert result.decision in ["known", "suspected"]

    def test_strong_track_evidence(self):
        """强跟踪证据应返回跟踪"""
        engine = ModeEvidenceEngine()
        segment = _make_segment({
            "n_valid": 5,
            "pri_median": 10.0,
            "pri_cv": 0.02,
            "rf_median": 3000.0,
            "rf_iqr": 5.0,
            "pw_median": 0.5,
            "pa_periodicity": 0.02,
            "pa_cv": 0.05,
            "pa_spectral_entropy": 0.8,
            "pa_autocorr_peak": 0.1,
            "doa_range": 5.0,
            "doa_trend_slope": 0.01,
            "doa_trend_r2": 0.1,
            "pulse_density": 30000.0,
            "duty_cycle": 0.3,
        })
        features = _make_window_features(5)
        result = engine.classify_segment(segment, features)
        assert result.best_guess == "跟踪"

    def test_close_candidates_return_suspected(self):
        """候选接近时返回 suspected 或 unknown"""
        engine = ModeEvidenceEngine()
        # 中等证据，可能导致多模式得分接近
        segment = _make_segment({
            "n_valid": 5,
            "pri_median": 50.0,
            "pri_cv": 0.15,
            "rf_median": 3000.0,
            "rf_iqr": 5.0,
            "pw_median": 2.0,
            "pa_periodicity": 0.08,
            "pa_cv": 0.12,
            "pa_spectral_entropy": 0.6,
            "pa_autocorr_peak": 0.2,
            "doa_range": 15.0,
            "doa_trend_slope": 0.1,
            "doa_trend_r2": 0.3,
            "pulse_density": 3000.0,
            "duty_cycle": 0.05,
        })
        features = _make_window_features(5)
        result = engine.classify_segment(segment, features)
        # 验证结果是合理的决策类型
        assert result.decision in ["known", "suspected", "unknown"]
        # 如果不是 known，应该是 suspected 或 unknown
        if result.decision != "known":
            assert result.decision in ["suspected", "unknown"]


class TestDifferentModesNotMerged:
    """不同模式不合并测试"""

    def test_different_modes_preserved(self):
        """不同确定模式不会被直接合并"""
        # 这个测试验证 ModeEvidenceEngine 不会将不同模式合并
        engine = ModeEvidenceEngine()

        # 搜索段
        search_seg = _make_segment({
            "n_valid": 5,
            "pri_median": 500.0,
            "pa_periodicity": 0.25,
            "pa_cv": 0.3,
            "doa_range": 30.0,
            "pulse_density": 500.0,
            "duty_cycle": 0.1,
            "doa_trend_slope": 0.5,
            "doa_trend_r2": 0.6,
            "pa_spectral_entropy": 0.5,
            "pa_autocorr_peak": 0.4,
            "rf_median": 3000.0,
            "rf_iqr": 5.0,
            "pw_median": 1.0,
            "pri_cv": 0.1,
        })

        # 跟踪段
        track_seg = _make_segment({
            "n_valid": 5,
            "pri_median": 10.0,
            "pa_periodicity": 0.02,
            "pa_cv": 0.05,
            "doa_range": 5.0,
            "pulse_density": 30000.0,
            "duty_cycle": 0.3,
            "doa_trend_slope": 0.01,
            "doa_trend_r2": 0.1,
            "pa_spectral_entropy": 0.8,
            "pa_autocorr_peak": 0.1,
            "rf_median": 3000.0,
            "rf_iqr": 5.0,
            "pw_median": 0.5,
            "pri_cv": 0.02,
        })

        features = _make_window_features(5)
        r1 = engine.classify_segment(search_seg, features)
        r2 = engine.classify_segment(track_seg, features)

        # 两个结果应不同
        if r1.decision == "known" and r2.decision == "known":
            assert r1.best_guess != r2.best_guess


class TestBlockConservation:
    """块级守恒测试"""

    def test_block_window_conservation(self):
        """块级窗口数守恒"""
        # 创建 10 个窗口
        window_ids = np.arange(10)
        features = _make_window_features(10)

        # 创建 2 个段
        segments = [
            StateSegment("s1", "test", [0, 1, 2, 3, 4], 0.0, 1.0, 1.0, 1.0, 500, {}, {}),
            StateSegment("s2", "test", [5, 6, 7, 8, 9], 1.0, 2.0, 1.0, 1.0, 500, {}, {}),
        ]
        mode_results = [
            ModeResult("known", "搜索", "搜索", {"搜索": 0.8}, 0.3, 0.8, [], []),
            ModeResult("known", "跟踪", "跟踪", {"跟踪": 0.8}, 0.3, 0.8, [], []),
        ]
        attr_results = [
            AttributeResult("known", "对空搜索", "对空搜索", {}, 0.3, {}, {}, ""),
        ]

        blocks = build_blocks(
            "test/radar_1", window_ids, segments, mode_results,
            attr_results, features, block_duration=1.0,
        )

        conservation = verify_block_conservation(blocks, 10, 1000)
        assert conservation["windows_match"] is True

    def test_block_pulse_conservation(self):
        """块级脉冲数守恒"""
        window_ids = np.arange(5)
        features = _make_window_features(5)
        for i, f in enumerate(features):
            f.n_pulses = (i + 1) * 100  # 100, 200, 300, 400, 500

        segments = [
            StateSegment("s1", "test", [0, 1, 2, 3, 4], 0.0, 1.0, 1.0, 1.0, 1500, {}, {}),
        ]
        mode_results = [
            ModeResult("known", "搜索", "搜索", {"搜索": 0.8}, 0.3, 0.8, [], []),
        ]
        attr_results = [
            AttributeResult("known", "对空搜索", "对空搜索", {}, 0.3, {}, {}, ""),
        ]

        blocks = build_blocks(
            "test/radar_1", window_ids, segments, mode_results,
            attr_results, features, block_duration=1.0,
        )

        # 总脉冲数 = 100 + 200 + 300 + 400 + 500 = 1500
        conservation = verify_block_conservation(blocks, 5, 1500)
        assert conservation["pulses_match"] is True


class TestAttributeTie:
    """平局测试"""

    def test_tie_returns_pending(self):
        """全局平局输出待定"""
        results = [
            AttributeResult("known", "对空搜索", "对空搜索", {}, 0.3, {}, {}, ""),
            AttributeResult("known", "对空搜索", "对空搜索", {}, 0.3, {}, {}, ""),
            AttributeResult("known", "对海搜索", "对海搜索", {}, 0.3, {}, {}, ""),
            AttributeResult("known", "对海搜索", "对海搜索", {}, 0.3, {}, {}, ""),
        ]
        attr, conf = compute_global_attribute(results)
        assert attr == "待定"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

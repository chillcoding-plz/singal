"""
变化点检测测试

验证:
  - 特征尺度缩放不改变变化点位置
  - 无变化序列不产生规律性伪变化点
  - 已知切换点位置正确
  - 空窗口创建辐射边界
"""
import pytest
import numpy as np
from radar_pipeline.schemas import WindowFeatures
from radar_pipeline.change_detection import (
    detect_change_points, detect_radiation_boundaries,
    _build_feature_matrix, _robust_normalize,
)


def _make_features(n: int, base_values: dict = None) -> list[WindowFeatures]:
    """创建测试用窗口特征"""
    features = []
    for i in range(n):
        wf = WindowFeatures(
            window_id=i,
            n_pulses=100,
            is_empty=False,
            quality_flags=[],
        )
        if base_values:
            for k, v in base_values.items():
                setattr(wf, k, v)
        features.append(wf)
    return features


def _make_features_with_change(
    n: int,
    change_at: int,
    before: dict,
    after: dict,
) -> list[WindowFeatures]:
    """创建包含已知切换点的特征序列"""
    features = []
    for i in range(n):
        wf = WindowFeatures(
            window_id=i,
            n_pulses=100,
            is_empty=False,
            quality_flags=[],
        )
        values = before if i < change_at else after
        for k, v in values.items():
            setattr(wf, k, v)
        features.append(wf)
    return features


class TestScaleInvariance:
    """特征尺度不变性测试"""

    def test_scale_invariance(self):
        """特征整体乘以常数后变化点不改变"""
        n = 20
        # 前 10 个窗口: PRI=100, 后 10 个: PRI=200
        features1 = _make_features_with_change(
            n, 10,
            {"pri_median": 100.0, "pa_periodicity": 0.1},
            {"pri_median": 200.0, "pa_periodicity": 0.3},
        )
        # 尺度放大 10000 倍
        features2 = _make_features_with_change(
            n, 10,
            {"pri_median": 1000000.0, "pa_periodicity": 0.1},
            {"pri_median": 2000000.0, "pa_periodicity": 0.3},
        )

        mask = np.ones(n, dtype=bool)
        cps1 = detect_change_points(features1, mask, method="custom", min_gap=2)
        cps2 = detect_change_points(features2, mask, method="custom", min_gap=2)

        # 变化点位置应相同
        indices1 = [cp.index for cp in cps1]
        indices2 = [cp.index for cp in cps2]
        assert indices1 == indices2


class TestNoFalseChangePoints:
    """无伪变化点测试"""

    def test_constant_no_change_points(self):
        """恒定序列不产生变化点"""
        n = 30
        features = _make_features(n, {
            "pri_median": 100.0,
            "pa_periodicity": 0.1,
            "pw_median": 1.0,
            "rf_iqr": 5.0,
            "pulse_density": 1000.0,
        })
        mask = np.ones(n, dtype=bool)
        cps = detect_change_points(features, mask, method="custom", min_gap=2)
        assert len(cps) == 0

    def test_no_periodic_false_change_points(self):
        """无变化序列不会规律性每隔固定窗口产生变化点"""
        n = 50
        # 添加小的随机噪声
        np.random.seed(42)
        features = []
        for i in range(n):
            wf = WindowFeatures(
                window_id=i,
                n_pulses=100,
                is_empty=False,
                quality_flags=[],
                pri_median=100.0 + np.random.normal(0, 0.5),
                pa_periodicity=0.1 + np.random.normal(0, 0.005),
                pw_median=1.0,
                rf_iqr=5.0,
                pulse_density=1000.0,
            )
            features.append(wf)

        mask = np.ones(n, dtype=bool)
        cps = detect_change_points(features, mask, method="custom", min_gap=2)

        # 不应有规律间隔
        if len(cps) >= 3:
            indices = [cp.index for cp in cps]
            gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
            # 所有间隔不应相同
            assert len(set(gaps)) > 1


class TestKnownChangePoint:
    """已知变化点检测测试"""

    def test_detect_strong_change(self):
        """强变化点应被检测到"""
        n = 20
        np.random.seed(42)
        # 添加小噪声使 MAD 不为 0，更接近真实数据
        features = []
        for i in range(n):
            wf = WindowFeatures(
                window_id=i, n_pulses=100, is_empty=False, quality_flags=[],
                pri_median=(100 if i < 10 else 300) + np.random.normal(0, 2),
                pa_periodicity=(0.05 if i < 10 else 0.30) + np.random.normal(0, 0.005),
                pulse_density=(500 if i < 10 else 5000) + np.random.normal(0, 10),
                pw_median=(1.0 if i < 10 else 3.0) + np.random.normal(0, 0.05),
                rf_iqr=(5.0 if i < 10 else 50.0) + np.random.normal(0, 1.0),
                duty_cycle=(0.01 if i < 10 else 0.1) + np.random.normal(0, 0.002),
                pa_dynamic_range=(10.0 if i < 10 else 30.0) + np.random.normal(0, 1.0),
                pa_spectral_entropy=(0.5 if i < 10 else 0.3) + np.random.normal(0, 0.02),
                pa_autocorr_peak=(0.1 if i < 10 else 0.5) + np.random.normal(0, 0.02),
                doa_unwrapped_range=(5.0 if i < 10 else 30.0) + np.random.normal(0, 1.0),
            )
            features.append(wf)

        mask = np.ones(n, dtype=bool)
        cps = detect_change_points(features, mask, method="custom", min_gap=2)

        # 应检测到接近位置 10 的变化点
        assert len(cps) >= 1
        indices = [cp.index for cp in cps]
        assert any(abs(idx - 10) <= 2 for idx in indices)


class TestRadiationBoundaries:
    """辐射边界测试"""

    def test_empty_window_creates_boundary(self):
        """空窗口与非空窗口交界处创建辐射边界"""
        features = [
            WindowFeatures(0, 100, False, []),
            WindowFeatures(1, 100, False, []),
            WindowFeatures(2, 0, True, ["empty"]),
            WindowFeatures(3, 0, True, ["empty"]),
            WindowFeatures(4, 100, False, []),
        ]
        boundaries = detect_radiation_boundaries(features)
        # 应在位置 2 (非空→空) 和位置 4 (空→非空) 有边界
        assert len(boundaries) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

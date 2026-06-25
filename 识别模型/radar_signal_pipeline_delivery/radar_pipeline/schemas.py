"""
新管线核心数据结构

定义所有模块共享的数据类型，确保数据契约明确。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ── 窗口级数据 ────────────────────────────────────────────────────────

@dataclass
class WindowRecord:
    """单个 200ms 窗口的完整记录"""
    radar_id: str
    sample: str
    window_id: int                    # 全局唯一窗口编号 (基于 floor(TOA/0.2))
    start_time: float                 # float64, 窗口起始时间 (s)
    end_time: float                   # float64, 窗口结束时间 (s)
    pdw: Optional[np.ndarray]         # (N, 6) = [TOA(float64), RF, PW, PA, DOA, Param6]
    n_pulses: int
    is_empty: bool
    quality_flags: list[str] = field(default_factory=list)
    upstream_label: Optional[int] = None
    upstream_confidence: Optional[float] = None


@dataclass
class WindowFeatures:
    """单个窗口的基础特征 (200ms 尺度)"""
    window_id: int
    n_pulses: int
    is_empty: bool
    quality_flags: list[str]

    # PRI
    pri_median: float = 0.0
    pri_mean: float = 0.0
    pri_iqr: float = 0.0
    pri_cv: float = 0.0
    pri_q10: float = 0.0
    pri_q90: float = 0.0
    pri_outlier_ratio: float = 0.0

    # RF
    rf_median: float = 0.0
    rf_iqr: float = 0.0
    rf_jump_ratio: float = 0.0
    rf_discrete_count: int = 0

    # PW
    pw_median: float = 0.0
    pw_iqr: float = 0.0
    pw_q10: float = 0.0
    pw_q90: float = 0.0
    pw_category: str = ""

    # PA
    pa_median: float = 0.0
    pa_iqr: float = 0.0
    pa_dynamic_range: float = 0.0
    pa_local_trend: float = 0.0
    pa_periodicity: float = 0.0         # 单窗口 periodogram (局部波动指标)
    pa_spectral_entropy: float = 0.0
    pa_autocorr_peak: float = 0.0
    pa_zero_cross_rate: float = 0.0
    pa_cv: float = 0.0

    # DOA (圆统计)
    doa_circular_mean: float = 0.0      # 圆均值 (度)
    doa_circular_std: float = 0.0       # 圆标准差 (度)
    doa_unwrapped_range: float = 0.0    # unwrap 后的范围 (度)
    doa_unwrapped_trend: float = 0.0    # unwrap 后的线性趋势 (度/脉冲)
    doa_trend_r2: float = 0.0
    doa_valid: bool = True              # DOA 数据是否有效

    # 密度与占空比
    pulse_density: float = 0.0          # n_pulses / 0.2 (固定窗口长度)
    duty_cycle: float = 0.0             # 总脉宽 / 0.2

    # 中间分类
    prf_level: str = ""
    pri_mode: str = ""
    beam_mode: str = ""
    rf_mode: str = ""


@dataclass
class TemporalFeatures:
    """跨窗口时序特征 (多时间尺度)"""
    window_id: int
    center_time: float

    # 1s (5窗口) 尺度
    pa_envelope_autocorr: float = 0.0
    pa_envelope_peak_freq: float = 0.0
    doa_movement_speed: float = 0.0     # 圆均值移动速度 (度/s)
    pri_rf_pw_stability: float = 0.0    # 跨窗口参数稳定性
    radiation_on_ratio: float = 0.0     # 辐射开关占比

    # 2s (10窗口) 尺度
    doa_revisit_period: float = 0.0     # DOA 回访周期 (s)
    consecutive_active: int = 0         # 连续活跃窗口数
    consecutive_silent: int = 0         # 连续静默窗口数

    # 5s (25窗口) 尺度
    pa_slow_period: float = 0.0         # PA 慢周期行为
    scan_revisit_evidence: float = 0.0  # 扫描回访证据

    # 覆盖率
    valid_window_ratio_1s: float = 0.0
    valid_window_ratio_2s: float = 0.0


# ── 变化点 ────────────────────────────────────────────────────────────

@dataclass
class ChangePoint:
    """变化点"""
    index: int                          # 窗口序列中的位置
    score: float                        # 变化强度得分
    contributing_features: list[str]    # 贡献特征名称
    feature_scores: dict[str, float] = field(default_factory=dict)


# ── 状态段 ────────────────────────────────────────────────────────────

@dataclass
class StateSegment:
    """状态段: 由变化点分割的连续窗口序列"""
    segment_id: str
    radar_id: str
    window_ids: list[int]
    start_time: float
    end_time: float
    duration_s: float
    valid_window_ratio: float
    n_pulses: int
    feature_summary: dict               # 段级特征聚合 (中位数/IQR)
    boundary_evidence: dict             # 边界变化点证据


@dataclass
class ModeResult:
    """工作模式判决结果"""
    decision: str                       # "known" | "suspected" | "unknown"
    mode: str                           # 最终模式标签 (如 "搜索", "疑似跟踪", "未知")
    best_guess: str                     # 最佳猜测模式
    mode_scores: dict[str, float]       # 各模式得分
    margin: float                       # 最高与次高得分差距
    evidence_score: float               # 证据得分 (非准确率)
    supporting_evidence: list[str]      # 支持证据列表
    conflicting_evidence: list[str]     # 冲突证据列表
    reason: str = ""                    # 判决原因


@dataclass
class AttributeResult:
    """功能属性判决结果"""
    decision: str                       # "known" | "suspected" | "unknown" | "pending"
    attr: str                           # 最终属性标签
    best_guess: str
    attr_scores: dict[str, float]
    margin: float
    signal_scores: dict[str, float]     # 纯信号路径得分
    mode_context_scores: dict[str, float]  # 模式上下文得分
    reason: str = ""


# ── 时间线输出 ────────────────────────────────────────────────────────

@dataclass
class ModeTimelineEntry:
    """工作模式时间线条目"""
    segment_id: str
    start_time: float
    end_time: float
    duration_s: float
    mode_result: ModeResult
    window_ids: list[int]
    n_pulses: int
    feature_summary: dict


@dataclass
class AttributeTimelineEntry:
    """功能属性时间线条目"""
    window_ids: list[int]
    start_time: float
    end_time: float
    duration_s: float
    attr_result: AttributeResult
    n_pulses: int


@dataclass
class BlockOutput:
    """5秒块输出"""
    block_index: int
    radar_id: str
    sample: str
    time_start: float
    time_end: float
    mode_timeline: list[ModeTimelineEntry]
    attribute_timeline: list[AttributeTimelineEntry]
    n_windows: int                      # 块内窗口数 (守恒)
    n_pulses: int                       # 块内脉冲数 (守恒)
    block_summary: dict


@dataclass
class RunManifest:
    """运行清单"""
    run_id: str
    timestamp: str
    code_version: str
    python_version: str
    input_files: list[str]
    input_hashes: dict[str, str]
    config_hash: str
    total_windows: int
    total_pulses: int
    elapsed_seconds: float
    parameters: dict

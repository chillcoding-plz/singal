"""
窗口级特征计算

对每个 200ms 窗口计算基础信号特征。
修复原 stats_computer.py 的多个问题:
  - DOA 使用圆统计和角度 unwrap
  - 脉冲密度使用固定 0.2s 窗口长度
  - 空窗口特征为缺失，不填 0
  - 低脉冲窗口只计算可支持的特征
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.signal import periodogram
from scipy.ndimage import median_filter
from .schemas import WindowRecord, WindowFeatures


def compute_window_features(record: WindowRecord) -> WindowFeatures:
    """
    计算单个窗口的基础特征。

    Args:
        record: 窗口记录

    Returns:
        WindowFeatures, 空窗口的数值特征为 NaN (缺失)
    """
    wf = WindowFeatures(
        window_id=record.window_id,
        n_pulses=record.n_pulses,
        is_empty=record.is_empty,
        quality_flags=list(record.quality_flags),
    )

    if record.is_empty or record.pdw is None or record.n_pulses == 0:
        _fill_nan(wf)
        return wf

    pdw = record.pdw
    toa = pdw[:, 0].astype(np.float64)   # float64
    rf = pdw[:, 1]
    pw = pdw[:, 2]
    pa = pdw[:, 3]
    doa = pdw[:, 4]
    n = len(pdw)

    # ── PRI (使用 float64 TOA 差值) ──
    if n >= 2:
        pri = np.diff(toa) * 1e6  # 转换为微秒
        pri = pri[pri > 0]
        if len(pri) > 0:
            wf.pri_median = float(np.median(pri))
            wf.pri_mean = float(np.mean(pri))
            q25, q75 = np.percentile(pri, [25, 75])
            wf.pri_iqr = float(q75 - q25)
            wf.pri_cv = float(np.std(pri) / (wf.pri_mean + 1e-12))
            wf.pri_q10 = float(np.percentile(pri, 10))
            wf.pri_q90 = float(np.percentile(pri, 90))
            # 离群比例: 超出 IQR 1.5 倍的比例
            iqr = wf.pri_iqr
            if iqr > 0:
                lower = q25 - 1.5 * iqr
                upper = q75 + 1.5 * iqr
                wf.pri_outlier_ratio = float(np.mean((pri < lower) | (pri > upper)))

    # ── RF ──
    wf.rf_median = float(np.median(rf))
    q25, q75 = np.percentile(rf, [25, 75])
    wf.rf_iqr = float(q75 - q25)
    if n >= 2:
        rf_diffs = np.abs(np.diff(rf))
        wf.rf_jump_ratio = float(np.mean(rf_diffs > 5.0))
    # 离散频点数 (聚类到 1MHz 精度)
    if n >= 2:
        rf_rounded = np.round(rf, 0)
        wf.rf_discrete_count = int(len(np.unique(rf_rounded)))

    # ── PW ──
    wf.pw_median = float(np.median(pw))
    q25, q75 = np.percentile(pw, [25, 75])
    wf.pw_iqr = float(q75 - q25)
    wf.pw_q10 = float(np.percentile(pw, 10))
    wf.pw_q90 = float(np.percentile(pw, 90))

    # ── PA ──
    wf.pa_median = float(np.median(pa))
    q25, q75 = np.percentile(pa, [25, 75])
    wf.pa_iqr = float(q75 - q25)
    wf.pa_dynamic_range = float(np.ptp(pa))
    pa_mean = float(np.mean(pa))
    wf.pa_cv = float(np.std(pa) / (abs(pa_mean) + 1e-12))
    # PA 局部趋势 (线性拟合斜率)
    if n >= 4:
        coeffs = np.polyfit(np.arange(n, dtype=np.float64), pa, 1)
        wf.pa_local_trend = float(coeffs[0])

    # PA 频域特征
    if n >= 8:
        _compute_pa_spectral_features(pa, wf)

    # ── DOA (圆统计) ──
    doa_invalid = "doa_invalid" in record.quality_flags
    wf.doa_valid = not doa_invalid
    if not doa_invalid and n >= 2:
        _compute_doa_circular(doa, toa, wf)
    else:
        wf.doa_circular_mean = 0.0
        wf.doa_circular_std = 0.0
        wf.doa_unwrapped_range = 0.0
        wf.doa_unwrapped_trend = 0.0
        wf.doa_trend_r2 = 0.0

    # ── 脉冲密度 (使用固定窗口长度 0.2s) ──
    wf.pulse_density = float(n / WINDOW_DURATION)

    # ── 占空比 ──
    total_pw = float(np.sum(pw))
    wf.duty_cycle = total_pw / WINDOW_DURATION

    # ── PRF 等级 (中间分类) ──
    wf.prf_level = _classify_prf(wf.pri_median if wf.pri_median > 0 else wf.pri_mean)

    # ── PW 类别 ──
    wf.pw_category = _classify_pw(wf.pw_median)

    # ── RF 模式 ──
    wf.rf_mode = _classify_rf(wf.rf_median, wf.rf_iqr, wf.rf_jump_ratio, n)

    # ── PA 周期性 (单窗口, 局部波动指标) ──
    wf.pa_periodicity = _compute_pa_periodicity(pa)

    return wf


WINDOW_DURATION = 0.200  # 固定 200ms


def _fill_nan(wf: WindowFeatures):
    """空窗口: 数值特征填 NaN，不填 0"""
    nan_fields = [
        "pri_median", "pri_mean", "pri_iqr", "pri_cv", "pri_q10", "pri_q90",
        "pri_outlier_ratio", "rf_median", "rf_iqr", "rf_jump_ratio",
        "pw_median", "pw_iqr", "pw_q10", "pw_q90",
        "pa_median", "pa_iqr", "pa_dynamic_range", "pa_local_trend",
        "pa_periodicity", "pa_spectral_entropy", "pa_autocorr_peak",
        "pa_zero_cross_rate", "pa_cv",
        "doa_circular_mean", "doa_circular_std", "doa_unwrapped_range",
        "doa_unwrapped_trend", "doa_trend_r2",
        "pulse_density", "duty_cycle",
    ]
    for f in nan_fields:
        setattr(wf, f, float('nan'))
    wf.rf_discrete_count = 0
    wf.prf_level = ""
    wf.pri_mode = ""
    wf.beam_mode = ""
    wf.rf_mode = ""
    wf.pw_category = ""


def _compute_pa_spectral_features(pa: np.ndarray, wf: WindowFeatures):
    """计算 PA 频域特征 (优化: 使用 FFT 自相关)"""
    try:
        smoothed = median_filter(pa, size=min(5, len(pa)))
        f, pxx = periodogram(smoothed)
        if pxx.sum() > 0:
            # 频谱熵
            pxx_norm = pxx / (pxx.sum() + 1e-12)
            pxx_nz = pxx_norm[pxx_norm > 0]
            if len(pxx_nz) > 1:
                entropy = float(-np.sum(pxx_nz * np.log2(pxx_nz + 1e-12)))
                max_entropy = np.log2(len(pxx_nz))
                wf.pa_spectral_entropy = entropy / (max_entropy + 1e-12)

            # 周期性 (最大峰占比)
            wf.pa_periodicity = float(pxx.max() / pxx.sum())

        # 自相关第一峰 (使用 FFT 加速)
        pa_centered = smoothed - np.mean(smoothed)
        n = len(pa_centered)
        # 使用 FFT 计算自相关 (O(n log n) vs O(n²))
        fft_size = 2 ** int(np.ceil(np.log2(2 * n)))
        fft_pa = np.fft.fft(pa_centered, n=fft_size)
        acf = np.fft.ifft(fft_pa * np.conj(fft_pa)).real[:n]
        acf = acf / (acf[0] + 1e-12)  # 归一化

        if len(acf) > 2:
            # 只搜索前 1/4 的 lag (避免长序列的远距离伪峰)
            search_end = min(len(acf), n // 4)
            peaks = [
                acf[i] for i in range(1, search_end - 1)
                if acf[i] > acf[i - 1] and acf[i] > acf[i + 1]
            ]
            wf.pa_autocorr_peak = float(max(peaks)) if peaks else 0.0

        # 过零率
        pa_diff = np.diff(smoothed)
        zero_crosses = np.sum(np.abs(np.diff(np.sign(pa_diff))) > 0)
        wf.pa_zero_cross_rate = float(zero_crosses / max(len(pa_diff) - 1, 1))

    except Exception as e:
        warnings.warn(f"PA频域特征计算失败: {e}", RuntimeWarning)


def _compute_doa_circular(doa: np.ndarray, toa: np.ndarray, wf: WindowFeatures):
    """
    计算 DOA 圆统计和 unwrap 后的趋势。

    修复原代码:
      - 使用圆均值和圆方差 (处理 359°->0° 回绕)
      - 使用 unwrap 后的角度计算范围和趋势
      - 使用真实 TOA 拟合趋势 (不是脉冲序号)
    """
    # 圆均值和圆方差 (度)
    doa_rad = np.deg2rad(doa)
    sin_mean = np.mean(np.sin(doa_rad))
    cos_mean = np.mean(np.cos(doa_rad))
    wf.doa_circular_mean = float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360)
    # 圆标准差 (度)
    R = np.sqrt(sin_mean ** 2 + cos_mean ** 2)
    wf.doa_circular_std = float(np.rad2deg(np.sqrt(-2 * np.log(max(R, 1e-12)))))

    # unwrap 后的范围和趋势
    doa_unwrapped = np.unwrap(doa_rad)
    doa_unwrapped_deg = np.rad2deg(doa_unwrapped)
    wf.doa_unwrapped_range = float(np.ptp(doa_unwrapped_deg))

    # 使用真实 TOA 拟合线性趋势 (度/s)
    if len(toa) >= 4:
        try:
            # 归一化 TOA 到窗口起始
            t_norm = toa - toa[0]
            coeffs = np.polyfit(t_norm, doa_unwrapped_deg, 1)
            wf.doa_unwrapped_trend = float(coeffs[0])  # 度/s
            # R²
            pred = np.polyval(coeffs, t_norm)
            ss_res = np.sum((doa_unwrapped_deg - pred) ** 2)
            ss_tot = np.sum((doa_unwrapped_deg - np.mean(doa_unwrapped_deg)) ** 2)
            wf.doa_trend_r2 = float(1 - ss_res / (ss_tot + 1e-12)) if ss_tot > 0 else 0.0
        except Exception:
            pass


def _compute_pa_periodicity(pa: np.ndarray) -> float:
    """PA 周期性检测 (单窗口, 局部波动指标)"""
    if len(pa) < 4:
        return 0.0
    try:
        smoothed = median_filter(pa, size=min(5, len(pa)))
        f, pxx = periodogram(smoothed)
        if pxx.sum() > 0:
            return float(pxx.max() / pxx.sum())
    except Exception:
        pass
    return 0.0


def _classify_prf(pri_us: float) -> str:
    """基于 PRI 分类 PRF 等级"""
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
    """基于脉宽分类"""
    if pw_us < 1.5:
        return "窄脉宽"
    elif pw_us > 5.0:
        return "宽脉宽"
    return "中等脉宽"


def _classify_rf(rf_median: float, rf_iqr: float, rf_jump_ratio: float, n: int) -> str:
    """基于 RF 特征分类"""
    if rf_iqr < 15.0:
        return "固定"
    if rf_jump_ratio > 0.01 and rf_iqr > 50.0:
        return "跳频"
    if rf_jump_ratio > 0.005 and rf_iqr > 10.0:
        return "捷变"
    return "固定"


def compute_features_batch(records: list[WindowRecord]) -> list[WindowFeatures]:
    """批量计算窗口特征"""
    return [compute_window_features(r) for r in records]


def build_valid_mask(features: list[WindowFeatures], min_pulses: int = 5) -> np.ndarray:
    """
    构建有效窗口掩码。

    Args:
        features: 窗口特征列表
        min_pulses: 最低脉冲数阈值

    Returns:
        bool 数组, True 表示窗口有效
    """
    mask = np.array([
        (not f.is_empty) and (f.n_pulses >= min_pulses)
        for f in features
    ])
    return mask

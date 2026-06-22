# -*- coding: utf-8 -*-
"""
实验内容：
1. Exp1_PreSort_Only
   只看预分选结果：PredID = SigIdx

2. Exp2_Tframe_Feature_Only
   预分选 + 框架周期特征估计
   不改变 PredID，只为每个 SigIdx 估计 Tframe / 置信度 / 命中率等特征

3. Exp3_Conservative_Merge
   基于实验2的框架周期特征，进行保守跨批合并
   只合并 Tframe 高度一致、置信度高、合并后周期结构仍稳定的 SigIdx

评价指标：
- 分选正确率 Sort_ACC
- 增批率 Add_Rate
- 错批率 Err_Rate

评价方式：
每个 200 ms 节拍内构造 PredID × TrueLabel 交叉表。
PredID 和 TrueLabel 的编号不要求相等，只判断预测批次是否主要来自同一真实目标。
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================
# 1. 参数配置
# ============================================================

@dataclass
class Config:
    # ---------- 进度条 ----------
    show_progress: bool = True

    # ---------- 官方评价相关 ----------
    beat_len: float = 0.2
    purity_thr: float = 0.9
    cover_thr: float = 0.1
    swallow_count_thr: int = 150

    # ---------- TOA 对齐相关 ----------
    toa_match_tolerance: float = 1e-9

    # ---------- 框架周期估计相关 ----------
    # 注意：这里是框架周期范围，不是子 PRI 范围
    frame_T_min: float = 50e-6
    frame_T_max: float = 5e-3

    # 候选周期直方图分辨率
    frame_candidate_bin: float = 0.5e-6

    # 每个 SigIdx 采样多少个 base 点
    num_base_samples: int = 80

    # 每个 base 后面最多采样多少个 ref 点
    num_ref_per_base: int = 500

    # 每个 SigIdx 最多保留多少个候选 Tframe
    top_k_frame_candidates: int = 10

    # 周期验证时的 TOA 容差
    frame_toa_tol: float = 3e-6

    # 给候选 Tframe 打分时最多采样多少个脉冲
    max_score_points: int = 8000

    # 小批次至少多少个脉冲才估计 Tframe
    min_pulses_for_tframe: int = 10

    # 有效 Tframe 的最低要求
    min_hit_rate_for_valid: float = 0.45
    min_span_rate_for_valid: float = 0.40
    min_confidence_for_valid: float = 0.40

    # ---------- 保守跨批合并相关 ----------
    merge_min_confidence: float = 0.75
    merge_min_hit_rate: float = 0.65
    merge_min_span_rate: float = 0.50

    # 周期必须非常接近
    merge_T_rel_tol: float = 0.005
    merge_abs_tol: float = 0.5e-6

    # 第一版先禁用倍周期合并
    allow_harmonic_merge: bool = False
    merge_harmonic_order: int = 4

    # 合并后复验要求
    merge_verify: bool = True
    merge_verify_min_hit_rate: float = 0.60
    merge_verify_min_span_rate: float = 0.45
    merge_verify_min_confidence: float = 0.55

    # 小批次过小不自动合并，避免被大批次吞并
    merge_min_size_ratio: float = 0.02

    # 限制每个周期 bin 内最多比较多少个 SigIdx，防止极端情况下过慢
    max_compare_per_bin: int = 500


def pbar(iterable, desc="", unit="", leave=True, total=None, cfg=None):
    if cfg is not None and not cfg.show_progress:
        return iterable
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, leave=leave, total=total)


# ============================================================
# 2. 数据读取与标准化
# ============================================================

def read_table_auto(path):
    """
    自动读取 txt/csv 文件。
    优先按任意空白符读取，避免出现 Unnamed 空列。
    """
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        df = pd.read_csv(path, sep=",", engine="python")

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if len(unnamed_cols) > 0:
        df = df.drop(columns=unnamed_cols)

    df = df.dropna(axis=1, how="all")
    return df


def normalize_presort_df(df, toa_col="TOA(s)", sigidx_col="SigIdx"):
    """
    预分选结果标准化。
    输出至少包含 PulseID, TOA, SigIdx。
    """
    df = df.copy()

    if toa_col not in df.columns:
        raise ValueError(f"找不到预分选表 TOA 列：{toa_col}，当前列名为：{list(df.columns)}")

    if sigidx_col not in df.columns:
        raise ValueError(f"找不到预分选表 SigIdx 列：{sigidx_col}，当前列名为：{list(df.columns)}")

    df = df.rename(columns={
        toa_col: "TOA",
        sigidx_col: "SigIdx"
    })

    if "PulseID" not in df.columns:
        df["PulseID"] = np.arange(len(df), dtype=np.int64)

    df["TOA"] = df["TOA"].astype(float)
    df["SigIdx"] = df["SigIdx"].astype(int)

    df = df.sort_values("TOA").reset_index(drop=True)
    return df


def normalize_truth_df(df, toa_col="TOA(s)", true_col="SigIdx"):
    """
    验证标签标准化。
    默认使用验证表中的 SigIdx 作为真实分选标签。
    LABEL 一般是型号识别标签，本实验先不用。
    """
    df = df.copy()

    if toa_col not in df.columns:
        raise ValueError(f"找不到验证表 TOA 列：{toa_col}，当前列名为：{list(df.columns)}")

    if true_col not in df.columns:
        raise ValueError(
            f"找不到验证表真实标签列：{true_col}，当前列名为：{list(df.columns)}\n"
            f"如果验证表格式是 TOA(s) SigIdx LABEL，分选评价应设置 truth_label_col='SigIdx'。"
        )

    df = df.rename(columns={
        toa_col: "TOA",
        true_col: "TrueLabel"
    })

    if "PulseID" not in df.columns:
        df["PulseID"] = np.arange(len(df), dtype=np.int64)

    df["TOA"] = df["TOA"].astype(float)
    df["TrueLabel"] = df["TrueLabel"].astype(int)

    df = df.sort_values("TOA").reset_index(drop=True)
    return df


def align_pred_and_truth(pred_df, truth_df, cfg: Config, prefer_pulse_id=False):
    """
    将预测结果与验证标签对齐。
    prefer_pulse_id=True：用 PulseID 对齐，适合两个文件严格同一顺序。
    prefer_pulse_id=False：用 TOA 近邻匹配，适合两个文件顺序可能变化。
    """
    pred_df = pred_df.copy()
    truth_df = truth_df.copy()

    if prefer_pulse_id and "PulseID" in pred_df.columns and "PulseID" in truth_df.columns:
        eval_df = pred_df.merge(
            truth_df[["PulseID", "TrueLabel"]],
            on="PulseID",
            how="inner"
        )
        return eval_df

    pred_df = pred_df.sort_values("TOA").reset_index(drop=True)
    truth_df = truth_df.sort_values("TOA").reset_index(drop=True)

    eval_df = pd.merge_asof(
        pred_df,
        truth_df[["TOA", "TrueLabel"]],
        on="TOA",
        direction="nearest",
        tolerance=cfg.toa_match_tolerance
    )

    eval_df = eval_df.dropna(subset=["TrueLabel"]).copy()
    eval_df["TrueLabel"] = eval_df["TrueLabel"].astype(int)

    return eval_df


# ============================================================
# 3. 官方风格评价
# ============================================================

def evaluate_sorting_official_like(
    df,
    cfg: Config,
    pred_col="PredID",
    true_col="TrueLabel",
    toa_col="TOA",
    exp_name="Eval"
):
    """
    每个 200 ms 节拍内构造 PredID × TrueLabel 交叉表，计算：
    - 分选正确率 Sort_ACC
    - 增批率 Add_Rate
    - 错批率 Err_Rate

    输出是 0~1 小数形式。
    Add_Rate 是平均增批数量，不一定在 0~1 内。
    """
    data = df.copy()
    data = data.dropna(subset=[pred_col, true_col, toa_col])
    data = data.sort_values(toa_col).reset_index(drop=True)

    if len(data) == 0:
        return {
            "Sort_ACC": 0.0,
            "Add_Rate": 0.0,
            "Err_Rate": 0.0,
            "target_detail": pd.DataFrame(),
            "beat_detail": pd.DataFrame()
        }

    toa_start = data[toa_col].min()
    data["Beat"] = np.floor((data[toa_col] - toa_start) / cfg.beat_len).astype(int)

    all_targets = sorted(data[true_col].unique())

    target_appear_beats = {j: 0 for j in all_targets}
    target_success_beats = {j: 0 for j in all_targets}
    target_add_sum = {j: 0 for j in all_targets}

    beat_details = []
    err_rates = []

    beat_groups = list(data.groupby("Beat"))

    for beat_id, beat_df in pbar(
        beat_groups,
        desc=f"{exp_name}：评价200ms节拍",
        unit="beat",
        leave=False,
        cfg=cfg
    ):
        if len(beat_df) == 0:
            continue

        table = pd.crosstab(beat_df[pred_col], beat_df[true_col])

        pred_ids = table.index.tolist()
        true_ids = table.columns.tolist()

        pred_counts = table.sum(axis=1)
        true_counts = table.sum(axis=0)

        # ---------- 特殊错批：小目标被完全吞并 ----------
        swallow_wrong_pred_ids = set()

        for i in pred_ids:
            ni = pred_counts[i]
            if ni <= 0:
                continue

            row = table.loc[i]
            main_j = row.idxmax()
            main_count = row.max()
            main_purity = main_count / ni

            if main_purity >= cfg.purity_thr:
                for k in true_ids:
                    if k == main_j:
                        continue

                    Nk = true_counts[k]
                    Mik = table.loc[i, k]

                    if Nk > cfg.swallow_count_thr and Mik == Nk:
                        swallow_wrong_pred_ids.add(i)

        # ---------- 分选正确率 & 增批率 ----------
        beat_success_targets = 0
        beat_add_total = 0

        for j in true_ids:
            Nj = true_counts[j]

            if Nj <= 0:
                continue

            target_appear_beats[j] += 1

            valid_count = 0
            small_count = 0

            for i in pred_ids:
                if i in swallow_wrong_pred_ids:
                    continue

                ni = pred_counts[i]
                if ni <= 0:
                    continue

                Mij = table.loc[i, j] if j in table.columns else 0

                purity = Mij / ni
                coverage = Mij / Nj

                if purity >= cfg.purity_thr:
                    if coverage >= cfg.cover_thr:
                        valid_count += 1
                    else:
                        small_count += 1

            success = valid_count >= 1

            if success:
                target_success_beats[j] += 1
                beat_success_targets += 1

            add_count = max(valid_count - 1, 0) + small_count
            target_add_sum[j] += add_count
            beat_add_total += add_count

        # ---------- 错批率 ----------
        wrong_batch_num = 0
        total_batch_num = len(pred_ids)

        for i in pred_ids:
            if i in swallow_wrong_pred_ids:
                wrong_batch_num += 1
                continue

            ni = pred_counts[i]
            if ni <= 0:
                continue

            max_purity = 0.0

            for j in true_ids:
                Mij = table.loc[i, j]
                max_purity = max(max_purity, Mij / ni)

            if max_purity < cfg.purity_thr:
                wrong_batch_num += 1

        err_rate = wrong_batch_num / total_batch_num if total_batch_num > 0 else 0.0
        err_rates.append(err_rate)

        beat_details.append({
            "Beat": beat_id,
            "NumPulses": len(beat_df),
            "NumPredBatches": total_batch_num,
            "NumTrueTargets": len(true_ids),
            "SuccessTargets": beat_success_targets,
            "AddCountTotal": beat_add_total,
            "WrongBatchNum": wrong_batch_num,
            "ErrRate": err_rate
        })

    sort_acc_list = []
    add_rate_list = []
    target_rows = []

    for j in all_targets:
        Tj = target_appear_beats[j]
        if Tj == 0:
            continue

        sort_acc_j = target_success_beats[j] / Tj
        add_rate_j = target_add_sum[j] / Tj

        sort_acc_list.append(sort_acc_j)
        add_rate_list.append(add_rate_j)

        target_rows.append({
            "TrueLabel": j,
            "AppearBeats": Tj,
            "SuccessBeats": target_success_beats[j],
            "AddSum": target_add_sum[j],
            "SortACC": sort_acc_j,
            "AddRate": add_rate_j
        })

    sort_acc = float(np.mean(sort_acc_list)) if len(sort_acc_list) > 0 else 0.0
    add_rate = float(np.mean(add_rate_list)) if len(add_rate_list) > 0 else 0.0
    err_rate = float(np.mean(err_rates)) if len(err_rates) > 0 else 0.0

    return {
        "Sort_ACC": sort_acc,
        "Add_Rate": add_rate,
        "Err_Rate": err_rate,
        "target_detail": pd.DataFrame(target_rows),
        "beat_detail": pd.DataFrame(beat_details)
    }


# ============================================================
# 4. 框架周期特征估计
# ============================================================

def dedup_periods(periods, tol):
    if len(periods) == 0:
        return []

    periods = sorted(periods)
    dedup = [periods[0]]

    for p in periods[1:]:
        if abs(p - dedup[-1]) > tol:
            dedup.append(p)

    return dedup


def sample_nonadjacent_period_candidates(toa, cfg: Config):
    """
    使用非相邻 TOA 差值采样构造 Tframe 候选。
    不使用相邻 dTOA 作为主候选，避免子 PRI 被误当成框架周期。
    """
    toa = np.asarray(toa, dtype=float)
    n = len(toa)

    if n < 2:
        return []

    total_span = toa[-1] - toa[0]

    if total_span < cfg.frame_T_min:
        return []

    max_base = min(cfg.num_base_samples, n - 1)
    base_indices = np.linspace(0, n - 2, max_base).astype(int)

    candidates = []

    for base_idx in base_indices:
        base_t = toa[base_idx]

        left_t = base_t + cfg.frame_T_min
        right_t = base_t + cfg.frame_T_max

        left_idx = np.searchsorted(toa, left_t, side="left")
        right_idx = np.searchsorted(toa, right_t, side="right")

        if right_idx <= left_idx:
            continue

        num_ref = min(cfg.num_ref_per_base, right_idx - left_idx)
        ref_indices = np.linspace(left_idx, right_idx - 1, num_ref).astype(int)

        diffs = toa[ref_indices] - base_t
        diffs = diffs[(diffs >= cfg.frame_T_min) & (diffs <= cfg.frame_T_max)]

        if len(diffs) > 0:
            candidates.extend(diffs.tolist())

    if len(candidates) == 0:
        return []

    candidates = np.asarray(candidates)

    # 直方图聚合，取高频候选
    bins = np.round(candidates / cfg.frame_candidate_bin).astype(np.int64)
    unique_bins, counts = np.unique(bins, return_counts=True)

    order = np.argsort(counts)[::-1]
    top_bins = unique_bins[order[:cfg.top_k_frame_candidates]]

    top_periods = []

    for b in top_bins:
        T = float(b * cfg.frame_candidate_bin)
        if cfg.frame_T_min <= T <= cfg.frame_T_max:
            top_periods.append(T)

    top_periods = dedup_periods(top_periods, cfg.frame_candidate_bin)

    return top_periods


def compute_one_step_hit_rate(toa, T, cfg: Config):
    """
    对候选 Tframe 做快速全时段验证。

    思想：
    若 T 是框架周期，则大量脉冲 t 应该能在 t + T 附近找到对应脉冲。
    因此采样若干脉冲，检查 t+T 是否命中。

    返回：
    hit_rate: 命中率
    span_rate: 命中样本覆盖的时间跨度比例
    support_rate: 有效样本中被 T 解释的比例
    confidence: 综合置信度
    valid_pairs: 有效预测对数量
    hit_count: 命中数量
    """
    toa = np.asarray(toa, dtype=float)
    n = len(toa)

    if n < 2 or T <= 0:
        return {
            "hit_rate": 0.0,
            "span_rate": 0.0,
            "support_rate": 0.0,
            "confidence": 0.0,
            "valid_pairs": 0,
            "hit_count": 0
        }

    # 只选择 t+T 还在观测范围内的样本
    max_t = toa[-1] - T
    valid_idx_all = np.where(toa <= max_t)[0]

    if len(valid_idx_all) == 0:
        return {
            "hit_rate": 0.0,
            "span_rate": 0.0,
            "support_rate": 0.0,
            "confidence": 0.0,
            "valid_pairs": 0,
            "hit_count": 0
        }

    if len(valid_idx_all) > cfg.max_score_points:
        sample_idx = np.linspace(0, len(valid_idx_all) - 1, cfg.max_score_points).astype(int)
        valid_idx = valid_idx_all[sample_idx]
    else:
        valid_idx = valid_idx_all

    sample_t = toa[valid_idx]
    targets = sample_t + T

    pos = np.searchsorted(toa, targets, side="left")

    hits = np.zeros(len(targets), dtype=bool)

    # 检查 pos
    valid_pos = pos < n
    if np.any(valid_pos):
        dist = np.abs(toa[pos[valid_pos]] - targets[valid_pos])
        hits[valid_pos] |= dist <= cfg.frame_toa_tol

    # 检查 pos - 1
    valid_pos_left = pos > 0
    if np.any(valid_pos_left):
        left_pos = pos[valid_pos_left] - 1
        dist_left = np.abs(toa[left_pos] - targets[valid_pos_left])
        hits[valid_pos_left] |= dist_left <= cfg.frame_toa_tol

    valid_pairs = len(targets)
    hit_count = int(hits.sum())

    hit_rate = hit_count / valid_pairs if valid_pairs > 0 else 0.0
    support_rate = hit_rate

    total_span = toa[-1] - toa[0]

    if hit_count > 1 and total_span > 0:
        hit_times = sample_t[hits]
        span_rate = (hit_times.max() - hit_times.min()) / total_span
    else:
        span_rate = 0.0

    # 综合置信度：既要求命中多，也要求覆盖时间跨度
    confidence = 0.70 * hit_rate + 0.30 * span_rate

    return {
        "hit_rate": float(hit_rate),
        "span_rate": float(span_rate),
        "support_rate": float(support_rate),
        "confidence": float(confidence),
        "valid_pairs": int(valid_pairs),
        "hit_count": int(hit_count)
    }


def estimate_tframe_one_sigidx(sigidx, df_sig, cfg: Config):
    """
    对单个 SigIdx 估计框架周期特征。
    不改变分选结果，只输出特征。
    """
    df_sig = df_sig.sort_values("TOA").reset_index(drop=True)
    toa = df_sig["TOA"].values
    n = len(toa)

    toa_start = float(toa[0]) if n > 0 else np.nan
    toa_end = float(toa[-1]) if n > 0 else np.nan
    span = toa_end - toa_start if n > 1 else 0.0

    base_row = {
        "SigIdx": int(sigidx),
        "NumPulses": int(n),
        "TOA_Start": toa_start,
        "TOA_End": toa_end,
        "Span": float(span),
        "Tframe": np.nan,
        "Tframe_Confidence": 0.0,
        "Hit_Rate": 0.0,
        "Span_Rate": 0.0,
        "Support_Rate": 0.0,
        "ValidPairs": 0,
        "HitCount": 0,
        "NumCandidates": 0,
        "IsValidTframe": False
    }

    if n < cfg.min_pulses_for_tframe:
        return base_row

    candidates = sample_nonadjacent_period_candidates(toa, cfg)
    base_row["NumCandidates"] = len(candidates)

    if len(candidates) == 0:
        return base_row

    best = None

    for T in candidates:
        stat = compute_one_step_hit_rate(toa, T, cfg)
        score = stat["confidence"]

        if best is None or score > best["score"]:
            best = {
                "T": T,
                "score": score,
                **stat
            }

    if best is None:
        return base_row

    is_valid = (
        best["hit_rate"] >= cfg.min_hit_rate_for_valid
        and best["span_rate"] >= cfg.min_span_rate_for_valid
        and best["confidence"] >= cfg.min_confidence_for_valid
    )

    base_row.update({
        "Tframe": float(best["T"]),
        "Tframe_Confidence": float(best["confidence"]),
        "Hit_Rate": float(best["hit_rate"]),
        "Span_Rate": float(best["span_rate"]),
        "Support_Rate": float(best["support_rate"]),
        "ValidPairs": int(best["valid_pairs"]),
        "HitCount": int(best["hit_count"]),
        "IsValidTframe": bool(is_valid)
    })

    return base_row


def estimate_tframe_for_all_sigidx(presort_df, cfg: Config):
    """
    实验2：
    为每个 SigIdx 估计框架周期特征。
    不改变 PredID。
    """
    rows = []

    groups = list(presort_df.groupby("SigIdx"))

    for sigidx, g in pbar(
        groups,
        desc="实验2：估计每个SigIdx的框架周期特征",
        unit="batch",
        leave=True,
        cfg=cfg
    ):
        row = estimate_tframe_one_sigidx(sigidx, g, cfg)
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    return feature_df


# ============================================================
# 5. 保守跨批合并
# ============================================================

class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def is_feature_merge_candidate(row, cfg: Config):
    """
    判断单个 SigIdx 是否具备参与跨批合并的资格。
    """
    if not bool(row["IsValidTframe"]):
        return False

    if not np.isfinite(row["Tframe"]):
        return False

    if row["Tframe_Confidence"] < cfg.merge_min_confidence:
        return False

    if row["Hit_Rate"] < cfg.merge_min_hit_rate:
        return False

    if row["Span_Rate"] < cfg.merge_min_span_rate:
        return False

    return True


def periods_close(T1, T2, cfg: Config):
    """
    严格判断两个 Tframe 是否接近。
    第一版默认不允许倍周期合并。
    """
    if not np.isfinite(T1) or not np.isfinite(T2):
        return False, None

    diff = abs(T1 - T2)
    tol = max(cfg.merge_abs_tol, cfg.merge_T_rel_tol * max(T1, T2))

    if diff <= tol:
        return True, 0.5 * (T1 + T2)

    if not cfg.allow_harmonic_merge:
        return False, None

    small = min(T1, T2)
    large = max(T1, T2)

    if small <= 0:
        return False, None

    ratio = large / small
    k = round(ratio)

    if 2 <= k <= cfg.merge_harmonic_order:
        residual = abs(large - k * small)
        tol_h = max(cfg.merge_abs_tol, cfg.merge_T_rel_tol * large)
        if residual <= tol_h:
            return True, small

    return False, None


def verify_merged_sigidx_period(sigidx_a, sigidx_b, common_T, sigidx_to_toa, cfg: Config):
    """
    合并前复验：
    将两个 SigIdx 的 TOA 合并后，用共同 Tframe 检查周期结构是否仍稳定。
    """
    toa_a = sigidx_to_toa[sigidx_a]
    toa_b = sigidx_to_toa[sigidx_b]

    toa = np.concatenate([toa_a, toa_b])
    toa = np.unique(np.sort(toa))

    stat = compute_one_step_hit_rate(toa, common_T, cfg)

    ok = (
        stat["hit_rate"] >= cfg.merge_verify_min_hit_rate
        and stat["span_rate"] >= cfg.merge_verify_min_span_rate
        and stat["confidence"] >= cfg.merge_verify_min_confidence
    )

    return ok, stat


def build_sigidx_to_toa(presort_df):
    """
    构造 SigIdx -> TOA 数组映射。
    """
    mapping = {}

    for sigidx, g in presort_df.groupby("SigIdx"):
        toa = g.sort_values("TOA")["TOA"].values.astype(float)
        mapping[int(sigidx)] = toa

    return mapping


def merge_sigidx_by_tframe(feature_df, presort_df, cfg: Config):
    """
    实验3：
    基于 SigIdx 的 Tframe 特征进行加强版跨批合并。

    相比旧版：
    旧版先按 T_Bin 分组，只比较相邻 bin，很多可合并批次没有进入比较；
    新版对所有有效 Tframe 批次两两比较，使相对误差阈值和倍周期合并真正生效。

    返回：
    sigidx_to_predid: dict
    merge_pairs: DataFrame
    """
    feature_df = feature_df.copy()

    all_sigidx = sorted(feature_df["SigIdx"].astype(int).tolist())
    uf = UnionFind(all_sigidx)

    sigidx_to_toa = build_sigidx_to_toa(presort_df)

    # 1. 筛选可参与合并的批次
    valid_df = feature_df[
        feature_df.apply(lambda r: is_feature_merge_candidate(r, cfg), axis=1)
    ].copy()

    if len(valid_df) == 0:
        sigidx_to_predid = {sigidx: idx for idx, sigidx in enumerate(all_sigidx)}
        return sigidx_to_predid, pd.DataFrame()

    valid_df = valid_df.sort_values("Tframe").reset_index(drop=True)

    merge_records = []

    n_valid = len(valid_df)

    # 2. 对所有有效批次两两比较
    for a in pbar(
        range(n_valid),
        desc="实验3：加强跨批合并",
        unit="batch",
        leave=True,
        cfg=cfg
    ):
        row_a = valid_df.iloc[a]
        sig_a = int(row_a["SigIdx"])

        for b in range(a + 1, n_valid):
            row_b = valid_df.iloc[b]
            sig_b = int(row_b["SigIdx"])

            if sig_a == sig_b:
                continue

            if uf.find(sig_a) == uf.find(sig_b):
                continue

            n_a = int(row_a["NumPulses"])
            n_b = int(row_b["NumPulses"])

            # 小批次过小不自动合并，防止被大批次吞并
            if min(n_a, n_b) / max(n_a, n_b) < cfg.merge_min_size_ratio:
                continue

            # 3. 判断周期是否接近或是否存在倍周期关系
            close, common_T = periods_close(row_a["Tframe"], row_b["Tframe"], cfg)

            if not close:
                continue

            # 4. 合并前复验
            verify_ok = True
            verify_stat = {
                "hit_rate": np.nan,
                "span_rate": np.nan,
                "confidence": np.nan
            }

            if cfg.merge_verify:
                verify_ok, verify_stat = verify_merged_sigidx_period(
                    sig_a,
                    sig_b,
                    common_T,
                    sigidx_to_toa,
                    cfg
                )

            if not verify_ok:
                continue

            # 5. 合并
            uf.union(sig_a, sig_b)

            merge_records.append({
                "SigIdx_A": sig_a,
                "SigIdx_B": sig_b,
                "T_A": row_a["Tframe"],
                "T_B": row_b["Tframe"],
                "Common_T": common_T,
                "Conf_A": row_a["Tframe_Confidence"],
                "Conf_B": row_b["Tframe_Confidence"],
                "Hit_A": row_a["Hit_Rate"],
                "Hit_B": row_b["Hit_Rate"],
                "Span_A": row_a["Span_Rate"],
                "Span_B": row_b["Span_Rate"],
                "Verify_Hit": verify_stat["hit_rate"],
                "Verify_Span": verify_stat["span_rate"],
                "Verify_Conf": verify_stat["confidence"],
                "Is_Harmonic": (
                    abs(max(row_a["Tframe"], row_b["Tframe"]) /
                        min(row_a["Tframe"], row_b["Tframe"]) -
                        round(max(row_a["Tframe"], row_b["Tframe"]) /
                              min(row_a["Tframe"], row_b["Tframe"]))) < cfg.merge_T_rel_tol
                    if min(row_a["Tframe"], row_b["Tframe"]) > 0 else False
                )
            })

    # 6. 生成最终 PredID 映射
    root_to_predid = {}
    sigidx_to_predid = {}
    next_id = 0

    for sigidx in all_sigidx:
        root = uf.find(sigidx)
        if root not in root_to_predid:
            root_to_predid[root] = next_id
            next_id += 1
        sigidx_to_predid[sigidx] = root_to_predid[root]

    merge_pairs = pd.DataFrame(merge_records)
    return sigidx_to_predid, merge_pairs


def apply_sigidx_mapping(presort_df, sigidx_to_predid):
    """
    根据 SigIdx -> PredID 映射生成预测结果。
    """
    pred_df = presort_df.copy()
    pred_df["PredID"] = pred_df["SigIdx"].map(sigidx_to_predid)

    # 理论上不应该有空值，兜底处理
    missing = pred_df["PredID"].isna()
    if missing.any():
        max_id = int(np.nanmax(list(sigidx_to_predid.values()))) if len(sigidx_to_predid) > 0 else 0
        missing_sigidx = sorted(pred_df.loc[missing, "SigIdx"].unique())
        extra_map = {sig: max_id + 1 + i for i, sig in enumerate(missing_sigidx)}
        pred_df.loc[missing, "PredID"] = pred_df.loc[missing, "SigIdx"].map(extra_map)

    pred_df["PredID"] = pred_df["PredID"].astype(int)
    return pred_df


def build_mapping_df(sigidx_to_predid):
    rows = []

    for sigidx, predid in sorted(sigidx_to_predid.items(), key=lambda x: x[0]):
        rows.append({
            "SigIdx": sigidx,
            "PredID": predid
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. 三个实验统一运行
# ============================================================

def run_all_experiments(
    presort_path,
    truth_path,
    output_dir,
    presort_toa_col="TOA(s)",
    presort_sigidx_col="SigIdx",
    truth_toa_col="TOA(s)",
    truth_label_col="SigIdx",
    prefer_pulse_id=False,
    cfg=None
):
    if cfg is None:
        cfg = Config()

    os.makedirs(output_dir, exist_ok=True)

    print("\n========== 读取数据 ==========")
    presort_raw = read_table_auto(presort_path)
    truth_raw = read_table_auto(truth_path)

    print("预分选表列名：", list(presort_raw.columns))
    print("验证表列名：", list(truth_raw.columns))

    presort_df = normalize_presort_df(
        presort_raw,
        toa_col=presort_toa_col,
        sigidx_col=presort_sigidx_col
    )

    truth_df = normalize_truth_df(
        truth_raw,
        toa_col=truth_toa_col,
        true_col=truth_label_col
    )

    print("预分选脉冲数：", len(presort_df))
    print("验证标签脉冲数：", len(truth_df))
    print("预分选批次数：", presort_df["SigIdx"].nunique())
    print("真实目标数：", truth_df["TrueLabel"].nunique())

    # ========================================================
    # 实验1：只看预分选
    # ========================================================
    print("\n========== 实验1：只看预分选结果 ==========")

    exp1_pred = presort_df.copy()
    exp1_pred["PredID"] = exp1_pred["SigIdx"].astype(int)

    exp1_eval = align_pred_and_truth(
        exp1_pred,
        truth_df,
        cfg,
        prefer_pulse_id=prefer_pulse_id
    )

    exp1_metrics = evaluate_sorting_official_like(
        exp1_eval,
        cfg,
        pred_col="PredID",
        true_col="TrueLabel",
        toa_col="TOA",
        exp_name="实验1"
    )

    # ========================================================
    # 实验2：只估计框架周期特征，不改变 PredID
    # ========================================================
    print("\n========== 实验2：估计每个 SigIdx 的框架周期特征，不改变 PredID ==========")

    feature_df = estimate_tframe_for_all_sigidx(presort_df, cfg)

    exp2_pred = presort_df.copy()
    exp2_pred["PredID"] = exp2_pred["SigIdx"].astype(int)

    exp2_eval = align_pred_and_truth(
        exp2_pred,
        truth_df,
        cfg,
        prefer_pulse_id=prefer_pulse_id
    )

    exp2_metrics = evaluate_sorting_official_like(
        exp2_eval,
        cfg,
        pred_col="PredID",
        true_col="TrueLabel",
        toa_col="TOA",
        exp_name="实验2"
    )

    # ========================================================
    # 实验3：保守跨批合并
    # ========================================================
    print("\n========== 实验3：基于框架周期特征的保守跨批合并 ==========")

    sigidx_to_predid, merge_pairs = merge_sigidx_by_tframe(
        feature_df,
        presort_df,
        cfg
    )

    exp3_pred = apply_sigidx_mapping(presort_df, sigidx_to_predid)

    exp3_eval = align_pred_and_truth(
        exp3_pred,
        truth_df,
        cfg,
        prefer_pulse_id=prefer_pulse_id
    )

    exp3_metrics = evaluate_sorting_official_like(
        exp3_eval,
        cfg,
        pred_col="PredID",
        true_col="TrueLabel",
        toa_col="TOA",
        exp_name="实验3"
    )

    # ========================================================
    # 汇总与保存
    # ========================================================
    summary = pd.DataFrame([
        {
            "Experiment": "Exp1_PreSort_Only",
            "Sort_ACC": exp1_metrics["Sort_ACC"],
            "Add_Rate": exp1_metrics["Add_Rate"],
            "Err_Rate": exp1_metrics["Err_Rate"]
        },
        {
            "Experiment": "Exp2_Tframe_Feature_Only",
            "Sort_ACC": exp2_metrics["Sort_ACC"],
            "Add_Rate": exp2_metrics["Add_Rate"],
            "Err_Rate": exp2_metrics["Err_Rate"]
        },
        {
            "Experiment": "Exp3_Conservative_Merge",
            "Sort_ACC": exp3_metrics["Sort_ACC"],
            "Add_Rate": exp3_metrics["Add_Rate"],
            "Err_Rate": exp3_metrics["Err_Rate"]
        }
    ])

    summary_path = os.path.join(output_dir, "experiment_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    exp1_pred.to_csv(os.path.join(output_dir, "exp1_pred_presort_only.csv"), index=False, encoding="utf-8-sig")
    exp2_pred.to_csv(os.path.join(output_dir, "exp2_pred_tframe_feature_only.csv"), index=False, encoding="utf-8-sig")
    exp3_pred.to_csv(os.path.join(output_dir, "exp3_pred_conservative_merge.csv"), index=False, encoding="utf-8-sig")

    feature_df.to_csv(os.path.join(output_dir, "sigidx_tframe_features.csv"), index=False, encoding="utf-8-sig")
    merge_pairs.to_csv(os.path.join(output_dir, "exp3_merge_pairs.csv"), index=False, encoding="utf-8-sig")
    build_mapping_df(sigidx_to_predid).to_csv(os.path.join(output_dir, "exp3_sigidx_to_predid.csv"), index=False, encoding="utf-8-sig")

    exp1_metrics["target_detail"].to_csv(os.path.join(output_dir, "exp1_target_detail.csv"), index=False, encoding="utf-8-sig")
    exp2_metrics["target_detail"].to_csv(os.path.join(output_dir, "exp2_target_detail.csv"), index=False, encoding="utf-8-sig")
    exp3_metrics["target_detail"].to_csv(os.path.join(output_dir, "exp3_target_detail.csv"), index=False, encoding="utf-8-sig")

    exp1_metrics["beat_detail"].to_csv(os.path.join(output_dir, "exp1_beat_detail.csv"), index=False, encoding="utf-8-sig")
    exp2_metrics["beat_detail"].to_csv(os.path.join(output_dir, "exp2_beat_detail.csv"), index=False, encoding="utf-8-sig")
    exp3_metrics["beat_detail"].to_csv(os.path.join(output_dir, "exp3_beat_detail.csv"), index=False, encoding="utf-8-sig")

    print("\n========== 实验结果汇总 ==========")
    print(summary)

    print("\n百分比形式：")
    summary_percent = summary.copy()
    for col in ["Sort_ACC", "Err_Rate"]:
        summary_percent[col] = summary_percent[col] * 100
    print(summary_percent)

    print("\n有效 Tframe 批次数：", int(feature_df["IsValidTframe"].sum()))
    print("发生合并的 SigIdx 对数：", len(merge_pairs))

    print("\n结果已保存到：", output_dir)
    print("汇总文件：", summary_path)

    return {
        "summary": summary,
        "feature_df": feature_df,
        "merge_pairs": merge_pairs,
        "exp1_pred": exp1_pred,
        "exp2_pred": exp2_pred,
        "exp3_pred": exp3_pred,
        "exp1_metrics": exp1_metrics,
        "exp2_metrics": exp2_metrics,
        "exp3_metrics": exp3_metrics
    }


# ============================================================
# 7. 主程序入口
# ============================================================

if __name__ == "__main__":
    cfg = Config()

    # ========================================================
    # 改这里：文件路径
    # ========================================================

    presort_path = r"D:\apengpeng\.model_runs\hdbscan_sort_20260612_193751_903c724e\outputs_best_front_tracklet_graph\ui_sample\ui_sample_sort_before_tracklet_graph.txt"
    truth_path = r"D:\edata\Test_Data\Sample_1\Sorted_PDW.txt"
    output_dir = r"D:\apengpeng\分选模型\WU\result"

    # ========================================================
    # 列名设置
    # ========================================================

    presort_toa_col = "TOA(s)"
    presort_sigidx_col = "SigIdx"

    truth_toa_col = "TOA(s)"
    truth_label_col = "SigIdx"

    prefer_pulse_id = False

    # ========================================================
    # 评价参数
    # ========================================================

    cfg.show_progress = True
    cfg.beat_len = 0.2
    cfg.purity_thr = 0.9
    cfg.cover_thr = 0.1
    cfg.toa_match_tolerance = 1e-9

    # ========================================================
    # 框架周期估计参数
    # ========================================================

    # 注意：这是框架周期范围，不是子 PRI 范围
    cfg.frame_T_min = 50e-6
    cfg.frame_T_max = 5e-3

    # 如果仍然出现子 PRI 被选中，可以把 frame_T_min 提高到 100e-6
    # cfg.frame_T_min = 100e-6

    cfg.frame_candidate_bin = 0.5e-6

    cfg.num_base_samples = 80
    cfg.num_ref_per_base = 500
    cfg.top_k_frame_candidates = 10

    cfg.frame_toa_tol = 3e-6
    cfg.max_score_points = 8000

    cfg.min_pulses_for_tframe = 10

    cfg.min_hit_rate_for_valid = 0.45
    cfg.min_span_rate_for_valid = 0.40
    cfg.min_confidence_for_valid = 0.40

    # ========================================================
    # 加强跨批合并参数：新版两两比较 merge_sigidx_by_tframe
    # ========================================================

    cfg.merge_min_confidence = 0.55
    cfg.merge_min_hit_rate = 0.45
    cfg.merge_min_span_rate = 0.30

    cfg.merge_T_rel_tol = 0.022
    cfg.merge_abs_tol = 3.0e-6

    cfg.allow_harmonic_merge = False

    cfg.merge_verify = True
    cfg.merge_verify_min_hit_rate = 0.38
    cfg.merge_verify_min_span_rate = 0.25
    cfg.merge_verify_min_confidence = 0.34

    cfg.merge_min_size_ratio = 0.002

    # 新版两两比较函数不再依赖这个参数，保留也不影响
    cfg.max_compare_per_bin = 500

    results = run_all_experiments(
        presort_path=presort_path,
        truth_path=truth_path,
        output_dir=output_dir,
        presort_toa_col=presort_toa_col,
        presort_sigidx_col=presort_sigidx_col,
        truth_toa_col=truth_toa_col,
        truth_label_col=truth_label_col,
        prefer_pulse_id=prefer_pulse_id,
        cfg=cfg
    )
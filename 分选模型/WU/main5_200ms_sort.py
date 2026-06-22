# -*- coding: utf-8 -*-
"""
200 ms beat-wise version of the main.py workflow.

This script follows the method in "基于循环周期识别的信号主分选.docx":

1. Keep the raw pre-sort SigIdx as Exp1.
2. Estimate a Tframe feature for each SigIdx inside each 200 ms beat as Exp2.
3. Merge SigIdx fragments by Tframe consistency as Exp3.
4. Absorb small fragments into larger components with PDW parameter guards as Exp4.

Repeated local SigIdx merge evidence is converted into stable global labels
only when the pair is supported by enough beats.
"""

import os
import time
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


@dataclass
class LocalConfig:
    show_progress: bool = True

    beat_len: float = 0.2
    purity_thr: float = 0.9
    cover_thr: float = 0.1
    swallow_count_thr: int = 150
    toa_match_tolerance: float = 1e-9

    frame_T_min: float = 1e-6
    frame_T_max: float = 5e-3
    frame_candidate_bin: float = 0.5e-6
    num_base_samples: int = 80
    num_ref_per_base: int = 500
    top_k_frame_candidates: int = 14
    frame_toa_tol: float = 3e-6
    max_score_points: int = 2500
    min_pulses_for_tframe: int = 10
    min_hit_rate_for_valid: float = 0.45
    min_span_rate_for_valid: float = 0.40
    min_confidence_for_valid: float = 0.40

    merge_min_confidence: float = 0.75
    merge_min_hit_rate: float = 0.65
    merge_min_span_rate: float = 0.50
    merge_T_rel_tol: float = 0.005
    merge_abs_tol: float = 0.5e-6
    allow_harmonic_merge: bool = False
    merge_harmonic_order: int = 4
    merge_verify: bool = True
    merge_verify_min_hit_rate: float = 0.60
    merge_verify_min_span_rate: float = 0.45
    merge_verify_min_confidence: float = 0.55
    merge_min_size_ratio: float = 0.020

    enable_param_absorb: bool = True
    pdw_toa_match_tolerance: float = 1e-9
    pdw_param_cols: tuple = ("Param1", "Param2", "Param3", "Param4", "Param5")

    # Conservative 200 ms version of main.py Exp4.
    param_absorb_small_max_pulses: int = 40
    param_absorb_anchor_min_pulses: int = 160
    param_absorb_min_score: float = 5.1
    param_absorb_max_size_ratio: float = 0.08
    param_absorb_period_small_max_pulses: int = 10
    param_absorb_p5_hard_tol: float = 0.40
    param_absorb_period_rel_tol: float = 0.04
    param_absorb_period_abs_tol: float = 5e-6
    param_absorb_harmonic_order: int = 16
    param_absorb_top_small_modes: int = 8
    param_absorb_top_anchor_modes: int = 12


def pbar(iterable, desc="", unit="", leave=True, total=None, cfg=None):
    if cfg is not None and not cfg.show_progress:
        return iterable
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit, leave=leave, total=total)


class UnionFind:
    def __init__(self, items):
        self.parent = {int(x): int(x) for x in items}

    def find(self, x):
        x = int(x)
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def read_table_auto(path):
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    except Exception:
        df = pd.read_csv(path, sep=",", engine="python")

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df.dropna(axis=1, how="all")


def normalize_truth_df(df, toa_col="TOA(s)", true_col="SigIdx"):
    df = df.copy()
    if toa_col not in df.columns:
        raise ValueError(f"Truth file has no TOA column: {toa_col}")
    if true_col not in df.columns:
        raise ValueError(f"Truth file has no label column: {true_col}")

    df = df.rename(columns={toa_col: "TOA", true_col: "TrueLabel"})
    df["TOA"] = df["TOA"].astype(float)
    df["TrueLabel"] = df["TrueLabel"].astype(int)
    if "PulseID" not in df.columns:
        df["PulseID"] = np.arange(len(df), dtype=np.int64)
    return df.sort_values("TOA").reset_index(drop=True)


def align_pred_and_truth(pred_df, truth_df, cfg: LocalConfig, prefer_pulse_id=False):
    pred_df = pred_df.copy()
    truth_df = truth_df.copy()

    if prefer_pulse_id and "PulseID" in pred_df.columns and "PulseID" in truth_df.columns:
        return pred_df.merge(truth_df[["PulseID", "TrueLabel"]], on="PulseID", how="inner")

    pred_df = pred_df.sort_values("TOA").reset_index(drop=True)
    truth_df = truth_df.sort_values("TOA").reset_index(drop=True)
    eval_df = pd.merge_asof(
        pred_df,
        truth_df[["TOA", "TrueLabel"]],
        on="TOA",
        direction="nearest",
        tolerance=cfg.toa_match_tolerance,
    )
    eval_df = eval_df.dropna(subset=["TrueLabel"]).copy()
    eval_df["TrueLabel"] = eval_df["TrueLabel"].astype(int)
    return eval_df


def evaluate_sorting_official_like(df, cfg: LocalConfig,
                                   pred_col="PredID", true_col="TrueLabel",
                                   toa_col="TOA", exp_name="Eval"):
    data = df.dropna(subset=[pred_col, true_col, toa_col]).copy()
    data = data.sort_values(toa_col).reset_index(drop=True)
    if len(data) == 0:
        return {
            "Sort_ACC": 0.0,
            "Add_Rate": 0.0,
            "Err_Rate": 0.0,
            "target_detail": pd.DataFrame(),
            "beat_detail": pd.DataFrame(),
        }

    toa_start = data[toa_col].min()
    data["Beat"] = np.floor((data[toa_col] - toa_start) / cfg.beat_len).astype(int)
    all_targets = sorted(data[true_col].unique())

    target_appear_beats = {j: 0 for j in all_targets}
    target_success_beats = {j: 0 for j in all_targets}
    target_add_sum = {j: 0 for j in all_targets}
    beat_details = []
    err_rates = []

    for beat_id, beat_df in pbar(
        list(data.groupby("Beat")),
        desc=f"{exp_name}: eval 200ms beats",
        unit="beat",
        leave=False,
        cfg=cfg,
    ):
        table = pd.crosstab(beat_df[pred_col], beat_df[true_col])
        pred_ids = table.index.tolist()
        true_ids = table.columns.tolist()
        pred_counts = table.sum(axis=1)
        true_counts = table.sum(axis=0)

        swallow_wrong_pred_ids = set()
        for i in pred_ids:
            ni = pred_counts[i]
            if ni <= 0:
                continue
            row = table.loc[i]
            main_j = row.idxmax()
            main_count = row.max()
            if main_count / ni >= cfg.purity_thr:
                for k in true_ids:
                    if k == main_j:
                        continue
                    Nk = true_counts[k]
                    Mik = table.loc[i, k]
                    if Nk > cfg.swallow_count_thr and Mik == Nk:
                        swallow_wrong_pred_ids.add(i)

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

            if valid_count >= 1:
                target_success_beats[j] += 1
                beat_success_targets += 1
            matched_batch_count = valid_count + small_count
            add_count = max(matched_batch_count - 1, 0)
            target_add_sum[j] += add_count
            beat_add_total += add_count

        wrong_batch_num = 0
        total_batch_num = len(pred_ids)
        for i in pred_ids:
            if i in swallow_wrong_pred_ids:
                wrong_batch_num += 1
                continue
            ni = pred_counts[i]
            max_purity = 0.0
            for j in true_ids:
                max_purity = max(max_purity, table.loc[i, j] / ni)
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
            "ErrRate": err_rate,
        })

    sort_acc_list = []
    add_rate_list = []
    target_rows = []
    for j in all_targets:
        Tj = target_appear_beats[j]
        if Tj == 0:
            continue
        sort_acc_j = target_success_beats[j] / Tj
        add_rate_j = target_add_sum[j] / Tj * 100.0
        sort_acc_list.append(sort_acc_j)
        add_rate_list.append(add_rate_j)
        target_rows.append({
            "TrueLabel": j,
            "AppearBeats": Tj,
            "SuccessBeats": target_success_beats[j],
            "AddSum": target_add_sum[j],
            "SortACC": sort_acc_j,
            "AddRate": add_rate_j,
        })

    return {
        "Sort_ACC": float(np.mean(sort_acc_list)) if sort_acc_list else 0.0,
        "Add_Rate": float(np.mean(add_rate_list)) if add_rate_list else 0.0,
        "Err_Rate": float(np.mean(err_rates)) if err_rates else 0.0,
        "target_detail": pd.DataFrame(target_rows),
        "beat_detail": pd.DataFrame(beat_details),
    }


ABSORB_PARAM_WEIGHTS = {
    "Param1": 0.8,
    "Param2": 1.0,
    "Param3": 1.5,
    "Param4": 0.7,
    "Param5": 3.0,
}

ABSORB_PARAM_TOLS = {
    "Param1": 350.0,
    "Param2": 2.0,
    "Param3": 1.2,
    "Param4": 18.0,
    "Param5": 0.9,
}


@dataclass
class MainSortConfig:
    input_dir: str
    truth_path: str
    output_dir: str

    toa_col: str = "TOA(s)"
    sigidx_col: str = "SigIdx"
    truth_toa_col: str = "TOA(s)"
    truth_label_col: str = "SigIdx"

    beat_len: float = 0.2
    predid_offset_per_beat: int = 100000
    show_progress: bool = True
    param_cols: tuple = ("Param1", "Param2", "Param3", "Param4", "Param5")
    annotated_output_dir: str = ""
    annotated_label_col: str = "OurPredID"
    save_full_predictions: bool = True
    write_annotated_outputs: bool = True
    stream_output_per_beat: bool = False
    stream_output_dir: str = ""
    stream_watch_forever: bool = False
    stream_poll_seconds: float = 0.2
    stream_file_stable_seconds: float = 0.2
    stream_idle_timeout_seconds: float = 30.0
    stream_skip_existing_outputs: bool = True

    # Candidate cycle-period search in a short 200 ms beat.
    cycle_T_min: float = 1e-6
    cycle_T_max: float = 5e-3
    cycle_bin: float = 0.5e-6
    top_k_cycle_candidates: int = 14
    min_pulses_for_cycle: int = 10
    chain_toa_tol: float = 3e-6
    max_chain_score_points: int = 2500
    chain_min_len: int = 4
    use_short_window_toa_candidates: bool = True
    adjacent_candidate_bin: float = 0.5e-6
    adjacent_top_k: int = 8
    all_diff_max_pulses: int = 300
    all_diff_top_k: int = 8

    # Fragment merge gates.
    cycle_abs_tol: float = 4e-6
    cycle_rel_tol: float = 0.040
    harmonic_order: int = 16
    p5_hard_tol: float = 0.85
    min_merge_size_ratio: float = 0.001
    max_pair_candidates_per_beat: int = 900
    global_pair_min_support_beats: int = 5


def list_beat_files(input_dir):
    files = sorted(Path(input_dir).glob("beat_*.txt"))
    if not files:
        raise FileNotFoundError(f"No beat_*.txt found in {input_dir}")
    return files


def beat_predid(beat_id, local_predid, cfg: MainSortConfig):
    return int(beat_id) * int(cfg.predid_offset_per_beat) + int(local_predid)


class DynamicUnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        x = int(x)
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        x = int(x)
        self.add(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def read_beat(path, beat_id, cfg: MainSortConfig):
    if Path(path).stat().st_size == 0:
        return pd.DataFrame()

    raw = pd.read_csv(path, sep=r"\s+", engine="python")
    if raw.empty:
        return pd.DataFrame()

    df = raw.rename(columns={cfg.toa_col: "TOA", cfg.sigidx_col: "SigIdx"}).copy()
    if "TOA" not in df.columns or "SigIdx" not in df.columns:
        raise ValueError(f"{path} must contain {cfg.toa_col} and {cfg.sigidx_col}")

    df["TOA"] = pd.to_numeric(df["TOA"], errors="coerce")
    df["SigIdx"] = pd.to_numeric(df["SigIdx"], errors="coerce")
    df = df.dropna(subset=["TOA", "SigIdx"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["SigIdx"] = df["SigIdx"].astype(int)
    df["Beat"] = int(beat_id)
    return df.sort_values("TOA").reset_index(drop=True)


def build_main2_eval_cfg(cfg: MainSortConfig):
    return LocalConfig(
        show_progress=False,
        beat_len=cfg.beat_len,
        purity_thr=0.9,
        cover_thr=0.1,
        toa_match_tolerance=1e-9,
    )


def build_candidate_cfg(cfg: MainSortConfig):
    mcfg = LocalConfig(show_progress=False)
    mcfg.frame_T_min = cfg.cycle_T_min
    mcfg.frame_T_max = cfg.cycle_T_max
    mcfg.frame_candidate_bin = cfg.cycle_bin
    mcfg.top_k_frame_candidates = cfg.top_k_cycle_candidates
    mcfg.min_pulses_for_tframe = cfg.min_pulses_for_cycle
    mcfg.frame_toa_tol = cfg.chain_toa_tol
    mcfg.max_score_points = cfg.max_chain_score_points
    mcfg.pdw_param_cols = cfg.param_cols
    return mcfg


def base_cycle_merge(beat_df, beat_id, cfg: MainSortConfig, mcfg: LocalConfig):
    feature_df = estimate_tframe_for_beat_short(beat_df, cfg, mcfg)
    sigidx_to_predid, merge_pairs = merge_sigidx_by_tframe(feature_df, beat_df, mcfg)
    feature_df["Beat"] = int(beat_id)
    if len(merge_pairs) > 0:
        merge_pairs = merge_pairs.copy()
        merge_pairs["Beat"] = int(beat_id)
    return sigidx_to_predid, feature_df, merge_pairs


def raw_mapping(beat_df):
    return {int(x): int(x) for x in sorted(beat_df["SigIdx"].astype(int).unique())}


def make_prediction(beat_df, sigidx_to_local_predid, beat_id, cfg: MainSortConfig):
    pred = beat_df[["TOA", "Beat", "SigIdx"]].copy()
    pred["LocalPredID"] = pred["SigIdx"].astype(int).map(sigidx_to_local_predid).astype(int)
    pred["PredID"] = pred["LocalPredID"].map(lambda x: beat_predid(beat_id, x, cfg))
    return pred


def make_global_sigidx_prediction(df, sigidx_to_global_predid):
    pred = df[["TOA", "Beat", "SigIdx"]].copy()
    pred["PredID"] = pred["SigIdx"].astype(int).map(sigidx_to_global_predid).astype(int)
    return pred


def write_annotated_beat_outputs(beat_files, sigidx_to_predid, cfg: MainSortConfig,
                                 output_dir, pred_col="OurPredID"):
    os.makedirs(output_dir, exist_ok=True)
    if Path(output_dir).resolve() == Path(cfg.input_dir).resolve():
        raise ValueError("Annotated output directory must differ from input directory")

    written_files = 0
    written_rows = 0
    missing_labels = 0

    for path in beat_files:
        output_path = Path(output_dir) / path.name
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with Path(path).open("r", encoding="utf-8-sig", newline="") as src:
            with temp_path.open("w", encoding="utf-8-sig", newline="") as dst:
                header = src.readline()
                if header == "":
                    dst.write(f"{pred_col}\n")
                else:
                    header_text = header.rstrip("\r\n")
                    columns = header_text.split()
                    if cfg.sigidx_col not in columns:
                        raise ValueError(f"{path} must contain {cfg.sigidx_col}")
                    sigidx_position = columns.index(cfg.sigidx_col)
                    dst.write(f"{header_text}\t{pred_col}\n")

                    for line in src:
                        line_text = line.rstrip("\r\n")
                        if line_text.strip() == "":
                            dst.write(f"{line_text}\n")
                            continue

                        fields = line_text.split()
                        predid = -1
                        if len(fields) > sigidx_position:
                            try:
                                sigidx = int(float(fields[sigidx_position]))
                                predid = int(sigidx_to_predid[sigidx])
                            except (KeyError, TypeError, ValueError):
                                missing_labels += 1
                        else:
                            missing_labels += 1

                        dst.write(f"{line_text}\t{predid}\n")
                        written_rows += 1

        os.replace(temp_path, output_path)

        written_files += 1

    return {
        "output_dir": str(output_dir),
        "written_files": int(written_files),
        "written_rows": int(written_rows),
        "missing_labels": int(missing_labels),
    }


def collect_mapping_groups(sigidx_to_group, beat_id, experiment):
    records = []
    group_to_sigidx = {}
    for sigidx, group_id in sigidx_to_group.items():
        sigidx = int(sigidx)
        group_id = int(group_id)
        group_to_sigidx.setdefault(group_id, []).append(sigidx)

    for group_id, sigs in group_to_sigidx.items():
        sigs = sorted(set(sigs))
        if len(sigs) <= 1:
            continue
        records.append({
            "Beat": int(beat_id),
            "Experiment": experiment,
            "LocalGroupID": int(group_id),
            "NumSigIdx": int(len(sigs)),
            "SigIdxList": ";".join(str(x) for x in sigs),
        })
    return records


def build_pair_support(groups_df):
    pair_beats = {}
    if groups_df is None or len(groups_df) == 0:
        return pd.DataFrame(columns=["SigIdxA", "SigIdxB", "SupportBeats", "BeatList"])

    for _, row in groups_df.iterrows():
        sigs = [
            int(x)
            for x in str(row["SigIdxList"]).split(";")
            if str(x).strip() != ""
        ]
        beat = int(row["Beat"])
        for a, b in combinations(sorted(set(sigs)), 2):
            pair_beats.setdefault((a, b), set()).add(beat)

    rows = []
    for (a, b), beats in sorted(pair_beats.items()):
        rows.append({
            "SigIdxA": int(a),
            "SigIdxB": int(b),
            "SupportBeats": int(len(beats)),
            "BeatList": ";".join(str(x) for x in sorted(beats)),
        })
    return pd.DataFrame(rows)


def build_global_sigidx_mapping(all_sigidx, pair_support_df, cfg: MainSortConfig):
    global_uf = DynamicUnionFind()
    for sigidx in all_sigidx:
        global_uf.add(sigidx)

    if pair_support_df is not None and len(pair_support_df) > 0:
        supported = pair_support_df[
            pair_support_df["SupportBeats"].astype(int) >= cfg.global_pair_min_support_beats
        ]
        for _, row in supported.iterrows():
            global_uf.union(int(row["SigIdxA"]), int(row["SigIdxB"]))

    root_to_predid = {}
    mapping = {}
    next_id = 0
    for sigidx in sorted(global_uf.parent):
        root = global_uf.find(sigidx)
        if root not in root_to_predid:
            root_to_predid[root] = next_id
            next_id += 1
        mapping[int(sigidx)] = int(root_to_predid[root])
    return mapping


def compact_mapping(sigidx_to_component):
    root_to_local = {}
    out = {}
    next_id = 0
    for sigidx in sorted(sigidx_to_component):
        comp = int(sigidx_to_component[sigidx])
        if comp not in root_to_local:
            root_to_local[comp] = next_id
            next_id += 1
        out[int(sigidx)] = int(root_to_local[comp])
    return out


def dedup_periods(periods, tol):
    periods = sorted(float(p) for p in periods if np.isfinite(p) and p > 0)
    if not periods:
        return []

    dedup = [periods[0]]
    for p in periods[1:]:
        if abs(p - dedup[-1]) > tol:
            dedup.append(p)
    return dedup


def sample_nonadjacent_period_candidates(toa, cfg: LocalConfig):
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

    if not candidates:
        return []

    bins = np.round(np.asarray(candidates) / cfg.frame_candidate_bin).astype(np.int64)
    unique_bins, counts = np.unique(bins, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_bins = unique_bins[order[:cfg.top_k_frame_candidates]]

    periods = []
    for b in top_bins:
        T = float(b * cfg.frame_candidate_bin)
        if cfg.frame_T_min <= T <= cfg.frame_T_max:
            periods.append(T)
    return dedup_periods(periods, cfg.frame_candidate_bin)


def adjacent_period_candidates(toa, cfg: MainSortConfig, mcfg: LocalConfig):
    toa = np.asarray(toa, dtype=float)
    if len(toa) < 2:
        return []

    diffs = np.diff(np.sort(toa))
    diffs = diffs[(diffs >= mcfg.frame_T_min) & (diffs <= mcfg.frame_T_max)]
    if len(diffs) == 0:
        return []

    bins = np.round(diffs / cfg.adjacent_candidate_bin).astype(np.int64)
    unique_bins, counts = np.unique(bins, return_counts=True)
    order = np.argsort(counts)[::-1]

    periods = []
    for b in unique_bins[order[:cfg.adjacent_top_k]]:
        T = float(b * cfg.adjacent_candidate_bin)
        if mcfg.frame_T_min <= T <= mcfg.frame_T_max:
            periods.append(T)
    return periods


def all_diff_period_candidates(toa, cfg: MainSortConfig, mcfg: LocalConfig):
    toa = np.sort(np.asarray(toa, dtype=float))
    n = len(toa)
    if n < 3 or n > cfg.all_diff_max_pulses:
        return []

    diffs = []
    for i in range(n - 1):
        d = toa[i + 1:] - toa[i]
        d = d[(d >= mcfg.frame_T_min) & (d <= mcfg.frame_T_max)]
        if len(d) > 0:
            diffs.append(d)

    if not diffs:
        return []

    diffs = np.concatenate(diffs)
    bins = np.round(diffs / mcfg.frame_candidate_bin).astype(np.int64)
    unique_bins, counts = np.unique(bins, return_counts=True)
    order = np.argsort(counts)[::-1]

    periods = []
    for b in unique_bins[order[:cfg.all_diff_top_k]]:
        T = float(b * mcfg.frame_candidate_bin)
        if mcfg.frame_T_min <= T <= mcfg.frame_T_max:
            periods.append(T)
    return periods


def short_window_period_candidates(toa, cfg: MainSortConfig, mcfg: LocalConfig):
    candidates = []
    candidates.extend(sample_nonadjacent_period_candidates(toa, mcfg))
    if cfg.use_short_window_toa_candidates:
        candidates.extend(adjacent_period_candidates(toa, cfg, mcfg))
        candidates.extend(all_diff_period_candidates(toa, cfg, mcfg))
    return dedup_periods(candidates, mcfg.frame_candidate_bin)


def cycle_candidates(toa, cfg: MainSortConfig, mcfg: LocalConfig):
    toa = np.asarray(toa, dtype=float)
    if len(toa) < 2:
        return []

    return short_window_period_candidates(toa, cfg, mcfg)


def compute_one_step_hit_rate(toa, T, cfg: LocalConfig):
    toa = np.asarray(toa, dtype=float)
    n = len(toa)
    if n < 2 or T <= 0:
        return {
            "hit_rate": 0.0,
            "span_rate": 0.0,
            "support_rate": 0.0,
            "confidence": 0.0,
            "valid_pairs": 0,
            "hit_count": 0,
        }

    max_t = toa[-1] - T
    valid_idx_all = np.where(toa <= max_t)[0]
    if len(valid_idx_all) == 0:
        return {
            "hit_rate": 0.0,
            "span_rate": 0.0,
            "support_rate": 0.0,
            "confidence": 0.0,
            "valid_pairs": 0,
            "hit_count": 0,
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

    valid_pos = pos < n
    if np.any(valid_pos):
        dist = np.abs(toa[pos[valid_pos]] - targets[valid_pos])
        hits[valid_pos] |= dist <= cfg.frame_toa_tol

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

    confidence = 0.70 * hit_rate + 0.30 * span_rate
    return {
        "hit_rate": float(hit_rate),
        "span_rate": float(span_rate),
        "support_rate": float(support_rate),
        "confidence": float(confidence),
        "valid_pairs": int(valid_pairs),
        "hit_count": int(hit_count),
    }


def estimate_tframe_one_sigidx_short(sigidx, df_sig, cfg: MainSortConfig, mcfg: LocalConfig):
    df_sig = df_sig.sort_values("TOA").reset_index(drop=True)
    toa = df_sig["TOA"].values.astype(float)
    n = len(toa)

    row = {
        "SigIdx": int(sigidx),
        "NumPulses": int(n),
        "TOA_Start": float(toa[0]) if n > 0 else np.nan,
        "TOA_End": float(toa[-1]) if n > 0 else np.nan,
        "Span": float(toa[-1] - toa[0]) if n > 1 else 0.0,
        "Tframe": np.nan,
        "Tframe_Confidence": 0.0,
        "Hit_Rate": 0.0,
        "Span_Rate": 0.0,
        "Support_Rate": 0.0,
        "ValidPairs": 0,
        "HitCount": 0,
        "NumCandidates": 0,
        "IsValidTframe": False,
    }

    if n < mcfg.min_pulses_for_tframe:
        return row

    candidates = short_window_period_candidates(toa, cfg, mcfg)
    row["NumCandidates"] = len(candidates)
    if not candidates:
        return row

    best = None
    for T in candidates:
        stat = compute_one_step_hit_rate(toa, T, mcfg)
        score = stat["confidence"]
        if best is None or score > best["score"]:
            best = {"T": T, "score": score, **stat}

    is_valid = (
        best["hit_rate"] >= mcfg.min_hit_rate_for_valid
        and best["span_rate"] >= mcfg.min_span_rate_for_valid
        and best["confidence"] >= mcfg.min_confidence_for_valid
    )

    row.update({
        "Tframe": float(best["T"]),
        "Tframe_Confidence": float(best["confidence"]),
        "Hit_Rate": float(best["hit_rate"]),
        "Span_Rate": float(best["span_rate"]),
        "Support_Rate": float(best["support_rate"]),
        "ValidPairs": int(best["valid_pairs"]),
        "HitCount": int(best["hit_count"]),
        "IsValidTframe": bool(is_valid),
    })
    return row


def estimate_tframe_for_beat_short(presort_df, cfg: MainSortConfig, mcfg: LocalConfig):
    rows = []
    for sigidx, group in presort_df.groupby("SigIdx"):
        rows.append(estimate_tframe_one_sigidx_short(sigidx, group, cfg, mcfg))
    return pd.DataFrame(rows)


def is_feature_merge_candidate(row, cfg: LocalConfig):
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


def periods_close(T1, T2, cfg: LocalConfig):
    if not np.isfinite(T1) or not np.isfinite(T2):
        return False, None

    diff = abs(float(T1) - float(T2))
    tol = max(cfg.merge_abs_tol, cfg.merge_T_rel_tol * max(T1, T2))
    if diff <= tol:
        return True, 0.5 * (float(T1) + float(T2))

    if not cfg.allow_harmonic_merge:
        return False, None

    small = min(float(T1), float(T2))
    large = max(float(T1), float(T2))
    if small <= 0:
        return False, None

    k = round(large / small)
    if 2 <= k <= cfg.merge_harmonic_order:
        residual = abs(large - k * small)
        tol_h = max(cfg.merge_abs_tol, cfg.merge_T_rel_tol * large)
        if residual <= tol_h:
            return True, small

    return False, None


def build_sigidx_to_toa(presort_df):
    mapping = {}
    for sigidx, group in presort_df.groupby("SigIdx"):
        mapping[int(sigidx)] = group.sort_values("TOA")["TOA"].values.astype(float)
    return mapping


def verify_merged_sigidx_period(sigidx_a, sigidx_b, common_T, sigidx_to_toa, cfg: LocalConfig):
    toa = np.concatenate([sigidx_to_toa[int(sigidx_a)], sigidx_to_toa[int(sigidx_b)]])
    toa = np.unique(np.sort(toa))
    stat = compute_one_step_hit_rate(toa, common_T, cfg)
    ok = (
        stat["hit_rate"] >= cfg.merge_verify_min_hit_rate
        and stat["span_rate"] >= cfg.merge_verify_min_span_rate
        and stat["confidence"] >= cfg.merge_verify_min_confidence
    )
    return ok, stat


def merge_sigidx_by_tframe(feature_df, presort_df, cfg: LocalConfig):
    feature_df = feature_df.copy()
    all_sigidx = sorted(feature_df["SigIdx"].astype(int).tolist())
    uf = UnionFind(all_sigidx)
    sigidx_to_toa = build_sigidx_to_toa(presort_df)

    valid_df = feature_df[
        feature_df.apply(lambda row: is_feature_merge_candidate(row, cfg), axis=1)
    ].copy()

    if len(valid_df) == 0:
        return {sigidx: idx for idx, sigidx in enumerate(all_sigidx)}, pd.DataFrame()

    valid_df = valid_df.sort_values("Tframe").reset_index(drop=True)
    merge_records = []

    for a in pbar(
        range(len(valid_df)),
        desc="base cycle merge",
        unit="batch",
        leave=False,
        cfg=cfg,
    ):
        row_a = valid_df.iloc[a]
        sig_a = int(row_a["SigIdx"])

        for b in range(a + 1, len(valid_df)):
            row_b = valid_df.iloc[b]
            sig_b = int(row_b["SigIdx"])
            if sig_a == sig_b or uf.find(sig_a) == uf.find(sig_b):
                continue

            n_a = int(row_a["NumPulses"])
            n_b = int(row_b["NumPulses"])
            if min(n_a, n_b) / max(n_a, n_b) < cfg.merge_min_size_ratio:
                continue

            close, common_T = periods_close(row_a["Tframe"], row_b["Tframe"], cfg)
            if not close:
                continue

            verify_stat = {"hit_rate": np.nan, "span_rate": np.nan, "confidence": np.nan}
            if cfg.merge_verify:
                verify_ok, verify_stat = verify_merged_sigidx_period(
                    sig_a,
                    sig_b,
                    common_T,
                    sigidx_to_toa,
                    cfg,
                )
                if not verify_ok:
                    continue

            uf.union(sig_a, sig_b)
            small_T = min(float(row_a["Tframe"]), float(row_b["Tframe"]))
            large_T = max(float(row_a["Tframe"]), float(row_b["Tframe"]))
            ratio = large_T / small_T if small_T > 0 else np.nan
            merge_records.append({
                "SigIdx_A": sig_a,
                "SigIdx_B": sig_b,
                "T_A": float(row_a["Tframe"]),
                "T_B": float(row_b["Tframe"]),
                "Common_T": float(common_T),
                "Conf_A": float(row_a["Tframe_Confidence"]),
                "Conf_B": float(row_b["Tframe_Confidence"]),
                "Hit_A": float(row_a["Hit_Rate"]),
                "Hit_B": float(row_b["Hit_Rate"]),
                "Span_A": float(row_a["Span_Rate"]),
                "Span_B": float(row_b["Span_Rate"]),
                "Verify_Hit": float(verify_stat["hit_rate"]),
                "Verify_Span": float(verify_stat["span_rate"]),
                "Verify_Conf": float(verify_stat["confidence"]),
                "Is_Harmonic": bool(
                    np.isfinite(ratio)
                    and abs(ratio - round(ratio)) < cfg.merge_T_rel_tol
                ),
            })

    root_to_predid = {}
    sigidx_to_predid = {}
    next_id = 0
    for sigidx in all_sigidx:
        root = uf.find(sigidx)
        if root not in root_to_predid:
            root_to_predid[root] = next_id
            next_id += 1
        sigidx_to_predid[int(sigidx)] = int(root_to_predid[root])

    return sigidx_to_predid, pd.DataFrame(merge_records)


def apply_sigidx_mapping(presort_df, sigidx_to_predid):
    pred_df = presort_df.copy()
    pred_df["PredID"] = pred_df["SigIdx"].astype(int).map(sigidx_to_predid)
    missing = pred_df["PredID"].isna()
    if missing.any():
        max_id = int(np.nanmax(list(sigidx_to_predid.values()))) if sigidx_to_predid else 0
        missing_sigidx = sorted(pred_df.loc[missing, "SigIdx"].astype(int).unique())
        extra_map = {sig: max_id + 1 + i for i, sig in enumerate(missing_sigidx)}
        pred_df.loc[missing, "PredID"] = pred_df.loc[missing, "SigIdx"].astype(int).map(extra_map)
    pred_df["PredID"] = pred_df["PredID"].astype(int)
    return pred_df


def build_mapping_df(sigidx_to_predid):
    return pd.DataFrame([
        {"SigIdx": int(sigidx), "PredID": int(predid)}
        for sigidx, predid in sorted(sigidx_to_predid.items(), key=lambda x: x[0])
    ])


def canonicalize_stream_mapping(sigidx_to_local_predid):
    """Keep the globally associated pre-sort SigIdx namespace in online output."""
    members_by_local = {}
    for sigidx, local_predid in sigidx_to_local_predid.items():
        members_by_local.setdefault(int(local_predid), []).append(int(sigidx))

    local_to_canonical = {
        local_predid: min(members)
        for local_predid, members in members_by_local.items()
    }
    return {
        int(sigidx): int(local_to_canonical[int(local_predid)])
        for sigidx, local_predid in sigidx_to_local_predid.items()
    }


def select_pdw_param_cols(pdw_df, cfg: LocalConfig):
    preferred = [c for c in cfg.pdw_param_cols if c in pdw_df.columns]
    if not preferred:
        preferred = [
            c for c in pdw_df.columns
            if c not in ("TOA", "SigIdx", "Beat") and pd.api.types.is_numeric_dtype(pdw_df[c])
        ]

    param_cols = []
    for c in preferred:
        s = pd.to_numeric(pdw_df[c], errors="coerce")
        if s.notna().sum() == 0:
            continue
        if float(s.std(skipna=True) or 0.0) <= 1e-12:
            continue
        param_cols.append(c)
    return param_cols


def build_sigidx_pdw_param_features(presort_df, cfg: LocalConfig):
    pdw = presort_df.copy()
    param_cols = select_pdw_param_cols(pdw, cfg)
    if not param_cols:
        return pd.DataFrame(), []

    for c in param_cols:
        pdw[c] = pd.to_numeric(pdw[c], errors="coerce")

    aligned = pdw[["SigIdx"] + param_cols].dropna(subset=param_cols, how="all")
    if len(aligned) == 0:
        return pd.DataFrame(), param_cols

    agg = aligned.groupby("SigIdx")[param_cols].agg(["median", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()
    counts = aligned.groupby("SigIdx").size().rename("PDW_NumPulses").reset_index()
    agg = agg.merge(counts, on="SigIdx", how="left")
    agg["SigIdx"] = agg["SigIdx"].astype(int)
    return agg, param_cols


def periods_related_for_absorb(T1, T2, cfg: LocalConfig):
    if not (np.isfinite(T1) and np.isfinite(T2)) or T1 <= 0 or T2 <= 0:
        return False

    diff = abs(float(T1) - float(T2))
    tol = max(cfg.param_absorb_period_abs_tol,
              cfg.param_absorb_period_rel_tol * max(T1, T2))
    if diff <= tol:
        return True

    small = min(float(T1), float(T2))
    large = max(float(T1), float(T2))
    if small <= 0:
        return False

    k = round(large / small)
    if 2 <= k <= cfg.param_absorb_harmonic_order:
        residual = abs(large - k * small)
        tol_h = max(cfg.param_absorb_period_abs_tol,
                    cfg.param_absorb_period_rel_tol * large)
        if residual <= tol_h:
            return True

    return False


def pdw_param_compatibility_score(row_a, row_b, param_cols):
    score = 0.0
    diffs = {}
    for c in param_cols:
        ca = f"{c}_median"
        cb = f"{c}_median"
        if ca not in row_a or cb not in row_b:
            continue
        va = row_a[ca]
        vb = row_b[cb]
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue

        diff = abs(float(va) - float(vb))
        tol = ABSORB_PARAM_TOLS.get(c, 1.0)
        weight = ABSORB_PARAM_WEIGHTS.get(c, 1.0)
        score += max(0.0, 1.0 - diff / tol) * weight
        diffs[c] = float(diff)

    return float(score), diffs


def best_param_mode_pair(small_rows, anchor_rows, param_cols, cfg: LocalConfig):
    mode_cols = [f"{c}_median" for c in param_cols]
    small_modes = (
        small_rows
        .dropna(subset=mode_cols, how="all")
        .sort_values("NumPulses", ascending=False)
        .head(cfg.param_absorb_top_small_modes)
    )
    anchor_modes = (
        anchor_rows
        .dropna(subset=mode_cols, how="all")
        .sort_values("NumPulses", ascending=False)
        .head(cfg.param_absorb_top_anchor_modes)
    )

    if len(small_modes) == 0 or len(anchor_modes) == 0:
        return None

    best = None
    for _, row_s in small_modes.iterrows():
        for _, row_a in anchor_modes.iterrows():
            param_score, diffs = pdw_param_compatibility_score(row_s, row_a, param_cols)
            period_related = periods_related_for_absorb(
                row_s.get("Tframe", np.nan),
                row_a.get("Tframe", np.nan),
                cfg,
            )
            rank_score = param_score + (0.5 if period_related else 0.0)
            if best is None or rank_score > best["RankScore"]:
                best = {
                    "RankScore": float(rank_score),
                    "ParamScore": float(param_score),
                    "ParamDiffs": diffs,
                    "PeriodRelated": bool(period_related),
                    "SmallModeSigIdx": int(row_s["SigIdx"]),
                    "AnchorModeSigIdx": int(row_a["SigIdx"]),
                }
    return best


def absorb_fragments_by_pdw_params(feature_df, presort_df, sigidx_to_predid,
                                   pdw_param_df, param_cols, cfg: LocalConfig):
    if pdw_param_df is None or len(pdw_param_df) == 0 or len(param_cols) == 0:
        return sigidx_to_predid, pd.DataFrame()

    feature_cols = [
        "SigIdx",
        "NumPulses",
        "Tframe",
        "Tframe_Confidence",
        "Hit_Rate",
        "Span_Rate",
        "IsValidTframe",
    ]
    sig_rows = feature_df[feature_cols].copy()
    sig_rows = sig_rows.merge(pdw_param_df, on="SigIdx", how="left")
    sig_rows["BasePredID"] = sig_rows["SigIdx"].astype(int).map(sigidx_to_predid)
    sig_rows = sig_rows.dropna(subset=["BasePredID"]).copy()
    if len(sig_rows) == 0:
        return sigidx_to_predid, pd.DataFrame()
    sig_rows["BasePredID"] = sig_rows["BasePredID"].astype(int)

    component_df = (
        sig_rows
        .groupby("BasePredID")
        .agg(Size=("NumPulses", "sum"), NumSigIdx=("SigIdx", "count"))
        .reset_index()
        .rename(columns={"BasePredID": "PredID"})
    )
    if len(component_df) == 0:
        return sigidx_to_predid, pd.DataFrame()

    rows_by_predid = {
        int(predid): group.copy()
        for predid, group in sig_rows.groupby("BasePredID")
    }
    uf = UnionFind(component_df["PredID"].astype(int).tolist())
    absorb_records = []

    for _, small in component_df.sort_values("Size").iterrows():
        small_id = int(small["PredID"])
        small_size = int(small["Size"])
        if small_size > cfg.param_absorb_small_max_pulses:
            continue
        if small_id not in rows_by_predid:
            continue

        best_anchor = None
        small_rows = rows_by_predid[small_id]
        for _, anchor in component_df.iterrows():
            anchor_id = int(anchor["PredID"])
            anchor_size = int(anchor["Size"])
            if anchor_id == small_id:
                continue
            if uf.find(anchor_id) == uf.find(small_id):
                continue
            if anchor_size < cfg.param_absorb_anchor_min_pulses:
                continue
            if anchor_size < small_size:
                continue
            if small_size / max(anchor_size, 1) > cfg.param_absorb_max_size_ratio:
                continue
            if anchor_id not in rows_by_predid:
                continue

            mode_pair = best_param_mode_pair(
                small_rows,
                rows_by_predid[anchor_id],
                param_cols,
                cfg,
            )
            if mode_pair is None:
                continue
            if mode_pair["ParamScore"] < cfg.param_absorb_min_score:
                continue

            param_diffs = mode_pair["ParamDiffs"]
            if "Param5" in param_diffs and param_diffs["Param5"] > cfg.param_absorb_p5_hard_tol:
                continue
            if small_size > cfg.param_absorb_period_small_max_pulses and not mode_pair["PeriodRelated"]:
                continue

            rank = (
                mode_pair["RankScore"]
                + min(anchor_size, 50000) / 50000.0 * 0.4
                - small_size / max(anchor_size, 1) * 0.5
            )
            if best_anchor is None or rank > best_anchor["Rank"]:
                best_anchor = {
                    "Rank": float(rank),
                    "AnchorPredID": anchor_id,
                    "AnchorSize": anchor_size,
                    **mode_pair,
                }

        if best_anchor is None:
            continue

        uf.union(best_anchor["AnchorPredID"], small_id)
        record = {
            "SmallPredID": small_id,
            "AnchorPredID": int(best_anchor["AnchorPredID"]),
            "SmallSize": small_size,
            "AnchorSize": int(best_anchor["AnchorSize"]),
            "ParamScore": float(best_anchor["ParamScore"]),
            "RankScore": float(best_anchor["RankScore"]),
            "Rank": float(best_anchor["Rank"]),
            "PeriodRelated": bool(best_anchor["PeriodRelated"]),
            "SmallModeSigIdx": int(best_anchor["SmallModeSigIdx"]),
            "AnchorModeSigIdx": int(best_anchor["AnchorModeSigIdx"]),
        }
        for c, diff in best_anchor["ParamDiffs"].items():
            record[f"{c}_Diff"] = float(diff)
        absorb_records.append(record)

    root_to_predid = {}
    sigidx_to_new_predid = {}
    next_id = 0
    for sigidx in sorted(sigidx_to_predid.keys()):
        base_predid = int(sigidx_to_predid[sigidx])
        root = uf.find(base_predid)
        if root not in root_to_predid:
            root_to_predid[root] = next_id
            next_id += 1
        sigidx_to_new_predid[int(sigidx)] = int(root_to_predid[root])

    return sigidx_to_new_predid, pd.DataFrame(absorb_records)


def evaluate_prediction(pred_df, truth_df, cfg: MainSortConfig, exp_name):
    ecfg = build_main2_eval_cfg(cfg)
    eval_df = align_pred_and_truth(
        pred_df[["TOA", "PredID"]].sort_values("TOA").reset_index(drop=True),
        truth_df,
        ecfg,
        prefer_pulse_id=False,
    )
    return evaluate_sorting_official_like(
        eval_df,
        ecfg,
        pred_col="PredID",
        true_col="TrueLabel",
        toa_col="TOA",
        exp_name=exp_name,
    )


def summarize_batches(pred_df):
    per_beat = pred_df.groupby("Beat")["PredID"].nunique()
    return {
        "GlobalUniqueBatches": int(pred_df["PredID"].nunique()),
        "TotalBeatBatches": int(per_beat.sum()),
        "MeanBatchesPerBeat": float(per_beat.mean()),
        "MedianBatchesPerBeat": float(per_beat.median()),
        "MaxBatchesPerBeat": int(per_beat.max()),
        "MinBatchesPerBeat": int(per_beat.min()),
    }


def stream_file_is_stable(path, cfg: MainSortConfig):
    try:
        size_before = path.stat().st_size
        if size_before <= 0:
            return False
        time.sleep(max(0.0, cfg.stream_file_stable_seconds))
        return path.exists() and path.stat().st_size == size_before
    except OSError:
        return False


def process_one_streaming_beat(path, cfg: MainSortConfig, mcfg: LocalConfig,
                               output_dir):
    """Process and emit one beat without truth labels or future-beat evidence."""
    beat_id = int(path.stem.split("_")[-1])
    beat_df = read_beat(path, beat_id, cfg)
    if beat_df.empty:
        return None

    feature_df = estimate_tframe_for_beat_short(beat_df, cfg, mcfg)
    exp3_mapping, exp3_pairs = merge_sigidx_by_tframe(feature_df, beat_df, mcfg)
    pdw_param_df, pdw_param_cols = build_sigidx_pdw_param_features(beat_df, mcfg)
    if mcfg.enable_param_absorb:
        exp4_mapping, exp4_pairs = absorb_fragments_by_pdw_params(
            feature_df,
            beat_df,
            exp3_mapping,
            pdw_param_df,
            pdw_param_cols,
            mcfg,
        )
    else:
        exp4_mapping = exp3_mapping
        exp4_pairs = pd.DataFrame()

    # Pre-sort SigIdx values already carry cross-beat association.  Keep that
    # namespace and use the smallest member as the merged component label.
    stream_mapping = canonicalize_stream_mapping(exp4_mapping)
    stats = write_annotated_beat_outputs(
        [path],
        stream_mapping,
        cfg,
        output_dir,
        pred_col=cfg.annotated_label_col,
    )
    return {
        "Beat": beat_id,
        "NumPulses": int(len(beat_df)),
        "InputBatches": int(beat_df["SigIdx"].nunique()),
        "OutputBatches": int(len(set(stream_mapping.values()))),
        "CycleMerges": int(len(exp3_pairs)),
        "PDWAbsorbs": int(len(exp4_pairs)),
        **stats,
    }


def persist_stream_summary(result, output_dir):
    """Persist beat and cumulative summaries before the next beat starts."""
    output_dir = Path(output_dir)
    beat_summary_path = output_dir / "stream_beat_summary.csv"
    write_header = not beat_summary_path.exists() or beat_summary_path.stat().st_size == 0
    pd.DataFrame([result]).to_csv(
        beat_summary_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )

    all_rows = pd.read_csv(beat_summary_path, encoding="utf-8-sig")
    running = pd.DataFrame([{
        "ProcessedBeats": int(len(all_rows)),
        "FirstBeat": int(all_rows["Beat"].min()),
        "LastBeat": int(all_rows["Beat"].max()),
        "TotalPulses": int(all_rows["NumPulses"].sum()),
        "InputBatchesSum": int(all_rows["InputBatches"].sum()),
        "OutputBatchesSum": int(all_rows["OutputBatches"].sum()),
        "MeanInputBatches": float(all_rows["InputBatches"].mean()),
        "MeanOutputBatches": float(all_rows["OutputBatches"].mean()),
        "TotalCycleMerges": int(all_rows["CycleMerges"].sum()),
        "TotalPDWAbsorbs": int(all_rows["PDWAbsorbs"].sum()),
    }])
    running_path = output_dir / "stream_running_summary.csv"
    temp_path = running_path.with_suffix(running_path.suffix + ".tmp")
    running.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, running_path)


def run_streaming_online(cfg: MainSortConfig):
    """Watch beat files and emit each completed 200 ms beat immediately."""
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.stream_output_dir) if cfg.stream_output_dir else (
        input_dir.with_name(f"{input_dir.name}_main5_stream")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mcfg = build_candidate_cfg(cfg)
    processed = set()

    if cfg.stream_skip_existing_outputs:
        processed = {p.name for p in output_dir.glob("beat_*.txt")}
    else:
        # A deliberate rerun overwrites beat outputs and starts fresh summaries.
        for summary_name in ("stream_beat_summary.csv", "stream_running_summary.csv"):
            summary_path = output_dir / summary_name
            if summary_path.exists():
                summary_path.unlink()

    print("main5 online beat-wise sorting")
    print("Input dir:", input_dir)
    print("Output dir:", output_dir)
    print("Existing outputs skipped:", len(processed))
    print("Watch forever:", cfg.stream_watch_forever)
    if not cfg.stream_watch_forever:
        print("Idle timeout (s):", cfg.stream_idle_timeout_seconds)

    rows = []
    last_emit_time = time.monotonic()
    try:
        while True:
            beat_files = sorted(input_dir.glob("beat_*.txt"))
            pending = [p for p in beat_files if p.name not in processed]
            emitted_this_scan = 0

            for path in pending:
                if not stream_file_is_stable(path, cfg):
                    continue
                result = process_one_streaming_beat(path, cfg, mcfg, output_dir)
                if result is None:
                    continue
                processed.add(path.name)
                rows.append(result)
                persist_stream_summary(result, output_dir)
                emitted_this_scan += 1
                last_emit_time = time.monotonic()
                print(
                    f"Beat {result['Beat']:06d} emitted: "
                    f"pulses={result['NumPulses']}, "
                    f"batches={result['InputBatches']}->{result['OutputBatches']}, "
                    f"file={output_dir / path.name}",
                    flush=True,
                )

            if emitted_this_scan == 0:
                idle_seconds = time.monotonic() - last_emit_time
                if (
                    not cfg.stream_watch_forever
                    and idle_seconds >= cfg.stream_idle_timeout_seconds
                ):
                    print(
                        f"No new completed beat for {idle_seconds:.1f}s; exiting.",
                        flush=True,
                    )
                    break
                time.sleep(max(0.05, cfg.stream_poll_seconds))
    except KeyboardInterrupt:
        print("Streaming stopped by user.")

    summary_path = output_dir / "stream_beat_summary.csv"
    summary = (
        pd.read_csv(summary_path, encoding="utf-8-sig")
        if summary_path.exists()
        else pd.DataFrame()
    )
    print("Emitted beats this run:", len(rows))
    return summary


def run_main_sort_experiments(cfg: MainSortConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    mcfg = build_candidate_cfg(cfg)
    beat_files = list_beat_files(cfg.input_dir)
    input_path = Path(cfg.input_dir)
    stream_dir = cfg.stream_output_dir or str(
        input_path.with_name(f"{input_path.name}_main5_stream")
    )
    stream_stats = {
        "written_files": 0,
        "written_rows": 0,
        "missing_labels": 0,
    }

    print("main5: 200 ms version of main.py workflow")
    print("Input beats:", len(beat_files))
    print("Input dir:", cfg.input_dir)
    print("Output dir:", cfg.output_dir)
    print("Workflow: Exp1 raw, Exp2 Tframe feature, Exp3 Tframe merge, Exp4 PDW absorb")
    print("Global pair min support beats:", cfg.global_pair_min_support_beats)
    print("Immediate beat output:", cfg.stream_output_per_beat)
    if cfg.stream_output_per_beat:
        print("Streaming output dir:", stream_dir)

    exp1_name = "Exp1_PreSort_Only"
    exp2_name = "Exp2_Tframe_Feature_Only"
    exp3_name = "Exp3_Tframe_Merge"
    exp4_name = "Exp4_Param_Aware_Absorb"

    all_beat_parts = []
    all_raw_sigidx = set()
    feature_parts = []
    exp3_pair_parts = []
    exp4_absorb_parts = []
    pdw_param_parts = []
    global_union_rows = {
        exp3_name: [],
        exp4_name: [],
    }
    beat_rows = []

    iterator = pbar(
        beat_files,
        desc="main5: Tframe + PDW absorb",
        unit="beat",
        leave=True,
        cfg=LocalConfig(show_progress=cfg.show_progress),
    )

    for path in iterator:
        beat_id = int(path.stem.split("_")[-1])
        beat_df = read_beat(path, beat_id, cfg)
        if beat_df.empty:
            continue
        all_beat_parts.append(beat_df[["TOA", "Beat", "SigIdx"]].copy())
        for sigidx in beat_df["SigIdx"].astype(int).unique():
            all_raw_sigidx.add(int(sigidx))

        raw_map = raw_mapping(beat_df)

        feature_df = estimate_tframe_for_beat_short(beat_df, cfg, mcfg)
        feature_df["Beat"] = beat_id
        feature_parts.append(feature_df)

        exp3_mapping, exp3_pairs = merge_sigidx_by_tframe(feature_df, beat_df, mcfg)
        if len(exp3_pairs) > 0:
            exp3_pairs = exp3_pairs.copy()
            exp3_pairs["Beat"] = beat_id
            exp3_pair_parts.append(exp3_pairs)

        pdw_param_df, pdw_param_cols = build_sigidx_pdw_param_features(beat_df, mcfg)
        if len(pdw_param_df) > 0:
            pdw_param_df = pdw_param_df.copy()
            pdw_param_df["Beat"] = beat_id
            pdw_param_df["ParamCols"] = ";".join(pdw_param_cols)
            pdw_param_parts.append(pdw_param_df)

        if mcfg.enable_param_absorb:
            exp4_mapping, exp4_absorb_pairs = absorb_fragments_by_pdw_params(
                feature_df,
                beat_df,
                exp3_mapping,
                pdw_param_df,
                pdw_param_cols,
                mcfg,
            )
        else:
            exp4_mapping = exp3_mapping
            exp4_absorb_pairs = pd.DataFrame()

        if len(exp4_absorb_pairs) > 0:
            exp4_absorb_pairs = exp4_absorb_pairs.copy()
            exp4_absorb_pairs["Beat"] = beat_id
            exp4_absorb_parts.append(exp4_absorb_pairs)

        # Online output: the pre-sort SigIdx is already associated across beats.
        # Therefore each completed beat can be emitted immediately without using
        # future beats or the offline five-beat support mapping.
        if cfg.stream_output_per_beat:
            stream_mapping = canonicalize_stream_mapping(exp4_mapping)
            one_stats = write_annotated_beat_outputs(
                [path],
                stream_mapping,
                cfg,
                stream_dir,
                pred_col=cfg.annotated_label_col,
            )
            for key in stream_stats:
                stream_stats[key] += int(one_stats.get(key, 0))

        global_union_rows[exp3_name].extend(
            collect_mapping_groups(exp3_mapping, beat_id, exp3_name)
        )
        global_union_rows[exp4_name].extend(
            collect_mapping_groups(exp4_mapping, beat_id, exp4_name)
        )

        beat_row = {
            "Beat": beat_id,
            "NumPulses": int(len(beat_df)),
            "RawBatches": int(beat_df["SigIdx"].nunique()),
            "Exp1LocalBatches": int(len(set(raw_map.values()))),
            "Exp2LocalBatches": int(len(set(raw_map.values()))),
            "Exp3LocalBatches": int(len(set(exp3_mapping.values()))),
            "Exp3Merges": int(len(exp3_pairs)),
            "Exp4LocalBatches": int(len(set(exp4_mapping.values()))),
            "Exp4Absorbs": int(len(exp4_absorb_pairs)),
            "ValidTframeCount": int(feature_df["IsValidTframe"].sum()),
            "PDWParamCols": ";".join(pdw_param_cols),
        }
        beat_rows.append(beat_row)

    all_pred_base = pd.concat(all_beat_parts, ignore_index=True)
    raw_sigidx_to_predid = {int(sigidx): int(i) for i, sigidx in enumerate(sorted(all_raw_sigidx))}

    predictions = {
        exp1_name: make_global_sigidx_prediction(all_pred_base, raw_sigidx_to_predid),
        exp2_name: make_global_sigidx_prediction(all_pred_base, raw_sigidx_to_predid),
    }

    global_union_dfs = {}
    pair_support_dfs = {}
    global_mappings = {
        exp1_name: raw_sigidx_to_predid,
        exp2_name: raw_sigidx_to_predid,
    }
    global_mapping_dfs = {
        exp1_name: build_mapping_df(raw_sigidx_to_predid),
        exp2_name: build_mapping_df(raw_sigidx_to_predid),
    }
    for name in [exp3_name, exp4_name]:
        union_df = pd.DataFrame(global_union_rows[name])
        pair_support_df = build_pair_support(union_df)
        sigidx_to_global_predid = build_global_sigidx_mapping(
            all_raw_sigidx,
            pair_support_df,
            cfg,
        )
        global_union_dfs[name] = union_df
        pair_support_dfs[name] = pair_support_df
        global_mappings[name] = sigidx_to_global_predid
        global_mapping_dfs[name] = build_mapping_df(sigidx_to_global_predid)
        predictions[name] = make_global_sigidx_prediction(
            all_pred_base,
            sigidx_to_global_predid,
        )

    feature_all = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    exp3_pair_all = pd.concat(exp3_pair_parts, ignore_index=True) if exp3_pair_parts else pd.DataFrame()
    exp4_absorb_all = pd.concat(exp4_absorb_parts, ignore_index=True) if exp4_absorb_parts else pd.DataFrame()
    pdw_param_all = pd.concat(pdw_param_parts, ignore_index=True) if pdw_param_parts else pd.DataFrame()
    beat_summary = pd.DataFrame(beat_rows)

    truth_df = normalize_truth_df(
        read_table_auto(cfg.truth_path),
        toa_col=cfg.truth_toa_col,
        true_col=cfg.truth_label_col,
    )

    metrics_by_name = {}
    summary_rows = []
    for name, pred in predictions.items():
        metrics = evaluate_prediction(pred, truth_df, cfg, name)
        metrics_by_name[name] = metrics
        summary_rows.append({
            "Experiment": name,
            "Sort_ACC": metrics["Sort_ACC"],
            "Add_Rate": metrics["Add_Rate"],
            "Err_Rate": metrics["Err_Rate"],
            "GlobalPairMinSupportBeats": cfg.global_pair_min_support_beats,
            **summarize_batches(pred),
        })

    summary = pd.DataFrame(summary_rows)
    best_name = exp4_name

    summary.to_csv(os.path.join(cfg.output_dir, "experiment_summary.csv"), index=False, encoding="utf-8-sig")
    beat_summary.to_csv(os.path.join(cfg.output_dir, "beat_summary.csv"), index=False, encoding="utf-8-sig")
    feature_all.to_csv(os.path.join(cfg.output_dir, "sigidx_tframe_features.csv"), index=False, encoding="utf-8-sig")
    exp3_pair_all.to_csv(os.path.join(cfg.output_dir, "exp3_tframe_merge_pairs.csv"), index=False, encoding="utf-8-sig")
    exp4_absorb_all.to_csv(os.path.join(cfg.output_dir, "exp4_absorb_pairs.csv"), index=False, encoding="utf-8-sig")
    pdw_param_all.to_csv(os.path.join(cfg.output_dir, "sigidx_pdw_param_features.csv"), index=False, encoding="utf-8-sig")

    for name, df in global_union_dfs.items():
        df.to_csv(os.path.join(cfg.output_dir, f"{name}_global_sigidx_union_groups.csv"), index=False, encoding="utf-8-sig")
    for name, df in pair_support_dfs.items():
        df.to_csv(os.path.join(cfg.output_dir, f"{name}_global_sigidx_pair_support.csv"), index=False, encoding="utf-8-sig")
    for name, df in global_mapping_dfs.items():
        df.to_csv(os.path.join(cfg.output_dir, f"{name}_global_sigidx_to_predid.csv"), index=False, encoding="utf-8-sig")

    for name, metrics in metrics_by_name.items():
        metrics["target_detail"].to_csv(
            os.path.join(cfg.output_dir, f"{name}_target_detail.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        metrics["beat_detail"].to_csv(
            os.path.join(cfg.output_dir, f"{name}_beat_detail.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if cfg.save_full_predictions:
        for name, pred in predictions.items():
            pred.to_csv(
                os.path.join(cfg.output_dir, f"{name}_pred.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    annotated_dir = cfg.annotated_output_dir or str(
        input_path.with_name(f"{input_path.name}_1")
    )
    if cfg.stream_output_per_beat:
        annotated_dir = stream_dir
        annotation_stats = stream_stats
    elif cfg.write_annotated_outputs:
        annotation_stats = write_annotated_beat_outputs(
            beat_files,
            global_mappings[best_name],
            cfg,
            annotated_dir,
            pred_col=cfg.annotated_label_col,
        )
    else:
        annotation_stats = {
            "written_files": 0,
            "written_rows": 0,
            "missing_labels": 0,
        }

    print("\n========== main5 main.py-style summary ==========")
    print(summary)
    print("\nFinal experiment:", best_name)
    print("\nFinal target detail:")
    print(metrics_by_name[best_name]["target_detail"])
    print("\nAnnotated beat files:", annotated_dir)
    print("Annotated files:", annotation_stats["written_files"])
    print("Annotated rows:", annotation_stats["written_rows"])
    print("Missing labels:", annotation_stats["missing_labels"])
    print("\nSaved to:", cfg.output_dir)

    return {
        "summary": summary,
        "best_name": best_name,
        "metrics": metrics_by_name,
        "beat_summary": beat_summary,
        "features": feature_all,
        "exp3_pairs": exp3_pair_all,
        "exp4_absorbs": exp4_absorb_all,
        "annotated_dir": annotated_dir,
        "annotation_stats": annotation_stats,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = MainSortConfig(
        input_dir=os.path.join(base_dir, "outputs_streaming_200ms_sample1"),
        truth_path=os.path.join(base_dir, "Sorted1_PDW.txt"),
        output_dir=os.path.join(base_dir, "experiment_outputs_main5_mainpy_200ms_tframe_pdw"),
        stream_output_per_beat=True,
        stream_output_dir=os.path.join(
            base_dir, "outputs_streaming_200ms_sample1_main5_stream"
        ),
        # Direct PyCharm run: reprocess beats and exit after 30 s of inactivity.
        stream_watch_forever=False,
        stream_idle_timeout_seconds=30.0,
        stream_skip_existing_outputs=False,
        write_annotated_outputs=False,
    )
    run_streaming_online(cfg)

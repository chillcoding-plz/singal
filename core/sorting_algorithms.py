import time
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .external_model_adapters import run_cycle_period_sorting, run_hdbscan_sorting

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - fallback for minimal environments
    DBSCAN = None
    KMeans = None
    StandardScaler = None


@dataclass
class SortingOutput:
    data: pd.DataFrame
    method: str
    elapsed: float
    track_count: int
    assigned_count: int
    unassigned_count: int
    summary: Dict[str, object] = field(default_factory=dict)


FEATURE_COLUMNS = ["TOA", "RF", "PRI", "PW"]


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    values = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    if values.shape[1] == 0:
        return np.arange(len(df)).reshape(-1, 1)
    if StandardScaler is None:
        std = values.std(axis=0)
        std[std == 0] = 1.0
        return (values - values.mean(axis=0)) / std
    return StandardScaler().fit_transform(values)


def _assign_result(df: pd.DataFrame, labels: np.ndarray, method: str, elapsed: float) -> SortingOutput:
    out = df.copy()
    labels = np.asarray(labels, dtype=int)
    track_ids = np.where(labels < 0, 0, labels + 1)
    out["Track_ID"] = track_ids
    out["Assigned"] = track_ids > 0
    out["Sorting_Method"] = method

    assigned = int(out["Assigned"].sum())
    unassigned = int(len(out) - assigned)
    track_count = int(out.loc[out["Track_ID"] > 0, "Track_ID"].nunique())
    return SortingOutput(
        data=out,
        method=method,
        elapsed=elapsed,
        track_count=track_count,
        assigned_count=assigned,
        unassigned_count=unassigned,
        summary={
            "分选轨迹数": track_count,
            "已分配脉冲数": assigned,
            "未分配脉冲数": unassigned,
            "平均 PRI": float(out["PRI"].mean()) if "PRI" in out else 0.0,
            "平均脉宽": float(out["PW"].mean()) if "PW" in out else 0.0,
        },
    )


def run_sorting(df: pd.DataFrame, method: str, progress_callback=None, should_cancel=None, stream_callback=None, streaming_zeng=False) -> SortingOutput:
    start = time.perf_counter()
    method_key = method.lower()

    if method_key == "hdbscan":
        out = run_hdbscan_sorting(
            df,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
            stream_callback=stream_callback,
            streaming_zeng=streaming_zeng,
        )
        elapsed = time.perf_counter() - start
        assigned = int(out["Assigned"].sum()) if "Assigned" in out else int((out["Track_ID"].astype(int) > 0).sum())
        unassigned = int(len(out) - assigned)
        track_count = int(out.loc[out["Track_ID"].astype(int) > 0, "Track_ID"].nunique())
        result_method = str(out["Sorting_Method"].dropna().iloc[0]) if "Sorting_Method" in out and not out["Sorting_Method"].dropna().empty else method
        return SortingOutput(
            data=out,
            method=result_method,
            elapsed=elapsed,
            track_count=track_count,
            assigned_count=assigned,
            unassigned_count=unassigned,
            summary={
                "分选轨迹数": track_count,
                "已分配脉冲数": assigned,
                "未分配脉冲数": unassigned,
                "平均 PRI": float(out["PRI"].mean()) if "PRI" in out else 0.0,
                "平均脉宽": float(out["PW"].mean()) if "PW" in out else 0.0,
            },
        )
    if method_key in {"cycle_period", "cycle-period", "cycle period"}:
        out = run_cycle_period_sorting(df, progress_callback=progress_callback, should_cancel=should_cancel)
        elapsed = time.perf_counter() - start
        assigned = int(out["Assigned"].sum()) if "Assigned" in out else int((out["Track_ID"].astype(int) > 0).sum())
        unassigned = int(len(out) - assigned)
        track_count = int(out.loc[out["Track_ID"].astype(int) > 0, "Track_ID"].nunique())
        result_method = str(out["Sorting_Method"].dropna().iloc[0]) if "Sorting_Method" in out and not out["Sorting_Method"].dropna().empty else method
        return SortingOutput(
            data=out,
            method=result_method,
            elapsed=elapsed,
            track_count=track_count,
            assigned_count=assigned,
            unassigned_count=unassigned,
            summary={
                "分选轨迹数": track_count,
                "已分配脉冲数": assigned,
                "未分配脉冲数": unassigned,
                "平均 PRI": float(out["PRI"].mean()) if "PRI" in out else 0.0,
                "平均脉宽": float(out["PW"].mean()) if "PW" in out else 0.0,
            },
        )
    if method_key == "dbscan":
        labels = _dbscan(df)
    elif method_key in {"k-means", "kmeans"}:
        labels = _kmeans(df)
    else:
        labels = _mht_placeholder(df)

    return _assign_result(df, labels, method, time.perf_counter() - start)


def _dbscan(df: pd.DataFrame) -> np.ndarray:
    x = _feature_matrix(df)
    if DBSCAN is None:
        return _quantile_labels(df)
    min_samples = max(8, min(50, len(df) // 100))
    return DBSCAN(eps=0.45, min_samples=min_samples).fit_predict(x)


def _kmeans(df: pd.DataFrame) -> np.ndarray:
    x = _feature_matrix(df)
    if KMeans is None:
        return _quantile_labels(df)
    k = 4 if len(df) >= 4 else max(1, len(df))
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(x)


def _mht_placeholder(df: pd.DataFrame) -> np.ndarray:
    if "SigIdx" in df.columns:
        sig = pd.to_numeric(df["SigIdx"], errors="coerce").fillna(0).astype(int)
        unique = [value for value in sorted(sig.unique()) if value > 0]
        mapping = {value: index for index, value in enumerate(unique)}
        return sig.map(lambda value: mapping.get(value, -1)).to_numpy(int)
    if "Track_ID" in df.columns and pd.to_numeric(df["Track_ID"], errors="coerce").fillna(0).max() > 0:
        tracks = pd.to_numeric(df["Track_ID"], errors="coerce").fillna(0).astype(int)
        unique = [value for value in sorted(tracks.unique()) if value > 0]
        mapping = {value: index for index, value in enumerate(unique)}
        return tracks.map(lambda value: mapping.get(value, -1)).to_numpy(int)
    return _quantile_labels(df)


def _cycle_period_placeholder(df: pd.DataFrame) -> np.ndarray:
    if "PRI" in df.columns:
        pri = pd.to_numeric(df["PRI"], errors="coerce").fillna(df["PRI"].median())
        try:
            bins = np.unique(np.quantile(pri, [0.2, 0.4, 0.6, 0.8]))
            return np.digitize(pri, bins).astype(int)
        except Exception:
            pass
    return _quantile_labels(df)


def _quantile_labels(df: pd.DataFrame) -> np.ndarray:
    rf = pd.to_numeric(df["RF"], errors="coerce").fillna(df["RF"].median())
    try:
        bins = np.unique(np.quantile(rf, [0.25, 0.5, 0.75]))
        return np.digitize(rf, bins).astype(int)
    except Exception:
        return np.zeros(len(df), dtype=int)

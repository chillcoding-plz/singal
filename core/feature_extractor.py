from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class FeatureSet:
    features: pd.DataFrame
    feature_columns: List[str]


def extract_track_features(df: pd.DataFrame) -> FeatureSet:
    """Extract per-track statistical features for recognition."""

    if "Track_ID" not in df.columns:
        raise ValueError("请先完成分选，缺少 Track_ID 字段")

    valid = df[pd.to_numeric(df["Track_ID"], errors="coerce").fillna(0).astype(int) > 0].copy()
    if valid.empty:
        raise ValueError("没有可用于识别的已分配轨迹")

    rows = []
    for track_id, group in valid.groupby("Track_ID"):
        row = {
            "Track_ID": int(track_id),
            "Pulse_Count": int(len(group)),
            "Mean_RF": float(group["RF"].mean()),
            "Var_RF": float(group["RF"].var(ddof=0)),
            "Mean_PW": float(group["PW"].mean()) if "PW" in group else 0.0,
            "Var_PW": float(group["PW"].var(ddof=0)) if "PW" in group else 0.0,
            "Mean_PRI": float(group["PRI"].mean()) if "PRI" in group else 0.0,
            "Var_PRI": float(group["PRI"].var(ddof=0)) if "PRI" in group else 0.0,
        }
        rows.append(row)

    features = pd.DataFrame(rows).sort_values("Track_ID").reset_index(drop=True)
    feature_columns = ["Mean_RF", "Var_RF", "Mean_PW", "Var_PW", "Mean_PRI", "Var_PRI", "Pulse_Count"]
    return FeatureSet(features=features, feature_columns=feature_columns)

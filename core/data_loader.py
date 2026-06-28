import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class LoadedData:
    path: str
    filename: str
    data: pd.DataFrame
    fields: List[str]


ALIASES: Dict[str, Tuple[str, ...]] = {
    "TOA": ("toa", "toa_s", "toa_us", "TOA(s)", "time", "time_us", "arrival_time", "TOA"),
    "RF": ("rf", "rf_mhz", "freq", "frequency", "carrier_freq", "RF", "Param1"),
    "PW": ("pw", "pw_us", "pulse_width", "width", "PW", "Param2"),
    "PA": ("pa", "pa_db", "amplitude", "amp", "power", "PA", "Param4"),
    "DOA": ("doa", "doa_deg", "angle", "azimuth", "DOA", "Param5"),
    "LABEL": ("label", "class", "class_label", "target", "LABEL"),
    "SigIdx": ("sigidx", "sig_idx", "signal_id", "SigIdx"),
    "Track_ID": ("track_id", "track", "trackid", "Track_ID"),
    "Assigned": ("assigned", "Assigned"),
    "PRI": ("pri", "pri_us", "PRI", "Param3"),
}


def _canonical_name(name: str) -> str:
    key = name.strip().replace(" ", "_")
    lower = key.lower()
    for canonical, aliases in ALIASES.items():
        if lower in {alias.lower() for alias in aliases}:
            return canonical
    return key


def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)

    # The bundled PDW txt files are aligned with variable-width spaces.
    # Pandas' delimiter sniffer can misread them as many single-space columns,
    # so prefer explicit whitespace parsing before trying other separators.
    for sep in [r"\s+", ",", "\t", None]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    raise ValueError("无法识别文件分隔符，请检查 txt/csv 格式")


def load_pdw_file(path: str) -> LoadedData:
    df = _read_table(path)
    if df.empty:
        raise ValueError("文件为空或没有有效数据行")

    df = df.rename(columns={column: _canonical_name(str(column)) for column in df.columns})
    if "Track_ID" in df.columns and "Original_Track_ID" not in df.columns:
        df["Original_Track_ID"] = df["Track_ID"]
    for name in ["TOA", "RF", "PW", "PA", "DOA", "PRI", "Track_ID"]:
        if name in df.columns:
            df[name] = pd.to_numeric(df[name], errors="coerce")

    if "TOA" not in df.columns:
        raise ValueError("缺少必要字段 TOA")
    if "RF" not in df.columns:
        raise ValueError("缺少必要字段 RF")
    if "PW" not in df.columns:
        df["PW"] = 0.0
    if "PA" not in df.columns:
        df["PA"] = 0.0
    if "DOA" not in df.columns:
        df["DOA"] = np.nan

    df = df.dropna(subset=["TOA", "RF"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("TOA/RF 字段没有可用数值")

    if "Pulse_ID" not in df.columns:
        df.insert(0, "Pulse_ID", np.arange(1, len(df) + 1))

    df = df.sort_values("TOA").reset_index(drop=True)
    if "PRI" not in df.columns:
        pri = df["TOA"].diff().fillna(df["TOA"].diff().median())
        pri = pri.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["PRI"] = pri.clip(lower=0)

    # Do not automatically promote SigIdx to Track_ID at import time.
    # Track_ID represents current sorting output and is created only after
    # the user runs sorting analysis. SigIdx is kept as an original/reference
    # field for algorithms that want to use it.
    if "LABEL" in df.columns and "Predicted_Label" not in df.columns:
        df["Predicted_Label"] = df["LABEL"].astype(str)

    df.attrs["source_path"] = os.path.abspath(path)
    truth_path = os.path.join(os.path.dirname(os.path.abspath(path)), "Sorted_PDW.txt")
    if os.path.exists(truth_path):
        df.attrs["truth_path"] = truth_path

    return LoadedData(
        path=path,
        filename=os.path.basename(path),
        data=df,
        fields=list(df.columns),
    )

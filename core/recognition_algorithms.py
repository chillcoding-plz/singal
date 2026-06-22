import time
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from .external_model_adapters import run_zeng_recognition
from .feature_extractor import extract_track_features

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except Exception:  # pragma: no cover
    RandomForestClassifier = None
    KNeighborsClassifier = None
    StandardScaler = None
    SVC = None


@dataclass
class RecognitionOutput:
    data: pd.DataFrame
    track_results: pd.DataFrame
    model: str
    elapsed: float
    mean_confidence: float
    class_count: int
    summary: Dict[str, object] = field(default_factory=dict)


def _format_label(value):
    if pd.isna(value):
        return np.nan
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        label_id = int(numeric)
        return "Unknown" if label_id == 99 else f"Class_{label_id}"
    return str(value)


def _attach_truth_labels(out: pd.DataFrame, results: pd.DataFrame):
    if "True_Label" not in out.columns:
        return out, results, None
    truth_labels = out["True_Label"].map(_format_label)
    out["True_Label_Display"] = truth_labels
    accuracy = None
    if "Predicted_Label" in out.columns:
        valid = truth_labels.notna()
        if valid.any():
            accuracy = float((out.loc[valid, "Predicted_Label"].astype(str) == truth_labels[valid].astype(str)).mean())
    if not results.empty and "Track_ID" in results.columns and "Track_ID" in out.columns:
        track_ids = pd.to_numeric(out["Track_ID"], errors="coerce").fillna(0).astype(int)
        truth_by_track = {}
        grouped = pd.DataFrame({"Track_ID": track_ids, "True_Label": truth_labels})
        for track_id, group in grouped[grouped["Track_ID"] > 0].groupby("Track_ID", sort=False):
            labels = group["True_Label"].dropna()
            if not labels.empty:
                truth_by_track[int(track_id)] = str(labels.astype(str).mode().iloc[0])
        results = results.copy()
        result_tracks = pd.to_numeric(results["Track_ID"], errors="coerce").fillna(0).astype(int)
        results["True_Label"] = result_tracks.map(truth_by_track)
    return out, results, accuracy


def run_recognition(df: pd.DataFrame, model: str, progress_callback=None, should_cancel=None) -> RecognitionOutput:
    start = time.perf_counter()
    if model.lower() == "zeng":
        out, results = run_zeng_recognition(df, progress_callback=progress_callback, should_cancel=should_cancel)
        out, results, recognition_accuracy = _attach_truth_labels(out, results)
        if recognition_accuracy is not None:
            out.attrs["recognition_accuracy"] = recognition_accuracy
            results.attrs["recognition_accuracy"] = recognition_accuracy
        mean_conf = float(results["Confidence"].mean()) if not results.empty else 0.0
        class_count = int(results["Predicted_Label"].nunique()) if not results.empty else 0
        return RecognitionOutput(
            data=out,
            track_results=results,
            model=model,
            elapsed=time.perf_counter() - start,
            mean_confidence=mean_conf,
            class_count=class_count,
            summary={
                "识别轨迹数": int(len(results)),
                "已识别脉冲数": int((out["Confidence"] > 0).sum()) if "Confidence" in out else 0,
                "平均置信度": mean_conf,
                "类别数": class_count,
            },
        )
    if model.upper() == "CNN":
        raise NotImplementedError("CNN 模型接口待接入")

    feature_set = extract_track_features(df)
    features = feature_set.features
    labels = _heuristic_labels(features)
    confidence = _classify(features[feature_set.feature_columns], labels, model)

    results = features[["Track_ID", "Pulse_Count", "Mean_RF", "Mean_PW", "Mean_PRI"]].copy()
    results["Predicted_Label"] = labels
    results["Confidence"] = confidence

    out = df.copy()
    out["Predicted_Label"] = ""
    out["Confidence"] = 0.0
    for _, row in results.iterrows():
        mask = out["Track_ID"].astype(int) == int(row["Track_ID"])
        out.loc[mask, "Predicted_Label"] = row["Predicted_Label"]
        out.loc[mask, "Confidence"] = float(row["Confidence"])
    out, results, recognition_accuracy = _attach_truth_labels(out, results)
    if recognition_accuracy is not None:
        out.attrs["recognition_accuracy"] = recognition_accuracy
        results.attrs["recognition_accuracy"] = recognition_accuracy

    mean_conf = float(results["Confidence"].mean()) if not results.empty else 0.0
    class_count = int(results["Predicted_Label"].nunique())
    return RecognitionOutput(
        data=out,
        track_results=results,
        model=model,
        elapsed=time.perf_counter() - start,
        mean_confidence=mean_conf,
        class_count=class_count,
        summary={
            "识别轨迹数": int(len(results)),
            "已识别脉冲数": int((out["Confidence"] > 0).sum()),
            "平均置信度": mean_conf,
            "类别数": class_count,
        },
    )


def _heuristic_labels(features: pd.DataFrame) -> np.ndarray:
    labels = []
    for _, row in features.iterrows():
        rf = row["Mean_RF"]
        pw = row["Mean_PW"]
        if rf < 900:
            labels.append("Radar_A")
        elif rf < 1020:
            labels.append("Radar_B")
        elif pw > 2.3:
            labels.append("Comm_Pulse")
        else:
            labels.append("Unknown")
    return np.array(labels)


def _classify(x: pd.DataFrame, labels: np.ndarray, model: str) -> np.ndarray:
    if len(x) < 2 or len(set(labels)) < 2 or StandardScaler is None:
        return np.full(len(x), 0.88)

    scaled = StandardScaler().fit_transform(x)
    model_key = model.upper()
    if model_key == "KNN" and KNeighborsClassifier is not None:
        clf = KNeighborsClassifier(n_neighbors=min(3, len(x)))
    elif model_key == "随机森林" and RandomForestClassifier is not None:
        clf = RandomForestClassifier(n_estimators=60, random_state=42)
    elif SVC is not None:
        clf = SVC(probability=True, kernel="rbf", random_state=42)
    else:
        return np.full(len(x), 0.88)

    clf.fit(scaled, labels)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(scaled)
        return np.max(proba, axis=1).clip(0.5, 0.99)
    return np.full(len(x), 0.88)

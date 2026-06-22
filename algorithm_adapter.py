from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
from numpy.lib import recfunctions as rfn


@dataclass
class AlgorithmInput:
    """Data object passed from the UI to a sorting or recognition algorithm."""

    pdw_data: np.ndarray
    method: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SortingResult:
    """Expected sorting output returned by an external sorting algorithm."""

    track_ids: np.ndarray
    confidence: Optional[np.ndarray] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


@dataclass
class RecognitionResult:
    """Expected recognition output returned by an external recognition model."""

    class_labels: np.ndarray
    confidence: np.ndarray
    probabilities: Optional[np.ndarray] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None


class AlgorithmBridge:
    """
    Boundary between the PyQt UI and real algorithms.

    Replace the default runners by calling:
        bridge.register_sorting_runner(your_sorting_function)
        bridge.register_recognition_runner(your_recognition_function)

    Sorting runner signature:
        function(AlgorithmInput) -> SortingResult | dict

    Recognition runner signature:
        function(AlgorithmInput, sorting_result=None) -> RecognitionResult | dict
    """

    def __init__(self):
        self._sorting_runner: Optional[Callable[[AlgorithmInput], Any]] = None
        self._recognition_runner: Optional[Callable[[AlgorithmInput, Optional[SortingResult]], Any]] = None

    def register_sorting_runner(self, runner: Callable[[AlgorithmInput], Any]):
        self._sorting_runner = runner

    def register_recognition_runner(self, runner: Callable[[AlgorithmInput, Optional[SortingResult]], Any]):
        self._recognition_runner = runner

    def run_sorting(self, pdw_data: np.ndarray, method: str, params: Optional[Dict[str, Any]] = None) -> SortingResult:
        algorithm_input = AlgorithmInput(pdw_data=pdw_data, method=method, params=params or {})
        result = (
            self._sorting_runner(algorithm_input)
            if self._sorting_runner is not None
            else self._default_sorting(algorithm_input)
        )
        return self._coerce_sorting_result(result, len(pdw_data))

    def run_recognition(
        self,
        pdw_data: np.ndarray,
        model: str,
        sorting_result: Optional[SortingResult] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> RecognitionResult:
        algorithm_input = AlgorithmInput(pdw_data=pdw_data, method=model, params=params or {})
        result = (
            self._recognition_runner(algorithm_input, sorting_result)
            if self._recognition_runner is not None
            else self._default_recognition(algorithm_input, sorting_result)
        )
        return self._coerce_recognition_result(result, len(pdw_data))

    def _default_sorting(self, algorithm_input: AlgorithmInput) -> SortingResult:
        data = algorithm_input.pdw_data
        if "track_id" in data.dtype.names:
            track_ids = np.asarray(data["track_id"], dtype=int)
        elif "rf_mhz" in data.dtype.names:
            rf = np.asarray(data["rf_mhz"], dtype=float)
            bins = np.quantile(rf, [0.25, 0.5, 0.75])
            track_ids = np.digitize(rf, bins) + 1
        else:
            track_ids = np.ones(len(data), dtype=int)

        confidence = np.full(len(data), 0.9, dtype=float)
        return SortingResult(
            track_ids=track_ids,
            confidence=confidence,
            summary={"method": algorithm_input.method, "track_count": int(len(set(track_ids) - {0}))},
        )

    def _default_recognition(
        self,
        algorithm_input: AlgorithmInput,
        sorting_result: Optional[SortingResult],
    ) -> RecognitionResult:
        data = algorithm_input.pdw_data
        if "class_label" in data.dtype.names:
            labels = np.asarray(data["class_label"]).astype(str)
        else:
            tracks = sorting_result.track_ids if sorting_result is not None else np.ones(len(data), dtype=int)
            label_map = {
                1: "Radar_A",
                2: "Radar_B",
                3: "Comm_Pulse",
                4: "Unknown",
                0: "Unknown",
            }
            labels = np.array([label_map.get(int(track), "Unknown") for track in tracks])

        if "confidence" in data.dtype.names:
            confidence = np.asarray(data["confidence"], dtype=float)
        else:
            confidence = np.full(len(data), 0.88, dtype=float)

        return RecognitionResult(
            class_labels=labels,
            confidence=confidence,
            summary={"model": algorithm_input.method, "sample_count": int(len(data))},
        )

    def _coerce_sorting_result(self, result: Any, length: int) -> SortingResult:
        if isinstance(result, SortingResult):
            sorting_result = result
        elif isinstance(result, dict):
            sorting_result = SortingResult(
                track_ids=np.asarray(result.get("track_ids", result.get("track_id"))),
                confidence=None if result.get("confidence") is None else np.asarray(result["confidence"], dtype=float),
                summary=dict(result.get("summary", {})),
                raw=result.get("raw", result),
            )
        else:
            raise TypeError("Sorting algorithm must return SortingResult or dict")

        if len(sorting_result.track_ids) != length:
            raise ValueError("Sorting result length must match input PDW row count")
        if sorting_result.confidence is not None and len(sorting_result.confidence) != length:
            raise ValueError("Sorting confidence length must match input PDW row count")
        sorting_result.track_ids = np.asarray(sorting_result.track_ids, dtype=int)
        return sorting_result

    def _coerce_recognition_result(self, result: Any, length: int) -> RecognitionResult:
        if isinstance(result, RecognitionResult):
            recognition_result = result
        elif isinstance(result, dict):
            recognition_result = RecognitionResult(
                class_labels=np.asarray(result.get("class_labels", result.get("class_label"))).astype(str),
                confidence=np.asarray(result.get("confidence"), dtype=float),
                probabilities=result.get("probabilities"),
                summary=dict(result.get("summary", {})),
                raw=result.get("raw", result),
            )
        else:
            raise TypeError("Recognition algorithm must return RecognitionResult or dict")

        if len(recognition_result.class_labels) != length:
            raise ValueError("Recognition label length must match input PDW row count")
        if len(recognition_result.confidence) != length:
            raise ValueError("Recognition confidence length must match input PDW row count")
        return recognition_result


def with_updated_fields(data: np.ndarray, field_values: Dict[str, np.ndarray]) -> np.ndarray:
    """Return a structured array with fields appended or replaced."""

    updated = data
    for name, values in field_values.items():
        values = np.asarray(values)
        if len(values) != len(updated):
            raise ValueError(f"Field '{name}' length does not match data length")
        if name in updated.dtype.names:
            updated = rfn.drop_fields(updated, name, usemask=False)
        updated = rfn.append_fields(updated, name, values, usemask=False)
    return updated

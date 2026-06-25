import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from .recognition_algorithms import RecognitionOutput, run_recognition
from .sorting_algorithms import SortingOutput, run_sorting


ProgressCallback = Optional[Callable[[int, str], None]]
CancelCallback = Optional[Callable[[], bool]]


@dataclass(frozen=True)
class SortingStageDefinition:
    stage_id: str
    name: str
    sorter: str
    stage_key: str

    @property
    def label(self) -> str:
        return self.name


@dataclass
class PipelineStageResult:
    definition: SortingStageDefinition
    sorting: SortingOutput
    elapsed: float


@dataclass
class PipelineRunResult:
    pipeline_id: str
    pipeline_name: str
    sorting: SortingOutput
    recognition: RecognitionOutput
    stage_results: List[PipelineStageResult] = field(default_factory=list)
    elapsed: float = 0.0
    summary: Dict[str, object] = field(default_factory=dict)


@dataclass
class SortingPipelineResult:
    pipeline_id: str
    pipeline_name: str
    sorting: SortingOutput
    stage_results: List[PipelineStageResult] = field(default_factory=list)
    elapsed: float = 0.0
    summary: Dict[str, object] = field(default_factory=dict)


SORTING_STAGES: Dict[str, SortingStageDefinition] = {
    "hdbscan": SortingStageDefinition("hdbscan", "HDBSCAN", "HDBSCAN", "HDBSCAN"),
    "hdbscan_cycle_period": SortingStageDefinition("hdbscan_cycle_period", "HDBSCAN+cycle_period", "HDBSCAN", "CyclePeriod"),
    "hdbscan_cycle_period_mht": SortingStageDefinition("hdbscan_cycle_period_mht", "HDBSCAN+cycle_period+MHT", "HDBSCAN", "MHT"),
    "cycle_period": SortingStageDefinition("cycle_period", "cycle_period", "cycle_period", "CyclePeriod"),
    "mht": SortingStageDefinition("mht", "MHT", "MHT", "MHT"),
}

SORTING_PIPELINE_ID = "sorting_pipeline"
SORTING_PIPELINE_NAME = "预分选 → 主分选 → 细分选"
IMPORTED_HDBSCAN_PIPELINE_NAME = "预分选 → 主分选 → 细分选"
FULL_PIPELINE_ID = "full"
FULL_PIPELINE_NAME = "预分选 → 主分选 → 细分选 → 信号识别"
RECOGNITION_MODEL = "zeng"

PIPELINE_LABELS: Dict[str, str] = {SORTING_PIPELINE_ID: SORTING_PIPELINE_NAME}


def pipeline_id_from_label(label: str) -> str:
    for pipeline_id, text in PIPELINE_LABELS.items():
        if label == text:
            return pipeline_id
    return SORTING_PIPELINE_ID


def pipeline_label(pipeline_id: str) -> str:
    if pipeline_id == FULL_PIPELINE_ID:
        return FULL_PIPELINE_NAME
    return PIPELINE_LABELS.get(pipeline_id, SORTING_PIPELINE_NAME)


def pipeline_definition(pipeline_id: str) -> SortingStageDefinition:
    if pipeline_id == SORTING_PIPELINE_ID:
        return SORTING_STAGES["hdbscan_cycle_period_mht"]
    return SORTING_STAGES.get(pipeline_id, SORTING_STAGES["cycle_period"])


def run_sorting_pipeline(
    df: pd.DataFrame,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
    stream_callback=None,
    streaming_zeng: bool = False,
) -> SortingPipelineResult:
    start = time.perf_counter()
    stages = _run_sorting_stages(df, progress_callback, should_cancel, stream_callback, streaming_zeng=streaming_zeng)
    final_sorting = stages[-1].sorting
    elapsed = time.perf_counter() - start
    return SortingPipelineResult(
        pipeline_id=SORTING_PIPELINE_ID,
        pipeline_name=SORTING_PIPELINE_NAME,
        sorting=final_sorting,
        stage_results=stages,
        elapsed=elapsed,
        summary={
            "pipeline_id": SORTING_PIPELINE_ID,
            "pipeline_name": SORTING_PIPELINE_NAME,
            "stage_count": len(stages),
            "final_sorting_method": final_sorting.method,
            "elapsed": elapsed,
        },
    )


def run_cycle_period_pipeline_from_hdbscan(
    df: pd.DataFrame,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
) -> SortingPipelineResult:
    start = time.perf_counter()
    hdbscan_data = _add_stage_columns(df, SORTING_STAGES["hdbscan"])
    hdbscan_stage = PipelineStageResult(
        definition=SORTING_STAGES["hdbscan"],
        sorting=_sorting_output_from_data(hdbscan_data, "HDBSCAN", 0.0),
        elapsed=0.0,
    )
    _emit(progress_callback, 0, "HDBSCAN：使用导入的第一阶段结果")
    cycle_stage = _run_sort_stage(
        hdbscan_data,
        SORTING_STAGES["cycle_period"],
        progress_callback,
        should_cancel,
        5,
        99,
    )
    cycle_stage.sorting.data = _add_stage_columns(cycle_stage.sorting.data, SORTING_STAGES["cycle_period"])
    cycle_view = cycle_stage.sorting.data.copy()
    if "CyclePeriod_Track_ID" in cycle_view.columns:
        cycle_tracks = pd.to_numeric(cycle_view["CyclePeriod_Track_ID"], errors="coerce").fillna(0).astype(int)
        cycle_view["Track_ID"] = cycle_tracks
        cycle_view["Assigned"] = cycle_view.get("CyclePeriod_Assigned", cycle_tracks > 0)
        cycle_view["Sorting_Method"] = "cycle_period-200ms"
    cycle_view = cycle_view.drop(
        columns=[column for column in cycle_view.columns if column.startswith("MHT_") or column == "Display_Track_ID"],
        errors="ignore",
    )
    cycle_stage_view = PipelineStageResult(
        definition=SORTING_STAGES["hdbscan_cycle_period"],
        sorting=_sorting_output_from_data(_add_stage_columns(cycle_view, SORTING_STAGES["hdbscan_cycle_period"]), "cycle_period-200ms", cycle_stage.elapsed),
        elapsed=cycle_stage.elapsed,
    )
    mht_stage = PipelineStageResult(SORTING_STAGES["mht"], cycle_stage.sorting, cycle_stage.elapsed)
    stages = [hdbscan_stage, cycle_stage_view, mht_stage]
    final_sorting = cycle_stage.sorting
    elapsed = time.perf_counter() - start
    return SortingPipelineResult(
        pipeline_id=SORTING_PIPELINE_ID,
        pipeline_name=IMPORTED_HDBSCAN_PIPELINE_NAME,
        sorting=final_sorting,
        stage_results=stages,
        elapsed=elapsed,
        summary={
            "pipeline_id": SORTING_PIPELINE_ID,
            "pipeline_name": IMPORTED_HDBSCAN_PIPELINE_NAME,
            "stage_count": len(stages),
            "final_sorting_method": final_sorting.method,
            "elapsed": elapsed,
        },
    )


def run_pipeline(
    df: pd.DataFrame,
    pipeline_id: str = FULL_PIPELINE_ID,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
    stream_callback=None,
    streaming_zeng_done: bool = False,
) -> PipelineRunResult:
    start = time.perf_counter()
    if pipeline_id != FULL_PIPELINE_ID:
        pipeline_id = FULL_PIPELINE_ID

    sorting_pipeline = run_sorting_pipeline(
        df,
        progress_callback=_scale_progress(progress_callback, 0, 78, ""),
        should_cancel=should_cancel,
        stream_callback=stream_callback,
        streaming_zeng=streaming_zeng_done,
    )
    return _run_recognition_after_sorting_pipeline(
        sorting_pipeline,
        FULL_PIPELINE_NAME,
        start,
        progress_callback,
        should_cancel,
        streaming_zeng_done=streaming_zeng_done,
    )


def run_pipeline_from_hdbscan(
    df: pd.DataFrame,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
) -> PipelineRunResult:
    start = time.perf_counter()
    sorting_pipeline = run_cycle_period_pipeline_from_hdbscan(
        df,
        progress_callback=progress_callback,
        should_cancel=should_cancel,
    )
    return _run_recognition_after_sorting_pipeline(
        sorting_pipeline,
        f"{IMPORTED_HDBSCAN_PIPELINE_NAME} → zeng",
        start,
        progress_callback,
        should_cancel,
    )


def _run_recognition_after_sorting_pipeline(
    sorting_pipeline: SortingPipelineResult,
    pipeline_name: str,
    start: float,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
    streaming_zeng_done: bool = False,
) -> PipelineRunResult:
    final_sorting = sorting_pipeline.sorting
    data = final_sorting.data

    if streaming_zeng_done:
        _emit(progress_callback, 85, "复用流式识别结果")
        from .recognition_algorithms import _build_recognition_from_labeled_data
        recognition = _build_recognition_from_labeled_data(data)
    else:
        _emit(progress_callback, 79, "准备识别分选结果")
        recognition = run_recognition(
            data,
            RECOGNITION_MODEL,
            progress_callback=_scale_progress(progress_callback, 79, 98, "zeng"),
            should_cancel=should_cancel,
        )
    _check_cancelled(should_cancel)

    elapsed = time.perf_counter() - start
    return PipelineRunResult(
        pipeline_id=FULL_PIPELINE_ID,
        pipeline_name=pipeline_name,
        sorting=final_sorting,
        recognition=recognition,
        stage_results=sorting_pipeline.stage_results,
        elapsed=elapsed,
        summary={
            "pipeline_id": FULL_PIPELINE_ID,
            "pipeline_name": pipeline_name,
            "stage_count": len(sorting_pipeline.stage_results),
            "recognition_model": RECOGNITION_MODEL,
            "elapsed": elapsed,
        },
    )


def _run_sorting_stages(
    df: pd.DataFrame,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
    stream_callback=None,
    streaming_zeng: bool = False,
) -> List[PipelineStageResult]:
    stages: List[PipelineStageResult] = []
    current = df.copy()
    order = [SORTING_STAGES["hdbscan_cycle_period_mht"]]
    ranges = [(0, 100)]

    for definition, (start_pct, end_pct) in zip(order, ranges):
        _check_cancelled(should_cancel)
        stage_stream_callback = stream_callback if definition.sorter.lower() == "hdbscan" else None
        stage = _run_sort_stage(current, definition, progress_callback, should_cancel, start_pct, end_pct, stage_stream_callback, streaming_zeng=streaming_zeng)
        stage.sorting.data = _add_stage_columns(stage.sorting.data, definition)
        if definition.stage_id == "hdbscan_cycle_period_mht":
            stages.extend(_split_hdbscan_cycle_period_mht_stage(stage))
        else:
            stages.append(stage)
        current = stage.sorting.data
    return stages


def _split_hdbscan_cycle_period_mht_stage(stage: PipelineStageResult) -> List[PipelineStageResult]:
    data = stage.sorting.data
    if "HDBSCAN_Track_ID" not in data.columns:
        return [stage]

    hdbscan_data = data.copy()
    hdbscan_tracks = pd.to_numeric(hdbscan_data["HDBSCAN_Track_ID"], errors="coerce").fillna(0).astype(int)
    hdbscan_data["Track_ID"] = hdbscan_tracks
    hdbscan_data["Assigned"] = (
        hdbscan_data["HDBSCAN_Assigned"]
        if "HDBSCAN_Assigned" in hdbscan_data.columns
        else hdbscan_tracks > 0
    )
    hdbscan_data["Sorting_Method"] = (
        str(hdbscan_data["HDBSCAN_Sorting_Method"].dropna().iloc[0])
        if "HDBSCAN_Sorting_Method" in hdbscan_data.columns and not hdbscan_data["HDBSCAN_Sorting_Method"].dropna().empty
        else "HDBSCAN"
    )
    hdbscan_data = hdbscan_data.drop(
        columns=[
            column
            for column in hdbscan_data.columns
            if column.startswith("CyclePeriod_") or column.startswith("MHT_") or column == "Display_Track_ID"
        ],
        errors="ignore",
    )

    hdbscan_sorting = _sorting_output_from_data(
        _add_stage_columns(hdbscan_data, SORTING_STAGES["hdbscan"]),
        "HDBSCAN",
        stage.sorting.elapsed,
    )

    cycle_data = data.copy()
    if "CyclePeriod_Track_ID" in cycle_data.columns:
        cycle_tracks = pd.to_numeric(cycle_data["CyclePeriod_Track_ID"], errors="coerce").fillna(0).astype(int)
        cycle_data["Track_ID"] = cycle_tracks
        cycle_data["Assigned"] = (
            cycle_data["CyclePeriod_Assigned"]
            if "CyclePeriod_Assigned" in cycle_data.columns
            else cycle_tracks > 0
        )
        cycle_data["Sorting_Method"] = (
            str(cycle_data["CyclePeriod_Sorting_Method"].dropna().iloc[0])
            if "CyclePeriod_Sorting_Method" in cycle_data.columns and not cycle_data["CyclePeriod_Sorting_Method"].dropna().empty
            else "cycle_period-200ms"
        )
    cycle_data = cycle_data.drop(
        columns=[column for column in cycle_data.columns if column.startswith("MHT_") or column == "Display_Track_ID"],
        errors="ignore",
    )
    cycle_sorting = _sorting_output_from_data(
        _add_stage_columns(cycle_data, SORTING_STAGES["hdbscan_cycle_period"]),
        "cycle_period-200ms",
        stage.sorting.elapsed,
    )

    final_sorting = stage.sorting
    return [
        PipelineStageResult(SORTING_STAGES["hdbscan"], hdbscan_sorting, stage.elapsed),
        PipelineStageResult(SORTING_STAGES["hdbscan_cycle_period"], cycle_sorting, stage.elapsed),
        PipelineStageResult(SORTING_STAGES["mht"], final_sorting, stage.elapsed),
    ]


def _run_sort_stage(
    df: pd.DataFrame,
    definition: SortingStageDefinition,
    progress_callback: ProgressCallback = None,
    should_cancel: CancelCallback = None,
    start_pct: int = 0,
    end_pct: int = 98,
    stream_callback=None,
    streaming_zeng: bool = False,
) -> PipelineStageResult:
    start = time.perf_counter()
    _emit(progress_callback, start_pct, "准备分选")
    sorting = run_sorting(
        df,
        definition.sorter,
        progress_callback=_scale_progress(progress_callback, start_pct, end_pct, ""),
        should_cancel=should_cancel,
        stream_callback=stream_callback,
        streaming_zeng=streaming_zeng,
    )
    _check_cancelled(should_cancel)
    return PipelineStageResult(
        definition=definition,
        sorting=sorting,
        elapsed=time.perf_counter() - start,
    )


def _add_stage_columns(df: pd.DataFrame, definition: SortingStageDefinition) -> pd.DataFrame:
    out = df.copy()
    prefix = definition.stage_key
    if "Track_ID" in out.columns and f"{prefix}_Track_ID" not in out.columns:
        out[f"{prefix}_Track_ID"] = out["Track_ID"]
    if "Assigned" in out.columns and f"{prefix}_Assigned" not in out.columns:
        out[f"{prefix}_Assigned"] = out["Assigned"]
    if f"{prefix}_Sorting_Method" not in out.columns:
        out[f"{prefix}_Sorting_Method"] = definition.sorter
    return out


def _sorting_output_from_data(df: pd.DataFrame, method: str, elapsed: float) -> SortingOutput:
    tracks = pd.to_numeric(df["Track_ID"], errors="coerce").fillna(0).astype(int) if "Track_ID" in df else pd.Series([0] * len(df))
    assigned = tracks > 0
    track_count = int(tracks[assigned].nunique())
    assigned_count = int(assigned.sum())
    unassigned_count = int(len(df) - assigned_count)
    return SortingOutput(
        data=df.copy(),
        method=method,
        elapsed=elapsed,
        track_count=track_count,
        assigned_count=assigned_count,
        unassigned_count=unassigned_count,
        summary={
            "分选轨迹数": track_count,
            "已分配脉冲数": assigned_count,
            "未分配脉冲数": unassigned_count,
            "平均 PRI": float(df["PRI"].mean()) if "PRI" in df else 0.0,
            "平均脉宽": float(df["PW"].mean()) if "PW" in df else 0.0,
        },
    )


def _scale_progress(
    progress_callback: ProgressCallback,
    start_pct: int,
    end_pct: int,
    prefix: str,
) -> ProgressCallback:
    if progress_callback is None:
        return None

    def emit(value: int, text: str) -> None:
        value = max(0, min(100, int(value)))
        scaled = start_pct + int((end_pct - start_pct) * value / 100)
        if prefix:
            progress_callback(max(0, min(100, scaled)), f"{prefix}：{text}")
        else:
            progress_callback(max(0, min(100, scaled)), text)

    return emit


def _emit(progress_callback: ProgressCallback, value: int, text: str) -> None:
    if progress_callback is not None:
        progress_callback(max(0, min(100, int(value))), text)


def _check_cancelled(should_cancel: CancelCallback) -> None:
    if should_cancel is not None and should_cancel():
        raise RuntimeError("任务已取消")

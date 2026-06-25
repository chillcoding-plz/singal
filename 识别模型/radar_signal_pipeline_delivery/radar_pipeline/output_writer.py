"""
输出写入器

版本化输出目录，包含运行清单、配置快照和守恒验证。

修复原代码:
  - 块级输出基于窗口 ID 构建，禁止重复计数
  - 跨块模式段按块内窗口重新计算
  - 守恒约束: sum(block.n_windows) == total_windows
"""
from __future__ import annotations
import os
import json
import csv
import hashlib
import math
import platform
import numpy as np
from datetime import datetime
from typing import Optional, Any
from .schemas import (
    BlockOutput, ModeTimelineEntry, AttributeTimelineEntry,
    RunManifest, StateSegment, ModeResult, AttributeResult, WindowFeatures,
)


def create_run_directory(base_dir: str = "artifacts/runs") -> str:
    """创建版本化运行目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "timelines"), exist_ok=True)
    return run_dir


def save_manifest(
    run_dir: str,
    manifest: RunManifest,
):
    """保存运行清单"""
    path = os.path.join(run_dir, "manifest.json")
    data = {
        "run_id": manifest.run_id,
        "timestamp": manifest.timestamp,
        "code_version": manifest.code_version,
        "python_version": manifest.python_version,
        "input_files": manifest.input_files,
        "input_hashes": manifest.input_hashes,
        "config_hash": manifest.config_hash,
        "total_windows": manifest.total_windows,
        "total_pulses": manifest.total_pulses,
        "elapsed_seconds": manifest.elapsed_seconds,
        "parameters": manifest.parameters,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_config_snapshot(run_dir: str, config_path: str):
    """保存配置快照"""
    import shutil
    dst = os.path.join(run_dir, "config_snapshot.yaml")
    shutil.copy2(config_path, dst)


def compute_file_hash(path: str) -> str:
    """计算文件 SHA256"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_blocks(
    radar_key: str,
    window_ids: np.ndarray,
    segments: list[StateSegment],
    mode_results: list[ModeResult],
    attr_results: list[AttributeResult],
    window_features: list[WindowFeatures],
    block_duration: float = 5.0,
) -> list[BlockOutput]:
    """
    按窗口 ID 构建块输出。

    修复原代码:
      - 每个窗口只属于一个块
      - 跨块模式段按块内窗口重新计算 n_windows 和 n_pulses
      - 守恒约束
    """
    if len(window_ids) == 0:
        return []

    # 窗口 ID 范围
    min_wid = int(window_ids.min())
    max_wid = int(window_ids.max())

    # 每块的窗口数
    windows_per_block = max(1, int(round(block_duration / 0.2)))

    # 构建窗口 ID 到索引的映射
    wid_to_idx = {int(wid): i for i, wid in enumerate(window_ids)}

    blocks: list[BlockOutput] = []
    block_idx = 0

    for block_start_wid in range(min_wid, max_wid + 1, windows_per_block):
        block_end_wid = block_start_wid + windows_per_block

        # 块内窗口
        block_wids = [
            wid for wid in range(block_start_wid, block_end_wid)
            if wid in wid_to_idx
        ]

        if not block_wids:
            continue

        # 块内窗口索引
        block_indices = [wid_to_idx[wid] for wid in block_wids]

        # 块内窗口特征
        block_features = [window_features[i] for i in block_indices]

        # 块内脉冲数 (守恒: 每个窗口只属于一个块)
        block_n_pulses = sum(f.n_pulses for f in block_features)
        block_n_windows = len(block_wids)

        # 块内模式段 (裁剪并按块内窗口重新计算)
        block_mode_entries = _clip_mode_segments_to_block(
            segments, mode_results, block_wids, wid_to_idx,
        )

        # 块内属性时间线
        block_attr_entries = _clip_attr_to_block(
            attr_results, block_wids, block_indices, block_features,
        )

        # 块级汇总
        summary = _compute_block_summary(
            block_mode_entries, block_attr_entries,
            block_n_windows, block_n_pulses,
        )

        sample, rid = radar_key.split("/")

        blocks.append(BlockOutput(
            block_index=block_idx,
            radar_id=rid,
            sample=sample,
            time_start=float(block_start_wid) * 0.2,
            time_end=float(block_end_wid) * 0.2,
            mode_timeline=block_mode_entries,
            attribute_timeline=block_attr_entries,
            n_windows=block_n_windows,
            n_pulses=block_n_pulses,
            block_summary=summary,
        ))

        block_idx += 1

    return blocks


def _clip_mode_segments_to_block(
    segments: list[StateSegment],
    mode_results: list[ModeResult],
    block_wids: list[int],
    wid_to_idx: dict[int, int],
) -> list[ModeTimelineEntry]:
    """将模式段裁剪到块内，并按块内窗口重新计算"""
    entries = []
    block_wid_set = set(block_wids)

    for seg, mr in zip(segments, mode_results):
        # 段内属于该块的窗口
        seg_block_wids = [wid for wid in seg.window_ids if wid in block_wid_set]
        if not seg_block_wids:
            continue

        # 按块内窗口重新计算
        block_indices = [wid_to_idx[wid] for wid in seg_block_wids]

        # 计算块内脉冲数 (从段的脉冲数按窗口比例估算)
        seg_total_wids = len(seg.window_ids)
        block_wids_in_seg = len(seg_block_wids)
        seg_n_pulses = seg.n_pulses
        block_n_pulses = int(seg_n_pulses * block_wids_in_seg / max(seg_total_wids, 1))

        entries.append(ModeTimelineEntry(
            segment_id=seg.segment_id,
            start_time=float(seg_block_wids[0]) * 0.2,
            end_time=float(seg_block_wids[-1] + 1) * 0.2,
            duration_s=len(seg_block_wids) * 0.2,
            mode_result=mr,
            window_ids=seg_block_wids,
            n_pulses=block_n_pulses,
            feature_summary=seg.feature_summary,
        ))

    return entries


def _clip_attr_to_block(
    attr_results: list[AttributeResult],
    block_wids: list[int],
    block_indices: list[int],
    block_features: list[WindowFeatures],
) -> list[AttributeTimelineEntry]:
    """将属性结果裁剪到块内"""
    if not attr_results:
        return []

    entries = []
    windows_per_attr = 5

    for i in range(0, len(block_indices), windows_per_attr):
        chunk_wids = block_wids[i:i + windows_per_attr]
        chunk_features = block_features[i:i + windows_per_attr]

        # attr_results 是按全局窗口序列每 5 个窗口生成一次的。
        # 这里必须使用全局窗口索引定位，不能按块内 i 重新从 0 开始。
        first_global_index = block_indices[i]
        attr_idx = first_global_index // windows_per_attr
        if attr_idx < len(attr_results):
            ar = attr_results[attr_idx]
        else:
            # 尾部
            ar = AttributeResult(
                decision="unknown",
                attr="未知",
                best_guess="",
                attr_scores={},
                margin=0.0,
                signal_scores={},
                mode_context_scores={},
                reason="尾部窗口",
            )

        n_pulses = sum(f.n_pulses for f in chunk_features)

        entries.append(AttributeTimelineEntry(
            window_ids=chunk_wids,
            start_time=float(chunk_wids[0]) * 0.2 if chunk_wids else 0.0,
            end_time=float(chunk_wids[-1] + 1) * 0.2 if chunk_wids else 0.0,
            duration_s=len(chunk_wids) * 0.2,
            attr_result=ar,
            n_pulses=n_pulses,
        ))

    return entries


def _compute_block_summary(
    mode_entries: list[ModeTimelineEntry],
    attr_entries: list[AttributeTimelineEntry],
    n_windows: int,
    n_pulses: int,
) -> dict:
    """计算块级汇总"""
    summary: dict = {
        "n_windows": n_windows,
        "n_pulses": n_pulses,
    }

    # 模式分布
    if mode_entries:
        mode_counter: dict[str, int] = {}
        for entry in mode_entries:
            m = entry.mode_result.mode
            mode_counter[m] = mode_counter.get(m, 0) + len(entry.window_ids)

        total_mode_windows = sum(mode_counter.values())
        summary["mode_distribution"] = {
            m: round(c / max(total_mode_windows, 1), 3)
            for m, c in mode_counter.items()
        }
        summary["dominant_mode"] = max(mode_counter, key=mode_counter.get)
        summary["mode_transition_count"] = sum(
            1 for i in range(1, len(mode_entries))
            if mode_entries[i].mode_result.mode != mode_entries[i - 1].mode_result.mode
        )
        summary["unknown_mode_windows"] = sum(
            len(e.window_ids) for e in mode_entries
            if "未知" in e.mode_result.mode
        )

    # 属性分布
    if attr_entries:
        attr_counter: dict[str, int] = {}
        for entry in attr_entries:
            a = entry.attr_result.attr
            attr_counter[a] = attr_counter.get(a, 0) + 1

        total_attr = len(attr_entries)
        summary["attr_distribution"] = {
            a: round(c / max(total_attr, 1), 3)
            for a, c in attr_counter.items()
        }
        summary["dominant_attr"] = max(attr_counter, key=attr_counter.get)

    return summary


def verify_block_conservation(
    blocks: list[BlockOutput],
    total_windows: int,
    total_pulses: int,
) -> dict:
    """
    验证块级守恒约束。

    Returns:
        {windows_match, pulses_match, block_windows_sum, block_pulses_sum}
    """
    block_windows_sum = sum(b.n_windows for b in blocks)
    block_pulses_sum = sum(b.n_pulses for b in blocks)

    return {
        "total_windows": total_windows,
        "total_pulses": total_pulses,
        "block_windows_sum": block_windows_sum,
        "block_pulses_sum": block_pulses_sum,
        "windows_match": block_windows_sum == total_windows,
        "pulses_match": block_pulses_sum == total_pulses,
    }


def save_radar_output(
    run_dir: str,
    radar_key: str,
    blocks: list[BlockOutput],
    summary: dict,
    conservation: dict,
):
    """保存单部雷达的输出"""
    sample, rid = radar_key.split("/")
    radar_dir = os.path.join(run_dir, "timelines", sample, rid)
    os.makedirs(radar_dir, exist_ok=True)

    # 保存每个块
    for block in blocks:
        block_path = os.path.join(radar_dir, f"block_{block.block_index}.json")
        block_data = {
            "block_index": block.block_index,
            "radar_id": block.radar_id,
            "sample": block.sample,
            "time_start": block.time_start,
            "time_end": block.time_end,
            "n_windows": block.n_windows,
            "n_pulses": block.n_pulses,
            "mode_timeline": [
                {
                    "segment_id": e.segment_id,
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "duration_s": e.duration_s,
                    "mode": e.mode_result.mode,
                    "decision": e.mode_result.decision,
                    "evidence_score": e.mode_result.evidence_score,
                    "window_ids": e.window_ids,
                    "n_pulses": e.n_pulses,
                }
                for e in block.mode_timeline
            ],
            "attribute_timeline": [
                {
                    "start_time": e.start_time,
                    "end_time": e.end_time,
                    "attr": e.attr_result.attr,
                    "decision": e.attr_result.decision,
                    "n_pulses": e.n_pulses,
                }
                for e in block.attribute_timeline
            ],
            "block_summary": block.block_summary,
        }
        with open(block_path, "w", encoding="utf-8") as f:
            json.dump(block_data, f, indent=2, ensure_ascii=False)

    # 保存雷达级汇总
    summary_path = os.path.join(radar_dir, "summary.json")
    full_summary = {
        "radar_id": rid,
        "sample": sample,
        **summary,
        "conservation": conservation,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)


def save_global_time_display(
    run_dir: str,
    all_blocks: dict[str, list[BlockOutput]],
    display_interval: float = 30.0,
) -> dict[str, Any]:
    """
    保存跨雷达时间显示输出。

    每 display_interval 秒生成一个全局帧，帧内列出所有雷达在该时间段的
    主导工作模式和功能属性，同时保留区间内的模式/属性占比。
    """
    if display_interval <= 0:
        raise ValueError("display_interval must be positive")

    display_dir = os.path.join(run_dir, "display_frames")
    os.makedirs(display_dir, exist_ok=True)

    non_empty_blocks = [block for blocks in all_blocks.values() for block in blocks]
    if not non_empty_blocks:
        index_data = {
            "display_interval": display_interval,
            "n_frames": 0,
            "frames": [],
        }
        index_path = os.path.join(display_dir, "index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        return index_data

    timeline_end_times = [
        entry.end_time
        for block in non_empty_blocks
        for entry in [*block.mode_timeline, *block.attribute_timeline]
    ]
    max_time = max(timeline_end_times) if timeline_end_times else max(
        block.time_end for block in non_empty_blocks
    )
    n_frames = max(1, int(np.ceil(max_time / display_interval)))
    frames: list[dict[str, Any]] = []
    csv_rows = [
        "frame_index,time_start,time_end,sample,radar_id,radar_key,"
        "work_mode,mode_decision,mode_evidence_score,function_attribute,attr_decision"
    ]

    for frame_index in range(n_frames):
        time_start = round(frame_index * display_interval, 6)
        time_end = round(min((frame_index + 1) * display_interval, max_time), 6)
        frame = {
            "frame_index": frame_index,
            "time_start": time_start,
            "time_end": time_end,
            "display_interval": display_interval,
            "radars": [],
        }

        for radar_key in sorted(all_blocks):
            blocks = all_blocks[radar_key]
            sample, rid = radar_key.split("/")
            mode_info = _summarize_mode_for_interval(blocks, time_start, time_end)
            attr_info = _summarize_attr_for_interval(blocks, time_start, time_end)

            radar_state = {
                "sample": sample,
                "radar_id": rid,
                "radar_key": radar_key,
                "work_mode": mode_info["label"],
                "mode_decision": mode_info["decision"],
                "mode_evidence_score": mode_info["score"],
                "mode_distribution": mode_info["distribution"],
                "function_attribute": attr_info["label"],
                "attr_decision": attr_info["decision"],
                "attr_distribution": attr_info["distribution"],
                "has_data": mode_info["has_data"] or attr_info["has_data"],
            }
            frame["radars"].append(radar_state)

            csv_rows.append(
                ",".join([
                    str(frame_index),
                    f"{time_start:.3f}",
                    f"{time_end:.3f}",
                    sample,
                    rid,
                    radar_key,
                    radar_state["work_mode"],
                    radar_state["mode_decision"],
                    f"{radar_state['mode_evidence_score']:.3f}",
                    radar_state["function_attribute"],
                    radar_state["attr_decision"],
                ])
            )

        frame_path = os.path.join(display_dir, f"frame_{frame_index:04d}.json")
        with open(frame_path, "w", encoding="utf-8") as f:
            json.dump(frame, f, indent=2, ensure_ascii=False)
        frames.append({
            "frame_index": frame_index,
            "time_start": time_start,
            "time_end": time_end,
            "path": os.path.relpath(frame_path, run_dir),
        })

    index_data = {
        "display_interval": display_interval,
        "n_frames": len(frames),
        "frames": frames,
    }

    index_path = os.path.join(display_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(display_dir, "timeline_display.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_rows))

    md_path = os.path.join(display_dir, "timeline_display.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 雷达工作模式与功能属性时间显示\n\n")
        f.write(f"- 保存间隔: {display_interval:.1f}s\n")
        f.write(f"- 帧数: {len(frames)}\n\n")
        for frame in frames:
            frame_path = os.path.join(run_dir, frame["path"])
            with open(frame_path, "r", encoding="utf-8") as jf:
                frame_data = json.load(jf)
            f.write(
                f"## {frame_data['time_start']:.1f}s - "
                f"{frame_data['time_end']:.1f}s\n\n"
            )
            f.write("| 雷达 | 工作模式 | 模式判定 | 功能属性 | 属性判定 |\n")
            f.write("|---|---|---|---|---|\n")
            for state in frame_data["radars"]:
                f.write(
                    f"| {state['radar_key']} | {state['work_mode']} | "
                    f"{state['mode_decision']} | {state['function_attribute']} | "
                    f"{state['attr_decision']} |\n"
                )
            f.write("\n")

    return index_data


def save_analysis_report(
    run_dir: str,
    all_summaries: dict[str, dict],
    all_blocks: dict[str, list[BlockOutput]],
    manifest: RunManifest,
    display_index: dict[str, Any],
    llm_labels_path: Optional[str] = None,
):
    """保存完整 Markdown 分析报告。"""
    report_path = os.path.join(run_dir, "report.md")
    llm_labels = _load_llm_labels(llm_labels_path)
    radar_segments = {
        key: _annotate_segments_with_llm_accuracy(
            _merge_adjacent_segments(_collect_report_segments(blocks)),
            llm_labels,
        )
        for key, blocks in all_blocks.items()
    }
    all_segments = [
        segment
        for segments in radar_segments.values()
        for segment in segments
    ]
    evidence_scores = [
        segment["confidence"]
        for segment in all_segments
        if segment["confidence"] is not None
    ]
    mode_decisions = _count_values(segment["mode_decision"] for segment in all_segments)
    attr_decisions = _count_values(segment["attr_decision"] for segment in all_segments)
    unknown_segments = [
        segment for segment in all_segments
        if segment["mode_decision"] == "unknown"
        or segment["attr_decision"] in {"unknown", "pending"}
        or "未知" in segment["mode"]
        or "未知" in segment["attribute"]
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 雷达信号识别系统 - 完整分析报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 一、系统概览\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 雷达数 | {len(all_summaries)} |\n")
        f.write(f"| 总脉冲数 | {manifest.total_pulses:,} |\n")
        f.write(f"| 总窗口数 | {manifest.total_windows:,} |\n")
        f.write(f"| 输出段数 | {len(all_segments)} |\n")
        f.write(f"| 30秒显示帧 | {display_index.get('n_frames', 0)} |\n")
        if llm_labels:
            f.write("| 准确率定义 | 规则输出相对大模型参考伪标签的连续软评分；模式权重0.6，属性权重0.4，待定/未知给部分分 |\n")
        else:
            f.write("| 准确率定义 | 未提供大模型标签，表中退回显示系统置信度 |\n")
        f.write(f"| 运行版本 | {manifest.code_version} |\n\n")

        f.write("## 二、识别证据表现\n\n")
        f.write("### 工作模式识别\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| known段 | {mode_decisions.get('known', 0)} |\n")
        f.write(f"| suspected段 | {mode_decisions.get('suspected', 0)} |\n")
        f.write(f"| unknown段 | {mode_decisions.get('unknown', 0)} |\n")
        f.write(f"| 模式分布 | {_format_counter(_count_values(s['mode'] for s in all_segments))} |\n\n")

        f.write("### 功能属性识别\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| known段 | {attr_decisions.get('known', 0)} |\n")
        f.write(f"| pending段 | {attr_decisions.get('pending', 0)} |\n")
        f.write(f"| unknown段 | {attr_decisions.get('unknown', 0)} |\n")
        f.write(f"| 属性分布 | {_format_counter(_count_values(s['attribute'] for s in all_segments))} |\n\n")

        f.write("## 三、证据分数统计\n\n")
        f.write("| 指标 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 均值 | {_safe_stat(evidence_scores, 'mean')} |\n")
        f.write(f"| 中位数 | {_safe_stat(evidence_scores, 'median')} |\n")
        f.write(f"| 最小值 | {_safe_stat(evidence_scores, 'min')} |\n")
        f.write(f"| 最大值 | {_safe_stat(evidence_scores, 'max')} |\n")
        f.write(f"| 标准差 | {_safe_stat(evidence_scores, 'std')} |\n")
        f.write(f"| <0.6比例 | {_ratio(sum(1 for s in evidence_scores if s < 0.6), len(evidence_scores))} |\n\n")

        f.write("## 四、未知与待定分析\n\n")
        f.write(f"未知/待定/疑似需要复核的段数: {len(unknown_segments)}\n\n")
        f.write("| 雷达 | 时间 | 模式 | 模式判定 | 属性 | 属性判定 | 证据分数 |\n")
        f.write("|-----|------|------|---------|------|---------|---------|\n")
        for segment in unknown_segments:
            f.write(
                f"| {segment['radar_key']} | "
                f"{segment['start_time']:.3f}~{segment['end_time']:.3f} | "
                f"{segment['mode']} | {segment['mode_decision']} | "
                f"{segment['attribute']} | {segment['attr_decision']} | "
                f"{segment['confidence']:.3f} |\n"
            )
        if not unknown_segments:
            f.write("| - | - | - | - | - | - | - |\n")
        f.write("\n")

        f.write("## 五、逐雷达详细分析\n\n")
        if llm_labels:
            f.write("说明: 下表中的“准确率”为规则识别结果与大模型参考伪标签的连续软评分；"
                    "模式权重0.6，属性权重0.4，“待定”和“未知”按不确定程度给部分分。\n\n")
        else:
            f.write("说明: 未提供大模型标签，下表中的“准确率”退回为系统估计置信度。\n\n")
        for key in sorted(all_summaries):
            summary = all_summaries[key]
            segments = radar_segments.get(key, [])
            mode_counter = _count_values(segment["mode"] for segment in segments)
            attr_counter = _count_values(segment["attribute"] for segment in segments)
            scores = [s["accuracy"] for s in segments if s["accuracy"] is not None]
            mode_scores = [
                s["mode_accuracy"] for s in segments
                if s.get("mode_accuracy") is not None
            ]
            attr_scores = [
                s["attr_accuracy"] for s in segments
                if s.get("attr_accuracy") is not None
            ]
            unknown_count = sum(
                1 for s in segments
                if s["mode_decision"] == "unknown"
                or s["attr_decision"] in {"unknown", "pending"}
                or "未知" in s["mode"]
                or "未知" in s["attribute"]
            )

            f.write(f"### {key}\n\n")
            f.write("| 指标 | 值 |\n")
            f.write("|------|-----|\n")
            f.write(f"| 脉冲数 | {summary.get('total_pulses', 0):,} |\n")
            f.write(f"| 窗口数 | {summary.get('total_windows', 0)} |\n")
            f.write(f"| 有效窗口率 | {_pct(summary.get('quality', {}).get('valid_ratio', 0))} |\n")
            f.write(f"| 段数 | {len(segments)} |\n")
            f.write(f"| 需复核段 | {unknown_count} |\n")
            if llm_labels:
                f.write(f"| 平均模式准确率 | {_safe_stat(mode_scores, 'mean')} |\n")
                f.write(f"| 平均属性准确率 | {_safe_stat(attr_scores, 'mean')} |\n")
                f.write(f"| 平均联合准确率 | {_safe_stat(scores, 'mean')} |\n")
                f.write(f"| 最低联合准确率 | {_safe_stat(scores, 'min')} |\n")
            else:
                f.write(f"| 平均证据分数 | {_safe_stat(scores, 'mean')} |\n")
                f.write(f"| 最低证据分数 | {_safe_stat(scores, 'min')} |\n")
            f.write(f"| 模式分布 | {_format_counter(mode_counter)} |\n")
            f.write(f"| 属性分布 | {_format_counter(attr_counter)} |\n\n")

            if llm_labels:
                f.write("| # | 时间 | 脉冲 | 模式 | 属性 | 模式准确率 | 属性准确率 | 联合准确率 | 参考模式 | 参考属性 | 来源 |\n")
                f.write("|---|------|------|------|------|------------|------------|------------|----------|----------|------|\n")
            else:
                f.write("| # | 时间 | 脉冲 | 模式 | 属性 | 证据分数 | 来源 |\n")
                f.write("|---|------|------|------|------|----------|------|\n")
            for idx, segment in enumerate(segments, start=1):
                if llm_labels:
                    f.write(
                        f"| {idx} | {segment['start_time']:.3f}~{segment['end_time']:.3f} | "
                        f"{segment['n_pulses']:,} | {segment['mode']} | "
                        f"{segment['attribute']} | "
                        f"{_fmt_optional_float(segment.get('mode_accuracy'))} | "
                        f"{_fmt_optional_float(segment.get('attr_accuracy'))} | "
                        f"{_fmt_optional_float(segment.get('joint_accuracy'))} | "
                        f"{segment.get('llm_mode', '')} | {segment.get('llm_attr', '')} | "
                        f"{segment['source']} |\n"
                    )
                else:
                    f.write(
                        f"| {idx} | {segment['start_time']:.3f}~{segment['end_time']:.3f} | "
                        f"{segment['n_pulses']:,} | {segment['mode']} | "
                        f"{segment['attribute']} | {segment['accuracy']:.3f} | "
                        f"{segment['source']} |\n"
                    )
            if not segments:
                f.write("| - | - | - | - | - | - | - |\n")
            f.write("\n")

        f.write("## 六、总结\n\n")
        f.write("| 维度 | 状态 | 说明 |\n")
        f.write("|------|------|------|\n")
        f.write("| 工程输出 | OK | 窗口/脉冲守恒，报告与时间显示均已生成 |\n")
        f.write("| 准确率 | 仅伪标签一致率 | 缺少真实人工标签，当前只能报告规则输出与参考伪标签的一致率 |\n")
        f.write("| 模式识别 | 可用但需复核 | 模式多数可匹配，未知段和低质量段仍需复核 |\n")
        f.write("| 属性识别 | 需重点优化 | 属性分布若集中到单一类别，说明规则区分度不足 |\n")
        f.write("| 未知/待定 | 需复核 | unknown、pending和suspected段应进入人工或专家规则复核 |\n")


def save_segment_timeline_tables(
    run_dir: str,
    all_blocks: dict[str, list[BlockOutput]],
    llm_labels_path: Optional[str] = None,
):
    """保存便于读取的逐雷达时间段识别表。"""
    tables_dir = os.path.join(run_dir, "segment_tables")
    os.makedirs(tables_dir, exist_ok=True)
    llm_labels = _load_llm_labels(llm_labels_path)

    all_rows: list[dict[str, Any]] = []
    for radar_key in sorted(all_blocks):
        segments = _annotate_segments_with_llm_accuracy(
            _merge_adjacent_segments(_collect_report_segments(all_blocks[radar_key])),
            llm_labels,
        )
        safe_name = radar_key.replace("/", "_")
        md_path = os.path.join(tables_dir, f"{safe_name}.md")
        csv_path = os.path.join(tables_dir, f"{safe_name}.csv")
        _write_segment_md(md_path, radar_key, segments)
        _write_segment_csv(csv_path, radar_key, segments)
        all_rows.extend(segments)

    _write_segment_md(
        os.path.join(tables_dir, "all_radars.md"),
        "all_radars",
        all_rows,
        include_radar=True,
    )
    _write_segment_csv(
        os.path.join(tables_dir, "all_radars.csv"),
        "all_radars",
        all_rows,
        include_radar=True,
    )


def save_llm_reference_tables(
    run_dir: str,
    llm_labels_path: Optional[str] = None,
):
    """保存大模型参考伪标签的连续时间段表，不参与规则输出覆盖。"""
    llm_labels = _load_llm_labels(llm_labels_path)
    if not llm_labels:
        return

    tables_dir = os.path.join(run_dir, "llm_reference_tables")
    os.makedirs(tables_dir, exist_ok=True)

    all_segments: list[dict[str, Any]] = []
    for radar_key in sorted(llm_labels):
        segments = _merge_llm_label_segments(radar_key, llm_labels[radar_key])
        safe_name = radar_key.replace("/", "_")
        _write_reference_segment_md(
            os.path.join(tables_dir, f"{safe_name}.md"),
            radar_key,
            segments,
        )
        _write_reference_segment_csv(
            os.path.join(tables_dir, f"{safe_name}.csv"),
            radar_key,
            segments,
        )
        all_segments.extend(segments)

    _write_reference_segment_md(
        os.path.join(tables_dir, "all_radars.md"),
        "all_radars",
        all_segments,
        include_radar=True,
    )
    _write_reference_segment_csv(
        os.path.join(tables_dir, "all_radars.csv"),
        "all_radars",
        all_segments,
        include_radar=True,
    )


def _summarize_mode_for_interval(
    blocks: list[BlockOutput],
    start: float,
    end: float,
) -> dict[str, Any]:
    entries = [entry for block in blocks for entry in block.mode_timeline]
    if not entries:
        return _empty_interval_summary()

    distribution = _duration_distribution(entries, start, end, lambda e: e.mode_result.mode)
    current = _dominant_entry(entries, start, end, lambda e: e.mode_result.mode)
    if current is None:
        return _empty_interval_summary()

    return {
        "label": current.mode_result.mode,
        "decision": current.mode_result.decision,
        "score": float(current.mode_result.evidence_score),
        "distribution": distribution,
        "has_data": True,
    }


def _summarize_attr_for_interval(
    blocks: list[BlockOutput],
    start: float,
    end: float,
) -> dict[str, Any]:
    entries = [entry for block in blocks for entry in block.attribute_timeline]
    if not entries:
        return _empty_interval_summary()

    distribution = _duration_distribution(entries, start, end, lambda e: e.attr_result.attr)
    current = _dominant_entry(entries, start, end, lambda e: e.attr_result.attr)
    if current is None:
        return _empty_interval_summary()

    return {
        "label": current.attr_result.attr,
        "decision": current.attr_result.decision,
        "score": 0.0,
        "distribution": distribution,
        "has_data": True,
    }


def _duration_distribution(
    entries: list[Any],
    start: float,
    end: float,
    label_fn,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for entry in entries:
        overlap = max(0.0, min(entry.end_time, end) - max(entry.start_time, start))
        if overlap <= 0:
            continue
        label = label_fn(entry)
        totals[label] = totals.get(label, 0.0) + overlap

    total = sum(totals.values())
    if total <= 0:
        return {}
    return {label: round(value / total, 3) for label, value in totals.items()}


def _dominant_entry(
    entries: list[Any],
    start: float,
    end: float,
    label_fn,
) -> Optional[Any]:
    overlaps: dict[str, float] = {}
    label_to_entry: dict[str, Any] = {}
    for entry in entries:
        overlap = max(0.0, min(entry.end_time, end) - max(entry.start_time, start))
        if overlap <= 0:
            continue
        label = label_fn(entry)
        overlaps[label] = overlaps.get(label, 0.0) + overlap
        label_to_entry.setdefault(label, entry)

    if not overlaps:
        return None
    dominant_label = max(overlaps, key=overlaps.get)
    return label_to_entry[dominant_label]


def _empty_interval_summary() -> dict[str, Any]:
    return {
        "label": "无数据",
        "decision": "no_data",
        "score": 0.0,
        "distribution": {},
        "has_data": False,
    }


def _collect_report_segments(blocks: list[BlockOutput]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    attr_entries = [
        entry
        for block in blocks
        for entry in block.attribute_timeline
    ]
    for block in blocks:
        for entry in block.mode_timeline:
            attr = _dominant_entry(
                attr_entries, entry.start_time, entry.end_time,
                lambda e: e.attr_result.attr,
            )
            if attr is None:
                attr_label = "无数据"
                attr_decision = "no_data"
            else:
                attr_label = attr.attr_result.attr
                attr_decision = attr.attr_result.decision

            radar_key = f"{block.sample}/{block.radar_id}"
            segments.append({
                "radar_key": radar_key,
                "start_time": entry.start_time,
                "end_time": entry.end_time,
                "n_pulses": entry.n_pulses,
                "mode": entry.mode_result.mode,
                "mode_decision": entry.mode_result.decision,
                "attribute": attr_label,
                "attr_decision": attr_decision,
                "confidence": float(entry.mode_result.evidence_score),
                "accuracy": float(entry.mode_result.evidence_score),
                "mode_accuracy": None,
                "attr_accuracy": None,
                "joint_accuracy": None,
                "llm_mode": "",
                "llm_attr": "",
                "source": "unknown" if entry.mode_result.decision == "unknown" else "merged",
            })
    return segments


def _merge_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """合并相邻且模式/属性/来源相同的段，让阅读表更紧凑。"""
    if not segments:
        return []

    merged: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda s: (s["radar_key"], s["start_time"])):
        if not merged:
            merged.append(dict(segment))
            continue

        prev = merged[-1]
        is_same_label = (
            prev["radar_key"] == segment["radar_key"]
            and prev["mode"] == segment["mode"]
            and prev["attribute"] == segment["attribute"]
            and prev["source"] == segment["source"]
        )
        is_contiguous = abs(prev["end_time"] - segment["start_time"]) <= 1e-6
        if is_same_label and is_contiguous:
            total_pulses = prev["n_pulses"] + segment["n_pulses"]
            prev_duration = max(prev["end_time"] - prev["start_time"], 0.0)
            cur_duration = max(segment["end_time"] - segment["start_time"], 0.0)
            total_duration = prev_duration + cur_duration
            if total_duration > 0:
                prev["confidence"] = (
                    prev["confidence"] * prev_duration
                    + segment["confidence"] * cur_duration
                ) / total_duration
                prev["accuracy"] = (
                    prev["accuracy"] * prev_duration
                    + segment["accuracy"] * cur_duration
                ) / total_duration
            prev["end_time"] = segment["end_time"]
            prev["n_pulses"] = total_pulses
        else:
            merged.append(dict(segment))

    return merged


def _write_segment_md(
    path: str,
    title: str,
    segments: list[dict[str, Any]],
    include_radar: bool = False,
):
    has_eval = any(
        segment.get("mode_accuracy") is not None
        or segment.get("attr_accuracy") is not None
        or segment.get("llm_mode")
        or segment.get("llm_attr")
        for segment in segments
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title} 时间段识别结果\n\n")
        if has_eval:
            f.write(
                "说明: 这里是规则识别输出；模式/属性准确率均为相对大模型参考伪标签的连续软评分，"
                "“待定”和“未知”按不确定程度给部分分。\n\n"
            )
        else:
            f.write("说明: 未提供大模型参考标签时，“准确率”退回显示系统证据分数。\n\n")

        if include_radar:
            if has_eval:
                f.write("| # | 雷达 | 时间 | 脉冲 | 模式 | 属性 | 模式准确率 | 属性准确率 | 联合准确率 | 参考模式 | 参考属性 | 来源 |\n")
                f.write("|---|------|------|------|------|------|------------|------------|------------|----------|----------|------|\n")
            else:
                f.write("| # | 雷达 | 时间 | 脉冲 | 模式 | 属性 | 准确率 | 来源 |\n")
                f.write("|---|------|------|------|------|------|--------|------|\n")
        elif has_eval:
            f.write("| # | 时间 | 脉冲 | 模式 | 属性 | 模式准确率 | 属性准确率 | 联合准确率 | 参考模式 | 参考属性 | 来源 |\n")
            f.write("|---|------|------|------|------|------------|------------|------------|----------|----------|------|\n")
        else:
            f.write("| # | 时间 | 脉冲 | 模式 | 属性 | 准确率 | 来源 |\n")
            f.write("|---|------|------|------|------|--------|------|\n")

        for idx, segment in enumerate(segments, start=1):
            prefix = f"| {idx} | "
            if include_radar:
                prefix += f"{segment['radar_key']} | "
            if has_eval:
                f.write(
                    prefix
                    + f"{segment['start_time']:.3f}~{segment['end_time']:.3f} | "
                    + f"{segment['n_pulses']:,} | "
                    + f"{segment['mode']} | {segment['attribute']} | "
                    + f"{_fmt_optional_float(segment.get('mode_accuracy'))} | "
                    + f"{_fmt_optional_float(segment.get('attr_accuracy'))} | "
                    + f"{_fmt_optional_float(segment.get('joint_accuracy'))} | "
                    + f"{segment.get('llm_mode', '')} | {segment.get('llm_attr', '')} | "
                    + f"{segment['source']} |\n"
                )
            else:
                f.write(
                    prefix
                    + f"{segment['start_time']:.3f}~{segment['end_time']:.3f} | "
                    + f"{segment['n_pulses']:,} | "
                    + f"{segment['mode']} | {segment['attribute']} | "
                    + f"{segment['accuracy']:.3f} | {segment['source']} |\n"
                )


def _write_segment_csv(
    path: str,
    radar_key: str,
    segments: list[dict[str, Any]],
    include_radar: bool = False,
):
    fieldnames = ["index"]
    if include_radar:
        fieldnames.append("radar_key")
    fieldnames.extend([
        "time",
        "start_time",
        "end_time",
        "n_pulses",
        "mode",
        "attribute",
        "accuracy",
        "mode_accuracy",
        "attr_accuracy",
        "joint_accuracy",
        "llm_mode",
        "llm_attr",
        "source",
    ])

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, segment in enumerate(segments, start=1):
            row = {
                "index": idx,
                "time": f"{segment['start_time']:.3f}~{segment['end_time']:.3f}",
                "start_time": f"{segment['start_time']:.3f}",
                "end_time": f"{segment['end_time']:.3f}",
                "n_pulses": segment["n_pulses"],
                "mode": segment["mode"],
                "attribute": segment["attribute"],
                "accuracy": f"{segment['accuracy']:.3f}",
                "mode_accuracy": _fmt_optional_float(segment.get("mode_accuracy")),
                "attr_accuracy": _fmt_optional_float(segment.get("attr_accuracy")),
                "joint_accuracy": _fmt_optional_float(segment.get("joint_accuracy")),
                "llm_mode": segment.get("llm_mode", ""),
                "llm_attr": segment.get("llm_attr", ""),
                "source": segment["source"],
            }
            if include_radar:
                row["radar_key"] = segment["radar_key"]
            writer.writerow(row)


def _write_reference_segment_md(
    path: str,
    title: str,
    segments: list[dict[str, Any]],
    include_radar: bool = False,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title} 大模型参考伪标签时间段\n\n")
        f.write("说明: 该表仅为参考伪标签，不是规则模型输出，也不参与覆盖规则识别结果。\n\n")
        if include_radar:
            f.write("| # | 雷达 | 时间 | 参考模式 | 参考属性 | 伪标签置信度 | 来源 |\n")
            f.write("|---|------|------|----------|----------|--------------|------|\n")
        else:
            f.write("| # | 时间 | 参考模式 | 参考属性 | 伪标签置信度 | 来源 |\n")
            f.write("|---|------|----------|----------|--------------|------|\n")

        for idx, segment in enumerate(segments, start=1):
            prefix = f"| {idx} | "
            if include_radar:
                prefix += f"{segment['radar_key']} | "
            f.write(
                prefix
                + f"{segment['start_time']:.3f}~{segment['end_time']:.3f} | "
                + f"{segment['mode']} | {segment['attribute']} | "
                + f"{segment['confidence']:.3f} | {segment['source']} |\n"
            )


def _write_reference_segment_csv(
    path: str,
    radar_key: str,
    segments: list[dict[str, Any]],
    include_radar: bool = False,
):
    fieldnames = ["index"]
    if include_radar:
        fieldnames.append("radar_key")
    fieldnames.extend([
        "time",
        "start_time",
        "end_time",
        "reference_mode",
        "reference_attribute",
        "reference_confidence",
        "source",
    ])

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, segment in enumerate(segments, start=1):
            row = {
                "index": idx,
                "time": f"{segment['start_time']:.3f}~{segment['end_time']:.3f}",
                "start_time": f"{segment['start_time']:.3f}",
                "end_time": f"{segment['end_time']:.3f}",
                "reference_mode": segment["mode"],
                "reference_attribute": segment["attribute"],
                "reference_confidence": f"{segment['confidence']:.3f}",
                "source": segment["source"],
            }
            if include_radar:
                row["radar_key"] = segment["radar_key"]
            writer.writerow(row)


def _load_llm_labels(path: Optional[str]) -> dict[str, list[dict[str, Any]]]:
    if not path or not os.path.exists(path):
        return {}

    labels: dict[str, list[dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            record_id = str(row.get("record_id", ""))
            parts = record_id.split("/")
            if len(parts) != 3 or not parts[2].startswith("w"):
                continue
            try:
                window_index = int(parts[2][1:])
            except ValueError:
                continue
            radar_key = f"{parts[0]}/{parts[1]}"
            labels.setdefault(radar_key, []).append({
                "start_time": float(window_index),
                "end_time": float(window_index + 1),
                "mode": str(row.get("llm_work_mode", "")).strip(),
                "attr": str(row.get("llm_func_attr", "")).strip(),
                "confidence": float(row.get("llm_confidence", 0.0) or 0.0),
                "n_pulses": 0,
            })

    for radar_rows in labels.values():
        radar_rows.sort(key=lambda item: item["start_time"])
    return labels


def _merge_llm_label_segments(
    radar_key: str,
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for label in labels:
        segment = {
            "radar_key": radar_key,
            "start_time": label["start_time"],
            "end_time": label["end_time"],
            "n_pulses": label.get("n_pulses", 0),
            "mode": label["mode"],
            "mode_decision": "llm",
            "attribute": label["attr"],
            "attr_decision": "llm",
            "confidence": label.get("confidence", 1.0),
            "accuracy": 1.0,
            "mode_accuracy": 1.0,
            "attr_accuracy": 1.0,
            "joint_accuracy": 1.0,
            "llm_mode": label["mode"],
            "llm_attr": label["attr"],
            "source": "llm_reference",
        }
        if not segments:
            segments.append(segment)
            continue
        prev = segments[-1]
        if (
            prev["mode"] == segment["mode"]
            and prev["attribute"] == segment["attribute"]
            and abs(prev["end_time"] - segment["start_time"]) <= 1e-6
        ):
            prev_duration = prev["end_time"] - prev["start_time"]
            cur_duration = segment["end_time"] - segment["start_time"]
            total_duration = prev_duration + cur_duration
            if total_duration > 0:
                prev["confidence"] = (
                    prev["confidence"] * prev_duration
                    + segment["confidence"] * cur_duration
                ) / total_duration
            prev["end_time"] = segment["end_time"]
        else:
            segments.append(segment)
    return segments


def _annotate_segments_with_llm_accuracy(
    segments: list[dict[str, Any]],
    llm_labels: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if not llm_labels:
        return segments

    annotated = []
    for segment in segments:
        row = dict(segment)
        matches = _segment_llm_matches(row, llm_labels.get(row["radar_key"], []))
        row.update(matches)
        annotated.append(row)
    return annotated


def _segment_llm_matches(
    segment: dict[str, Any],
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    逐窗口比较规则输出与 LLM 伪标签。

    评价口径：
      - LLM 伪标签只作为参考标签，不覆盖规则输出。
      - 输出连续软分数，避免段级结果只有 0/1。
      - 工作模式权重 0.6，功能属性权重 0.4。
      - 即使标签一致，也会按规则证据分数和参考标签置信度折算。
      - “未知/未知”是有效未知识别，但分值低于已知类精确命中。
      - “待定”表示属性无法细分，只给部分分，不再一刀切为 0 或 1。
    """
    valid_total = 0.0
    mode_match = 0.0
    attr_match = 0.0
    joint_match = 0.0
    mode_votes: dict[str, float] = {}
    attr_votes: dict[str, float] = {}
    rule_mode = _normalize_mode(segment["mode"])
    rule_attr = _normalize_attr(segment["attribute"])
    rule_confidence = _bounded_float(segment.get("confidence", 0.0))

    empty_labels = {"", "无标签", "无数据"}

    for label in labels:
        overlap = max(
            0.0,
            min(segment["end_time"], label["end_time"])
            - max(segment["start_time"], label["start_time"]),
        )
        if overlap <= 0:
            continue
        llm_mode = _normalize_mode(label["mode"])
        llm_attr = _normalize_attr(label["attr"])
        if llm_mode in empty_labels and llm_attr in empty_labels:
            continue

        mode_votes[label["mode"]] = mode_votes.get(label["mode"], 0.0) + overlap
        attr_votes[label["attr"]] = attr_votes.get(label["attr"], 0.0) + overlap
        valid_total += overlap

        ref_confidence = _bounded_float(label.get("confidence", 0.0))
        mode_score = _soft_mode_score(rule_mode, llm_mode, rule_confidence, ref_confidence)
        attr_score = _soft_attr_score(rule_attr, llm_attr, rule_confidence, ref_confidence)
        joint_score = (
            0.6 * mode_score
            + 0.4 * attr_score
        )
        mode_match += overlap * mode_score
        attr_match += overlap * attr_score
        joint_match += overlap * joint_score

    if valid_total <= 0:
        return {
            "accuracy": 0.0,
            "mode_accuracy": 0.0,
            "attr_accuracy": 0.0,
            "joint_accuracy": 0.0,
            "llm_mode": max(mode_votes, key=mode_votes.get) if mode_votes else "无标签",
            "llm_attr": max(attr_votes, key=attr_votes.get) if attr_votes else "无标签",
        }

    mode_accuracy = mode_match / valid_total
    attr_accuracy = attr_match / valid_total
    joint_accuracy = joint_match / valid_total
    return {
        "accuracy": joint_accuracy,
        "mode_accuracy": mode_accuracy,
        "attr_accuracy": attr_accuracy,
        "joint_accuracy": joint_accuracy,
        "llm_mode": max(mode_votes, key=mode_votes.get) if mode_votes else "",
        "llm_attr": max(attr_votes, key=attr_votes.get) if attr_votes else "",
    }


def _normalize_mode(label: str) -> str:
    label = str(label).strip()
    if label.startswith("疑似"):
        label = label[2:]
    return label


def _normalize_attr(label: str) -> str:
    return str(label).strip()


def _soft_mode_score(
    rule_mode: str,
    ref_mode: str,
    rule_confidence: float = 0.0,
    ref_confidence: float = 0.0,
) -> float:
    """连续模式评分，精确匹配也按双方置信度折算。"""
    rule_mode = _normalize_mode(rule_mode)
    ref_mode = _normalize_mode(ref_mode)
    if not rule_mode or not ref_mode or rule_mode in {"无标签", "无数据"} or ref_mode in {"无标签", "无数据"}:
        return 0.0
    confidence_factor = _confidence_factor(rule_confidence, ref_confidence)
    if rule_mode == ref_mode:
        base = 0.65 if rule_mode == "未知" else 0.72
        span = 0.20 if rule_mode == "未知" else 0.26
        return base + span * confidence_factor
    if "未知" in {rule_mode, ref_mode}:
        return 0.12 + 0.12 * confidence_factor
    related = {
        ("搜索", "跟踪"),
        ("跟踪", "搜索"),
        ("跟踪", "制导"),
        ("制导", "跟踪"),
    }
    if (rule_mode, ref_mode) in related:
        return 0.35 + 0.12 * confidence_factor
    return 0.06 + 0.06 * confidence_factor


def _soft_attr_score(
    rule_attr: str,
    ref_attr: str,
    rule_confidence: float = 0.0,
    ref_confidence: float = 0.0,
) -> float:
    """连续功能属性评分，待定/未知/精确匹配都按置信度折算。"""
    rule_attr = _normalize_attr(rule_attr)
    ref_attr = _normalize_attr(ref_attr)
    empty = {"", "无标签", "无数据"}
    if rule_attr in empty or ref_attr in empty:
        return 0.0
    confidence_factor = _confidence_factor(rule_confidence, ref_confidence)
    if rule_attr == ref_attr:
        if rule_attr == "待定":
            return 0.35 + 0.18 * confidence_factor
        if rule_attr == "未知":
            return 0.58 + 0.20 * confidence_factor
        return 0.72 + 0.26 * confidence_factor
    if "待定" in {rule_attr, ref_attr}:
        other = ref_attr if rule_attr == "待定" else rule_attr
        if other == "未知":
            return 0.16 + 0.10 * confidence_factor
        return 0.30 + 0.16 * confidence_factor
    if "未知" in {rule_attr, ref_attr}:
        return 0.08 + 0.08 * confidence_factor
    search_attrs = {"对空搜索", "对海搜索"}
    if rule_attr in search_attrs and ref_attr in search_attrs:
        return 0.24 + 0.12 * confidence_factor
    if rule_attr == "火控" or ref_attr == "火控":
        return 0.06 + 0.08 * confidence_factor
    return 0.12 + 0.08 * confidence_factor


def _confidence_factor(rule_confidence: float, ref_confidence: float) -> float:
    rule = _bounded_float(rule_confidence)
    ref = _bounded_float(ref_confidence)
    if rule <= 0 and ref <= 0:
        return 0.0
    if rule <= 0:
        return ref
    if ref <= 0:
        return rule
    return math.sqrt(rule * ref)


def _bounded_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(out):
        return 0.0
    return max(0.0, min(1.0, out))


def _fmt_optional_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return ""


def _count_values(values) -> dict[str, int]:
    counter: dict[str, int] = {}
    for value in values:
        key = str(value)
        counter[key] = counter.get(key, 0) + 1
    return counter


def _format_counter(counter: dict[str, int]) -> str:
    if not counter:
        return "{}"
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return "{" + ", ".join(f"'{key}': {value}" for key, value in items) + "}"


def _safe_stat(values: list[float], kind: str) -> str:
    if not values:
        return "N/A"
    arr = np.array(values, dtype=float)
    if kind == "mean":
        value = float(np.mean(arr))
    elif kind == "median":
        value = float(np.median(arr))
    elif kind == "min":
        value = float(np.min(arr))
    elif kind == "max":
        value = float(np.max(arr))
    elif kind == "std":
        value = float(np.std(arr))
    else:
        value = float("nan")
    return f"{value:.3f}"


def _ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{count / total:.1%}"


def _pct(value: float) -> str:
    return f"{float(value):.1%}"


def save_global_summary(
    run_dir: str,
    all_summaries: dict[str, dict],
    manifest: RunManifest,
):
    """保存全局汇总"""
    global_summary = {
        "version": "v3.0",
        "run_id": manifest.run_id,
        "summary_type": "dominant_overview",
        "note": "This file reports dominant mode/attribute per radar. Time-segment details are in segment_tables/all_radars.md.",
        "total_radars": len(all_summaries),
        "total_pulses": manifest.total_pulses,
        "total_windows": manifest.total_windows,
        "elapsed_seconds": manifest.elapsed_seconds,
        "radar_summary": {},
    }

    for key, summary in all_summaries.items():
        diagnostics = summary.get("diagnostics", {})
        conservation = summary.get("conservation", {})
        global_summary["radar_summary"][key] = {
            "dominant_mode": summary.get("dominant_mode", "未知"),
            "dominant_attr": summary.get("global_func_attr", "未知"),
            "mode_known_coverage": diagnostics.get("known_coverage", 0),
            "mode_unknown_ratio": diagnostics.get("unknown_ratio", 0),
            "known_coverage": diagnostics.get("known_coverage", 0),
            "unknown_ratio": diagnostics.get("unknown_ratio", 0),
            "n_blocks": summary.get("n_blocks", 0),
            "conservation_ok": (
                conservation.get("windows_match", False)
                and conservation.get("pulses_match", False)
            ),
        }

    path = os.path.join(run_dir, "global_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)

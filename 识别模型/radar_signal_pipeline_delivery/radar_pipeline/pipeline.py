"""
新管线主入口 v3.0

完整流程:
  数据加载 → 窗口质量检查 → 窗口级特征 → 跨窗口时序特征 →
  变化点检测 → 状态段构建 → 段级模式识别 → 时序解码 →
  功能属性识别 → 块级输出 → 验证诊断

关键改进:
  - TOA 全链路 float64
  - 保留空窗口和窗口编号
  - 先构建状态段，后识别模式
  - 多证据评分 + 三级输出 (known/suspected/unknown)
  - 功能属性独立化 (85% 信号 + 15% 模式)
  - 块级守恒验证
  - 各项指标独立报告
"""
from __future__ import annotations
import os
import json
import time
import platform
import numpy as np
from pathlib import Path
from typing import Callable, Optional

from .schemas import (
    WindowRecord, WindowFeatures, TemporalFeatures,
    ChangePoint, StateSegment, ModeResult, AttributeResult,
    BlockOutput, RunManifest,
)
from .input_adapter import load_and_window, load_presegmented_files, verify_conservation
from .quality import check_window, compute_quality_summary
from .window_features import compute_features_batch, build_valid_mask
from .temporal_features import compute_temporal_features
from .change_detection import detect_change_points
from .state_segments import (
    build_state_segments, limited_merge_segments, build_fixed_window_segments,
)
from .mode_evidence import ModeEvidenceEngine
from .temporal_decoder import temporal_decode
from .function_attribute import FunctionAttributeEngine, compute_global_attribute
from .validation import compute_diagnostics, compare_with_degenerate_baselines
from .output_writer import (
    create_run_directory, save_manifest, save_config_snapshot,
    compute_file_hash, build_blocks, verify_block_conservation,
    save_radar_output, save_global_summary, save_global_time_display,
    save_analysis_report, save_segment_timeline_tables,
    save_llm_reference_tables,
)


CODE_VERSION = "v3.0"


def run_pipeline(
    input_files: list[str],
    config_path: Optional[str] = None,
    block_duration: float = 5.0,
    output_dir: str = "artifacts/runs",
    change_method: str = "auto",
    display_interval: float = 30.0,
    llm_labels_path: Optional[str] = "artifacts/llm_eval/v2.9/strict_codex_llm_labels.jsonl",
    pre_segmented: bool = False,
    partial_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """
    v3.0 管线执行。

    Args:
        input_files: 输入文件路径列表
        config_path: 配置文件路径 (可选)
        block_duration: 块时长 (秒)
        output_dir: 输出目录
        change_method: 变化点检测方法 ("auto", "custom", "ruptures")
        display_interval: 跨雷达显示保存间隔 (秒)
        llm_labels_path: 大模型伪标签路径，用于计算对比准确率
        pre_segmented: 输入是否已按 200ms 切分 (每文件一个窗口)

    Returns:
        {radar_key: summary_dict}
    """
    start_time = time.time()

    print("\n" + "=" * 60)
    print(f"  雷达信号识别管线 {CODE_VERSION}")
    print("=" * 60)

    # ── 创建运行目录 ──
    run_dir = create_run_directory(output_dir)
    print(f"\n  运行目录: {run_dir}")

    # ── 加载数据 ──
    print("\n  [1/10] 加载数据...")
    if pre_segmented:
        print("    模式: 200ms 预分段文件 (每文件一个窗口)")
        all_radars = load_presegmented_files(input_files, min_pulses=100)
    else:
        all_radars = load_and_window(input_files, min_pulses=100)

    # 统计脉冲数 (用于守恒验证)
    original_counts = {}
    for key, windows in all_radars.items():
        original_counts[key] = sum(w.n_pulses for w in windows)
        n_empty = sum(1 for w in windows if w.is_empty)
        print(f"    {key}: {len(windows)} 窗口 ({n_empty} 空), "
              f"{original_counts[key]:,} 脉冲")

    # 守恒验证
    conservation = verify_conservation(all_radars, original_counts)
    for key, report in conservation.items():
        if not report["match"]:
            print(f"    [WARN] {key}: 脉冲数不匹配!")

    # ── 逐雷达处理 ──
    all_summaries = {}
    all_blocks = {}

    for key, windows in all_radars.items():
        print(f"\n  处理 {key}...")
        sample, rid = key.split("/")

        # Stage 2: 窗口质量检查
        print(f"    [2/10] 窗口质量检查...")
        for w in windows:
            w.quality_flags = check_window(w)
        quality = compute_quality_summary(windows)
        print(f"    窗口: {quality['total']} 总, {quality['empty']} 空, "
              f"{quality['valid']} 有效 ({quality['valid_ratio']:.1%})")

        # Stage 3: 窗口级特征
        print(f"    [3/10] 计算窗口级特征...")
        window_features = compute_features_batch(windows)
        valid_mask = build_valid_mask(window_features)
        n_valid = int(valid_mask.sum())
        print(f"    有效窗口: {n_valid}")

        # Stage 4: 跨窗口时序特征
        print(f"    [4/10] 计算跨窗口时序特征...")
        window_ids = np.array([w.window_id for w in windows])
        temporal_features = compute_temporal_features(window_features, window_ids)

        # Stage 5: 变化点检测
        print(f"    [5/10] 变化点检测 (方法={change_method})...")
        change_points = detect_change_points(
            window_features, valid_mask, method=change_method,
        )
        # 辐射边界由 build_state_segments 内部处理 (带 min_silent_gap 过滤)
        # 不在此处添加，避免过度分段
        all_cps = change_points
        print(f"    变化点: {len(change_points)} 特征变化点")

        # Stage 6: 状态段构建
        print(f"    [6/10] 构建状态段...")
        segments = build_state_segments(
            window_features, window_ids, all_cps, radar_id=key,
        )
        segments = limited_merge_segments(segments)
        state_segments = segments
        segments = build_fixed_window_segments(
            window_features, window_ids, radar_id=key, windows_per_segment=5,
        )
        print(f"    状态段: {len(state_segments)}; 识别段: {len(segments)}")

        # Stage 7: 段级模式识别
        print(f"    [7/10] 段级模式识别...")
        mode_engine = ModeEvidenceEngine()
        mode_results = []
        for seg in segments:
            seg_features = [
                window_features[i]
                for i, wid in enumerate(window_ids)
                if wid in seg.window_ids
            ]
            result = mode_engine.classify_segment(seg, seg_features)
            mode_results.append(result)

        # Stage 8: 时序解码
        print(f"    [8/10] 时序解码...")
        decoded_results = temporal_decode(segments, mode_results)

        # 统计模式分布
        mode_counter = {}
        for mr in decoded_results:
            mode_counter[mr.mode] = mode_counter.get(mr.mode, 0) + 1
        dominant_mode = max(mode_counter, key=mode_counter.get) if mode_counter else "未知"
        print(f"    模式: {dominant_mode} ({mode_counter})")

        # Stage 9: 功能属性
        print(f"    [9/10] 功能属性识别...")
        attr_engine = FunctionAttributeEngine()
        attr_results = _compute_function_attributes(
            window_features, decoded_results, segments, window_ids, attr_engine,
        )
        global_attr, global_attr_conf = compute_global_attribute(attr_results)
        print(f"    属性: {global_attr}")

        # Stage 10: 块级输出
        print(f"    [10/10] 构建块级输出...")
        blocks = build_blocks(
            key, window_ids, segments, decoded_results,
            attr_results, window_features, block_duration,
        )

        # 守恒验证
        total_windows = len(windows)
        total_pulses = original_counts[key]
        conservation_report = verify_block_conservation(
            blocks, total_windows, total_pulses,
        )
        if not conservation_report["windows_match"]:
            print(f"    [WARN] 窗口数不守恒: {conservation_report}")
        if not conservation_report["pulses_match"]:
            print(f"    [WARN] 脉冲数不守恒: {conservation_report}")

        # 诊断
        diag = compute_diagnostics(
            window_features, segments, decoded_results, attr_results,
        )
        baselines = compare_with_degenerate_baselines(
            decoded_results, window_features,
        )

        # 汇总
        summary = {
            "radar_id": rid,
            "sample": sample,
            "total_pulses": total_pulses,
            "total_windows": total_windows,
            "n_empty_windows": quality["empty"],
            "n_valid_windows": quality["valid"],
            "n_segments": len(segments),
            "n_blocks": len(blocks),
            "dominant_mode": dominant_mode,
            "mode_distribution": mode_counter,
            "global_func_attr": global_attr,
            "global_func_attr_conf": global_attr_conf,
            "quality": quality,
            "diagnostics": diag,
            "baselines": baselines,
            "conservation": conservation_report,
        }

        # 保存
        save_radar_output(run_dir, key, blocks, summary, conservation_report)

        all_summaries[key] = summary
        all_blocks[key] = blocks
        if partial_callback is not None:
            partial_callback({
                "run_dir": run_dir,
                "radar_key": key,
                "summaries": dict(all_summaries),
            })

        # 打印关键指标
        print(f"    已知覆盖率: {diag.get('known_coverage', 0):.1%}")
        print(f"    未知率: {diag.get('unknown_ratio', 0):.1%}")
        print(f"    守恒: 窗口={conservation_report['windows_match']}, "
              f"脉冲={conservation_report['pulses_match']}")

    # ── 保存全局汇总 ──
    elapsed = time.time() - start_time
    total_pulses = sum(original_counts.values())
    total_windows = sum(len(w) for w in all_radars.values())

    manifest = RunManifest(
        run_id=Path(run_dir).name,
        timestamp=Path(run_dir).name,
        code_version=CODE_VERSION,
        python_version=platform.python_version(),
        input_files=input_files,
        input_hashes={f: compute_file_hash(f) for f in input_files},
        config_hash=compute_file_hash(config_path) if config_path else "",
        total_windows=total_windows,
        total_pulses=total_pulses,
        elapsed_seconds=round(elapsed, 2),
        parameters={
            "block_duration": block_duration,
            "change_method": change_method,
            "display_interval": display_interval,
            "llm_labels_path": llm_labels_path or "",
            "pre_segmented": pre_segmented,
        },
    )

    save_manifest(run_dir, manifest)
    if config_path and os.path.exists(config_path):
        save_config_snapshot(run_dir, config_path)
    display_index = save_global_time_display(
        run_dir, all_blocks, display_interval=display_interval,
    )
    save_global_summary(run_dir, all_summaries, manifest)
    save_analysis_report(
        run_dir, all_summaries, all_blocks, manifest, display_index,
        llm_labels_path=llm_labels_path,
    )
    save_segment_timeline_tables(
        run_dir, all_blocks, llm_labels_path=llm_labels_path,
    )
    save_llm_reference_tables(run_dir, llm_labels_path=llm_labels_path)

    print(f"\n  管线 {CODE_VERSION} 完成")
    print(f"  耗时: {elapsed:.1f}秒")
    print(f"  雷达: {len(all_summaries)}")
    print(f"  脉冲: {total_pulses:,}")
    print(f"  显示帧: {display_index.get('n_frames', 0)} "
          f"(间隔 {display_interval:.1f}s)")
    print(f"  输出: {run_dir}/")

    return all_summaries


def _compute_function_attributes(
    window_features: list[WindowFeatures],
    mode_results: list[ModeResult],
    segments: list[StateSegment],
    window_ids: np.ndarray,
    attr_engine: FunctionAttributeEngine,
) -> list[AttributeResult]:
    """
    计算功能属性时间线。

    每 5 个连续窗口 (1 秒) 判一次。
    尾部不足 5 窗口重新计算并降低质量。
    """
    n = len(window_features)
    if n == 0:
        return []

    # 构建窗口到段的映射 (用于获取模式结果)
    wid_to_mode = {}
    for seg, mr in zip(segments, mode_results):
        for wid in seg.window_ids:
            wid_to_mode[wid] = mr

    results: list[AttributeResult] = []
    windows_per_attr = 5

    for i in range(0, n, windows_per_attr):
        chunk_wf = window_features[i:i + windows_per_attr]
        chunk_wids = window_ids[i:i + windows_per_attr]

        # 获取对应的模式结果
        chunk_modes = []
        for wid in chunk_wids:
            mr = wid_to_mode.get(int(wid))
            if mr:
                chunk_modes.append(mr)

        # 判断是否为尾部
        if len(chunk_wf) < windows_per_attr:
            # 尾部: 重新计算，降低质量
            result = attr_engine.classify_tail_windows(chunk_wf, chunk_modes)
        else:
            result = attr_engine.classify_window(
                chunk_wf, chunk_modes, chunk_wids.tolist(),
            )

        results.append(result)

    return results


if __name__ == "__main__":
    import sys
    pre_seg = "--pre-segmented" in sys.argv
    input_files = [
        "F:/signals/sample1_template_match_pdw_with_pred_label.txt",
        "F:/signals/sample2_template_match_pdw_with_pred_label.txt",
    ]
    run_pipeline(input_files, pre_segmented=pre_seg)

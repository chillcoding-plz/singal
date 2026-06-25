"""
雷达信号识别系统 - 前端 API 接口

提供给 UI 前端调用的 Python 接口层。
所有接口返回 JSON 可序列化的字典，前端可直接使用。

使用方式:
  from app.api import RadarAPI
  api = RadarAPI()
  result = api.run(input_files=["data.txt"])
"""
from __future__ import annotations
import os
import json
import glob
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from radar_pipeline import run_pipeline
from radar_pipeline.schemas import (
    BlockOutput, ModeResult, AttributeResult,
    ModeTimelineEntry, AttributeTimelineEntry,
)


class RadarAPI:
    """雷达信号识别系统 API"""

    def __init__(self, output_dir: str = "artifacts/runs"):
        self.output_dir = output_dir

    # ── 核心接口 ──────────────────────────────────────────────────────

    def run(
        self,
        input_files: list[str],
        config_path: Optional[str] = None,
        block_duration: float = 5.0,
        change_method: str = "auto",
        display_interval: float = 30.0,
        llm_labels_path: Optional[str] = None,
        pre_segmented: bool = False,
        partial_callback=None,
    ) -> dict:
        """
        执行完整管线。

        Args:
            input_files: 输入 PDW 数据文件路径列表
            config_path: 配置文件路径 (可选)
            block_duration: 块时长 (秒), 默认 5.0
            change_method: 变化点检测方法 ("auto"/"custom"/"ruptures")
            display_interval: 跨雷达显示保存间隔 (秒), 默认 30.0
            llm_labels_path: 大模型伪标签路径 (可选, 用于准确率对比)
            pre_segmented: 输入已按 200ms 切分 (每文件一个窗口)

        Returns:
            {
                "status": "ok" | "error",
                "run_dir": "artifacts/runs/20260623_xxxxxx",
                "radars": {
                    "sample1/radar_1": {
                        "radar_id": "radar_1",
                        "sample": "sample1",
                        "total_pulses": 12345,
                        "total_windows": 500,
                        "dominant_mode": "搜索",
                        "mode_distribution": {"搜索": 45, "跟踪": 5},
                        "global_func_attr": "对空搜索",
                        "known_coverage": 0.85,
                        "unknown_ratio": 0.05,
                        "n_segments": 12,
                        "n_blocks": 20,
                        "conservation": {"windows_match": true, "pulses_match": true},
                    },
                    ...
                },
                "manifest": {...},
                "error": null
            }
        """
        try:
            summaries = run_pipeline(
                input_files=input_files,
                config_path=config_path,
                block_duration=block_duration,
                output_dir=output_dir if (output_dir := self.output_dir) else "artifacts/runs",
                change_method=change_method,
                display_interval=display_interval,
                llm_labels_path=llm_labels_path,
                pre_segmented=pre_segmented,
                partial_callback=partial_callback,
            )

            radars = {}
            for key, summary in summaries.items():
                diag = summary.get("diagnostics", {})
                conservation = summary.get("conservation", {})
                radars[key] = {
                    "radar_id": summary.get("radar_id", ""),
                    "sample": summary.get("sample", ""),
                    "total_pulses": summary.get("total_pulses", 0),
                    "total_windows": summary.get("total_windows", 0),
                    "n_empty_windows": summary.get("n_empty_windows", 0),
                    "n_valid_windows": summary.get("n_valid_windows", 0),
                    "dominant_mode": summary.get("dominant_mode", "未知"),
                    "mode_distribution": summary.get("mode_distribution", {}),
                    "global_func_attr": summary.get("global_func_attr", "未知"),
                    "global_func_attr_conf": summary.get("global_func_attr_conf", 0.0),
                    "known_coverage": diag.get("known_coverage", 0.0),
                    "unknown_ratio": diag.get("unknown_ratio", 0.0),
                    "n_segments": summary.get("n_segments", 0),
                    "n_blocks": summary.get("n_blocks", 0),
                    "conservation": conservation,
                    "quality": summary.get("quality", {}),
                }

            # 读取 manifest
            run_dir = self._find_latest_run()
            manifest = self._load_manifest(run_dir) if run_dir else {}

            return {
                "status": "ok",
                "run_dir": run_dir or "",
                "radars": radars,
                "manifest": manifest,
                "error": None,
            }

        except Exception as e:
            return {
                "status": "error",
                "run_dir": "",
                "radars": {},
                "manifest": {},
                "error": str(e),
            }

    # ── 查询接口 ──────────────────────────────────────────────────────

    def list_runs(self, limit: int = 20) -> list[dict]:
        """
        列出历史运行记录。

        Returns:
            [{"run_id": "20260623_120000", "timestamp": "...", "n_radars": 5, ...}, ...]
        """
        runs_dir = self.output_dir
        if not os.path.exists(runs_dir):
            return []

        entries = []
        for name in sorted(os.listdir(runs_dir), reverse=True)[:limit]:
            run_path = os.path.join(runs_dir, name)
            if not os.path.isdir(run_path):
                continue
            manifest = self._load_manifest(run_path)
            entries.append({
                "run_id": name,
                "run_dir": run_path,
                "timestamp": manifest.get("timestamp", name),
                "code_version": manifest.get("code_version", ""),
                "total_radars": len(manifest.get("parameters", {})),
                "total_pulses": manifest.get("total_pulses", 0),
                "total_windows": manifest.get("total_windows", 0),
                "elapsed_seconds": manifest.get("elapsed_seconds", 0),
            })
        return entries

    def get_run_summary(self, run_id: str) -> dict:
        """
        获取指定运行的全局汇总。

        Args:
            run_id: 运行 ID (如 "20260623_120000")

        Returns:
            全局汇总 JSON
        """
        run_dir = os.path.join(self.output_dir, run_id)
        summary_path = os.path.join(run_dir, "global_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": f"Run {run_id} not found"}

    def get_radar_blocks(self, run_id: str, sample: str, radar_id: str) -> list[dict]:
        """
        获取指定雷达的所有块级输出。

        Args:
            run_id: 运行 ID
            sample: 样本名 (如 "sample1")
            radar_id: 雷达 ID (如 "radar_1")

        Returns:
            [block_json, ...] 按时间排序
        """
        radar_dir = os.path.join(self.output_dir, run_id, "timelines", sample, radar_id)
        if not os.path.exists(radar_dir):
            return []

        blocks = []
        for name in sorted(os.listdir(radar_dir)):
            if name.startswith("block_") and name.endswith(".json"):
                path = os.path.join(radar_dir, name)
                with open(path, "r", encoding="utf-8") as f:
                    blocks.append(json.load(f))
        return blocks

    def get_radar_summary(self, run_id: str, sample: str, radar_id: str) -> dict:
        """
        获取指定雷达的汇总信息。

        Returns:
            雷达级汇总 JSON
        """
        path = os.path.join(
            self.output_dir, run_id, "timelines", sample, radar_id, "summary.json",
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": f"Radar {sample}/{radar_id} not found in run {run_id}"}

    def get_display_frames(self, run_id: str) -> dict:
        """
        获取跨雷达时间显示帧索引。

        Returns:
            {"display_interval": 30.0, "n_frames": 10, "frames": [...]}
        """
        path = os.path.join(self.output_dir, run_id, "display_frames", "index.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": f"Display frames not found for run {run_id}"}

    def get_display_frame(self, run_id: str, frame_index: int) -> dict:
        """
        获取指定时间帧的全局状态。

        Args:
            frame_index: 帧索引 (0-based)

        Returns:
            帧数据 JSON，包含所有雷达在该时间段的状态
        """
        path = os.path.join(
            self.output_dir, run_id, "display_frames",
            f"frame_{frame_index:04d}.json",
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"error": f"Frame {frame_index} not found"}

    def get_segment_tables(self, run_id: str, radar_key: Optional[str] = None) -> dict:
        """
        获取时间段识别结果表。

        Args:
            radar_key: 雷达标识 (如 "sample1/radar_1"), 为空则返回全部

        Returns:
            {"tables": [{"radar_key": "...", "segments": [...]}, ...]}
        """
        tables_dir = os.path.join(self.output_dir, run_id, "segment_tables")
        if not os.path.exists(tables_dir):
            return {"error": f"Segment tables not found for run {run_id}"}

        if radar_key:
            safe_name = radar_key.replace("/", "_")
            csv_path = os.path.join(tables_dir, f"{safe_name}.csv")
            if os.path.exists(csv_path):
                return {"tables": [self._parse_segment_csv(csv_path)]}
            return {"error": f"Table for {radar_key} not found"}

        # 返回全部
        tables = []
        for name in sorted(os.listdir(tables_dir)):
            if name.endswith(".csv") and name != "all_radars.csv":
                tables.append(self._parse_segment_csv(os.path.join(tables_dir, name)))
        return {"tables": tables}

    def get_report(self, run_id: str) -> str:
        """
        获取完整分析报告 (Markdown)。

        Returns:
            Markdown 格式的分析报告文本
        """
        path = os.path.join(self.output_dir, run_id, "report.md")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return f"Report not found for run {run_id}"

    # ── 内部方法 ──────────────────────────────────────────────────────

    def _find_latest_run(self) -> Optional[str]:
        """查找最新的运行目录"""
        if not os.path.exists(self.output_dir):
            return None
        entries = sorted(
            [e for e in os.listdir(self.output_dir)
             if os.path.isdir(os.path.join(self.output_dir, e))],
            reverse=True,
        )
        return os.path.join(self.output_dir, entries[0]) if entries else None

    def _load_manifest(self, run_dir: Optional[str]) -> dict:
        """加载运行清单"""
        if not run_dir:
            return {}
        path = os.path.join(run_dir, "manifest.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _parse_segment_csv(self, csv_path: str) -> dict:
        """解析时间段 CSV 为结构化数据"""
        import csv
        segments = []
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                segments.append({
                    "index": int(row.get("index", 0)),
                    "time": row.get("time", ""),
                    "start_time": float(row.get("start_time", 0)),
                    "end_time": float(row.get("end_time", 0)),
                    "n_pulses": int(row.get("n_pulses", 0)),
                    "mode": row.get("mode", ""),
                    "attribute": row.get("attribute", ""),
                    "accuracy": float(row.get("accuracy", 0)),
                    "source": row.get("source", ""),
                    "radar_key": row.get("radar_key", ""),
                })
        return {
            "radar_key": segments[0]["radar_key"] if segments else "",
            "segments": segments,
        }

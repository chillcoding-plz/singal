from datetime import datetime
import os
from typing import Iterable

import pandas as pd


def export_sorting_csv(df: pd.DataFrame, path: str):
    preferred = [
        "TOA",
        "RF",
        "PW",
        "PA",
        "DOA",
        "Track_ID",
        "Display_Track_ID",
        "Original_Track_ID",
        "True_Track_ID",
        "Original_Label",
        "True_Label",
        "Assigned",
        "Sorting_Method",
        "HDBSCAN_Track_ID",
        "HDBSCAN_Input_Track_ID",
        "HDBSCAN_Assigned",
        "HDBSCAN_Sorting_Method",
        "CyclePeriod_Track_ID",
        "CyclePeriod_OurPredID",
        "CyclePeriod_Assigned",
        "CyclePeriod_Sorting_Method",
    ]
    columns = [col for col in preferred if col in df.columns]
    df[columns].to_csv(path, index=False, encoding="utf-8-sig")


def export_recognition_csv(track_results: pd.DataFrame, path: str):
    columns = [
        "Track_ID",
        "Pulse_Count",
        "Predicted_Label",
        "True_Label",
        "Confidence",
        "Mean_RF",
        "Mean_PW",
        "Mean_PRI",
    ]
    track_results[[col for col in columns if col in track_results.columns]].to_csv(path, index=False, encoding="utf-8-sig")


def export_full_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def export_log(lines: Iterable[str], path: str):
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def export_report(df: pd.DataFrame, recognition_results: pd.DataFrame, log_lines: Iterable[str], path: str):
    track_count = int(df["Track_ID"].nunique()) if "Track_ID" in df.columns else 0
    assigned = int(df["Assigned"].sum()) if "Assigned" in df.columns else 0
    mean_conf = float(recognition_results["Confidence"].mean()) if recognition_results is not None and not recognition_results.empty else 0.0
    ext = os.path.splitext(path)[1].lower()

    if ext == ".html":
        content = f"""
<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>信号分选与识别报告</title></head>
<body>
<h1>信号分选与识别系统报告</h1>
<p>生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}</p>
<ul>
<li>脉冲总数：{len(df):,}</li>
<li>分选轨迹数：{track_count}</li>
<li>已分配脉冲数：{assigned:,}</li>
<li>平均置信度：{mean_conf * 100:.1f}%</li>
</ul>
<h2>识别结果</h2>
{recognition_results.to_html(index=False) if recognition_results is not None else ""}
<h2>运行日志</h2>
<pre>{chr(10).join(log_lines)}</pre>
</body></html>
"""
    else:
        content = (
            "信号分选与识别系统报告\n"
            "==============================\n"
            f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"脉冲总数：{len(df):,}\n"
            f"分选轨迹数：{track_count}\n"
            f"已分配脉冲数：{assigned:,}\n"
            f"平均置信度：{mean_conf * 100:.1f}%\n\n"
            "运行日志：\n"
            + "\n".join(log_lines)
        )
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

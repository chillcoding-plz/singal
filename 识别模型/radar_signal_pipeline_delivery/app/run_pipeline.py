"""
新管线命令行入口

使用方式:
  python -m app.run_pipeline
  python -m app.run_pipeline --input file1.txt file2.txt
"""
import argparse
import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radar_pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="雷达信号识别管线 v3.0",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (YAML)",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[
            "sample1_template_match_pdw_with_pred_label.txt",
            "sample2_template_match_pdw_with_pred_label.txt",
        ],
        help="输入文件路径，默认使用 sample1/sample2 两个数据集",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/runs",
        help="输出目录",
    )
    parser.add_argument(
        "--block-duration",
        type=float,
        default=30.0,
        help="块时长 (秒)",
    )
    parser.add_argument(
        "--change-method",
        type=str,
        default="auto",
        choices=["auto", "custom", "ruptures"],
        help="变化点检测方法",
    )
    parser.add_argument(
        "--display-interval",
        type=float,
        default=30.0,
        help="跨雷达显示保存间隔 (秒)",
    )
    parser.add_argument(
        "--llm-labels",
        type=str,
        default="artifacts/llm_eval/hybrid_v1/hybrid_llm_labels.jsonl",
        help="大模型伪标签 JSONL，用于计算规则输出对比准确率",
    )
    parser.add_argument(
        "--pre-segmented",
        action="store_true",
        default=False,
        help="输入已按 200ms 切分 (每文件一个窗口)",
    )

    args = parser.parse_args()

    results = run_pipeline(
        input_files=args.input,
        config_path=args.config,
        block_duration=args.block_duration,
        output_dir=args.output,
        change_method=args.change_method,
        display_interval=args.display_interval,
        llm_labels_path=args.llm_labels,
        pre_segmented=args.pre_segmented,
    )

    # 打印主导概览。注意：这里不是逐时间段输出，详细结果见 segment_tables/all_radars.md。
    print("\n" + "=" * 60)
    print("  全局主导概览")
    print("=" * 60)
    print("  说明: 这里显示每个雷达的主导模式/主导属性；逐时间段结果请查看 segment_tables/all_radars.md")
    for key, summary in results.items():
        mode = summary.get("dominant_mode", "未知")
        attr = summary.get("global_func_attr", "未知")
        known = summary.get("diagnostics", {}).get("known_coverage", 0)
        unknown = summary.get("diagnostics", {}).get("unknown_ratio", 0)
        conservation = summary.get("conservation", {})
        ok = "OK" if conservation.get("windows_match") and conservation.get("pulses_match") else "FAIL"
        print(f"  {key}: 主导模式={mode}, 主导属性={attr}, "
              f"模式已知覆盖率={known:.1%}, 模式未知率={unknown:.1%}, 守恒={ok}")


if __name__ == "__main__":
    main()

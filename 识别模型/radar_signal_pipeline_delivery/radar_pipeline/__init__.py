"""
雷达信号识别管线 v3.0

独立于现有管线的新实现，用于验证重构方案的效果。

主要改进:
  - TOA 全链路 float64
  - 保留空窗口和窗口编号
  - 先构建状态段，后识别模式
  - 多证据评分 + 三级输出 (known/suspected/unknown)
  - 功能属性独立化 (85% 信号 + 15% 模式)
  - 块级守恒验证
  - 各项指标独立报告
"""
from .pipeline import run_pipeline

__version__ = "3.0.0"
__all__ = ["run_pipeline"]

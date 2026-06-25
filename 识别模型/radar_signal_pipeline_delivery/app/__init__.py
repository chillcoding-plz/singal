"""
雷达信号识别系统 - 应用层

前端集成入口:
  from app.api import RadarAPI
  api = RadarAPI()
  result = api.run(input_files=["data.txt"])
"""
from .api import RadarAPI

__all__ = ["RadarAPI"]

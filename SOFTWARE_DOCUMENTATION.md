# 信号分选与识别系统软件说明

## 1. 软件概述

`信号分选与识别系统` 是一个基于 `Python + PyQt5 + matplotlib` 的桌面端工程软件，用于导入 PDW 脉冲数据、进行信号分选、信号识别、结果可视化和结果导出。

软件保持蓝白色 Windows 工程软件风格，界面包含顶部标题栏、导航页签、Ribbon 工具栏、中间图表卡片、右侧参数面板、底部运行日志和状态栏。

## 2. 运行环境

- Python 3.8+
- PyQt5
- matplotlib
- numpy
- pandas
- scikit-learn

安装依赖：

```powershell
cd D:\apengpeng
pip install -r requirements.txt
```

运行程序：

```powershell
python main.py
```

## 3. 项目结构

```text
D:\apengpeng
├── main.py                         程序入口
├── requirements.txt                依赖列表
├── resources/
│   └── style.qss                   界面样式
├── ui/
│   ├── main_window.py              主窗口、页面、线程、交互逻辑
│   ├── widgets.py                  Ribbon、轨迹面板、参数面板
│   └── charts.py                   图表卡片、表格、绘图函数
├── core/
│   ├── data_loader.py              数据导入与字段兼容
│   ├── sorting_algorithms.py       分选算法
│   ├── feature_extractor.py        轨迹特征提取
│   ├── recognition_algorithms.py   信号识别算法
│   └── exporter.py                 结果、日志、报告导出
└── utils/
    └── helpers.py                  通用工具函数
```

## 4. 数据导入

软件支持导入 `csv` 和 `txt` 文件。

推荐字段：

```text
TOA, RF, PW, PA, DOA, LABEL, SigIdx
```

兼容字段示例：

```text
toa, toa_us, time
rf, rf_mhz, frequency
pw, pw_us, pulse_width
pa, pa_db, amplitude
label, class_label
sigidx, sig_idx, signal_id
```

说明：

- `TOA` 和 `RF` 是必要字段。
- 如果没有 `PW`、`PA`、`DOA`，软件会自动补默认值。
- 如果没有 `PRI`，软件会根据 `TOA` 差分计算。
- 导入后不会自动把 `SigIdx` 当作当前分选结果。
- 只有点击 `开始分析` 后，软件才会生成当前 `Track_ID`。

导入成功后，日志会显示：

- 文件名
- 脉冲数量
- 数据字段

## 5. 主页

主页用于查看导入后的原始数据。

当前包含：

- `TOA 波形 / 脉冲时间轴`
- `原始 PRI vs TOA`

导入数据后，主页不会显示分选轨迹。只有完成分选分析后，分选页面才会显示多颜色轨迹。

## 6. 分选分析

支持分选方法：

- `MHT`
- `DBSCAN`
- `K-means`

点击 `开始分析` 后，软件会根据当前选择的方法生成：

- `Track_ID`
- `Assigned`

分选分析页面显示：

- `PRI vs TOA` 散点图
- `RF vs TOA` 散点图
- 轨迹脉冲统计柱状图
- 已分配 / 未分配比例图

左侧轨迹列表会在分选完成后动态生成。用户可以通过复选框控制某条轨迹是否显示。

右侧分析摘要会动态显示：

- 分选轨迹数
- 已分配脉冲数
- 未分配脉冲数
- 平均 PRI
- 平均脉宽

## 7. 信号识别

支持识别模型：

- `SVM`
- `KNN`
- `随机森林`
- `CNN`

说明：

- `SVM`、`KNN`、`随机森林` 当前为传统机器学习版本。
- `CNN` 当前为预留接口，暂未接入真实模型。

识别前需要先完成分选分析。

每条轨迹会提取统计特征：

- 平均 RF
- RF 方差
- 平均 PW
- PW 方差
- 平均 PRI
- PRI 方差
- 脉冲数量

识别输出：

- `Predicted_Label`
- `Confidence`

信号识别页面显示：

- 特征二维投影散点图
- 分类概率条形图
- 识别结果表格
- 类别统计柱状图

右侧识别摘要会动态显示：

- 识别轨迹数
- 平均置信度
- 类别数

## 8. 结果导出

结果导出页面支持：

- 分选结果 CSV
- 识别结果 CSV
- 完整结果 CSV
- 当前图表 PNG
- 运行日志 TXT
- 总结报告 TXT / HTML

分选结果至少包括：

- TOA
- RF
- PW
- PA
- DOA
- Track_ID
- Assigned

识别结果至少包括：

- Track_ID
- Pulse_Count
- Predicted_Label
- Confidence
- Mean_RF
- Mean_PW
- Mean_PRI

## 9. 运行状态与日志

底部区域包含：

- 运行日志
- 当前操作
- 进度条
- 本次耗时

分选和识别过程使用后台线程执行。

运行时：

- 开始按钮会禁用
- 导入按钮会禁用
- 停止按钮会启用
- 任务结束后按钮恢复

停止按钮会发送取消请求。由于部分 scikit-learn 算法本身不支持强制中断，因此停止通常在任务阶段边界生效。

## 10. 基本使用流程

1. 点击 `导入数据`
2. 选择 `csv` 或 `txt` 文件
3. 在主页查看原始 TOA 和 PRI 图
4. 进入 `分选分析`
5. 选择 `MHT / DBSCAN / K-means`
6. 点击 `开始分析`
7. 查看分选结果和轨迹列表
8. 进入 `信号识别`
9. 选择 `SVM / KNN / 随机森林`
10. 点击 `开始识别`
11. 查看识别表格和类别统计
12. 进入 `结果导出`
13. 导出结果文件、日志或报告

## 11. 当前限制

- `MHT` 为简化占位实现。
- `CNN` 为预留接口，暂未接入真实模型。
- 当前识别算法为传统机器学习 / 规则辅助版本，适合第一版完整流程验证。
- 停止按钮不能强制中断所有 scikit-learn 内部计算，只能在任务阶段边界生效。

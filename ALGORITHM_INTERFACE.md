# 算法对接接口说明

界面与算法之间的入口在 `algorithm_adapter.py`。

## 1. 输入数据

导入 CSV 后，界面会把数据读成 `numpy` 结构化数组 `pdw_data`，传给算法：

```python
AlgorithmInput(
    pdw_data=pdw_data,
    method="MHT",       # 或 DBSCAN / K-means / SVM / CNN 等
    params={}
)
```

推荐原始数据字段：

```text
pulse_id, toa_us, pri_us, rf_mhz, pw_us, pa_db
```

其中 `toa_us`、`pri_us`、`rf_mhz` 为必需字段。`track_id`、`class_label`、`confidence` 可以由算法返回后写入。

## 2. 分选算法返回

真实分选算法需要返回 `SortingResult` 或同结构 `dict`：

```python
from algorithm_adapter import SortingResult

def run_sorting(input_data):
    pdw = input_data.pdw_data
    method = input_data.method

    track_ids = ...
    confidence = ...

    return SortingResult(
        track_ids=track_ids,       # 长度必须等于输入脉冲数
        confidence=confidence,     # 可选
        summary={"method": method}
    )
```

也可以返回：

```python
{
    "track_ids": track_ids,
    "confidence": confidence,
    "summary": {}
}
```

界面会把 `track_ids` 写回数据字段 `track_id`，把 `confidence` 写回 `sorting_confidence`。

## 3. 识别模型返回

真实识别模型需要返回 `RecognitionResult` 或同结构 `dict`：

```python
from algorithm_adapter import RecognitionResult

def run_recognition(input_data, sorting_result=None):
    pdw = input_data.pdw_data
    model = input_data.method

    class_labels = ...
    confidence = ...

    return RecognitionResult(
        class_labels=class_labels,   # 长度必须等于输入脉冲数
        confidence=confidence,       # 0-1 浮点数
        probabilities=None,
        summary={"model": model}
    )
```

也可以返回：

```python
{
    "class_labels": class_labels,
    "confidence": confidence,
    "probabilities": probabilities,
    "summary": {}
}
```

界面会把 `class_labels` 写回 `class_label`，把 `confidence` 写回 `confidence`。

## 4. 注册真实算法

在创建主窗口后注册：

```python
window.algorithm_bridge.register_sorting_runner(run_sorting)
window.algorithm_bridge.register_recognition_runner(run_recognition)
```

当前默认实现只是占位：如果导入数据本身已有 `track_id`、`class_label`、`confidence`，会直接复用这些字段；否则会生成简单占位结果，方便 UI 流程先跑通。

# 分选模型可运行包

入口文件：

```bash
python Best_tracklet_graph_front.py --root . --sample all
```

也可以直接运行：

```bash
run_sort_all.bat
```

包内只保留 `Best_tracklet_graph_front.py` 正常运行需要的源码和数据：

- `edata/Test_Data`
- `edata/Train_Data`
- 分选主函数 `Best_tracklet_graph_front.py`
- 直接依赖模块 `Best.py`、`tracklet_graph_sort.py`
- 默认/可选排序后端 `pa_tsr_hdbscan_sort.py`、`pa_tgr_hdbscan_sort.py`
- 后端依赖 `dbscan_sort.py`、`hdbscan_sort.py`、`hdbscan_conservative_merge_sort.py`、`tracklet_pri_sort.py`
- `识别/` 中仅保留分选评估和 SigIdx 规约会调用的辅助文件

输出目录默认是：

```text
outputs_best_front_tracklet_graph
```

依赖：

```bash
pip install -r requirements.txt
```

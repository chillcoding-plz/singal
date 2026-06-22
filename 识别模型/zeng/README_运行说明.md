# 识别模型可运行包

入口文件：

```bash
python template_match_recognition.py --sample all
```

也可以直接运行：

```bash
run_recognition_all.bat
```

包内已包含默认运行需要的文件：

- `edata/Test_Data`
- `outputs_best_front_tracklet_graph/sample1/sample1_sort.txt`
- `outputs_best_front_tracklet_graph/sample2/sample2_sort.txt`
- `outputs_expanded_template_library/template_library.json`
- `outputs_expanded_template_library/tuned_match_parameters.json`

输出目录默认是：

```text
outputs_template_match_recognition
```

依赖：

```bash
pip install -r requirements.txt
```

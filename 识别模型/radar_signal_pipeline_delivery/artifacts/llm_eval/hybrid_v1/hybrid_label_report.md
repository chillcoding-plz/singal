# Hybrid LLM Pseudo Labels

This label set uses individualized radar PDW as the primary evidence and raw Train_Data/Class profiles only as auxiliary confidence calibration.
Raw profiles never overwrite the individualized radar timeline labels.

- labels: 144
- primary_weight: 0.85
- raw_auxiliary_weight: 0.15
- raw_auxiliary_role: confidence_only_no_label_override

| radar | state distribution |
|---|---|
| sample1/radar_1 | 制导/火控: 15, 搜索/对海搜索: 1 |
| sample1/radar_2 | 搜索/待定: 1, 未知/未知: 15 |
| sample1/radar_4 | 搜索/待定: 14, 未知/未知: 2 |
| sample1/radar_99 | 搜索/对空搜索: 16 |
| sample2/radar_1 | 搜索/待定: 16 |
| sample2/radar_2 | 搜索/待定: 14, 跟踪/火控: 2 |
| sample2/radar_3 | 搜索/待定: 16 |
| sample2/radar_4 | 搜索/对海搜索: 16 |
| sample2/radar_99 | 搜索/对空搜索: 16 |
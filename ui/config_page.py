import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class ParamSpec:
    group: str
    key: str
    label: str
    recommended: str = "-"


@dataclass(frozen=True)
class ConfigSection:
    key: str
    title: str
    description: str
    current_path: Optional[Path]
    default_path: Optional[Path]
    params: tuple[ParamSpec, ...]


PARAMS = {
    "pre_sort": (
        ParamSpec("streaming_sort_params", "max_track_gap_beats", "最大轨迹间隔 beat 数", "2 ~ 10"),
        ParamSpec("streaming_sort_params", "link_threshold", "轨迹关联阈值", "1.2 ~ 3.5"),
        ParamSpec("streaming_sort_params", "archive_link_threshold", "归档轨迹关联阈值", "1.0 ~ 2.8"),
        ParamSpec("streaming_sort_params", "fragment_link_threshold", "碎片关联阈值", "1.5 ~ 4.0"),
        ParamSpec("streaming_sort_params", "fragment_pair_threshold", "碎片配对阈值", "0.5 ~ 2.5"),
        ParamSpec("streaming_sort_params", "prototype_add_threshold", "原型新增阈值", "0.5 ~ 2.5"),
        ParamSpec("streaming_sort_params", "update_alpha", "更新系数 alpha", "0.05 ~ 0.50"),
    ),
    "main_sort": (
        ParamSpec("main5_params", "cycle_T_min", "周期搜索最小值", "1e-6 ~ 5e-6"),
        ParamSpec("main5_params", "cycle_T_max", "周期搜索最大值", "1e-3 ~ 5e-3"),
        ParamSpec("main5_params", "min_pulses_for_cycle", "周期判定最小脉冲数", "8 ~ 20"),
        ParamSpec("main5_params", "chain_toa_tol", "TOA 链容差", "1e-6 ~ 3e-6"),
        ParamSpec("main5_params", "merge_T_rel_tol", "周期合并相对容差", "0.003 ~ 0.015"),
        ParamSpec("main5_params", "param_absorb_min_score", "参数吸收最小得分", "4.8 ~ 5.6"),
    ),
    "fine_sort": (
        ParamSpec("core_mht_params", "feature_gate", "特征门限", "1.2 ~ 2.2"),
        ParamSpec("core_mht_params", "prefilter_p1", "预筛 Param1 门限", "60 ~ 140"),
        ParamSpec("core_mht_params", "prefilter_p2", "预筛 Param2 门限", "0.20 ~ 0.45"),
        ParamSpec("core_mht_params", "prefilter_p4", "预筛 Param4 门限", "6 ~ 15"),
        ParamSpec("core_mht_params", "prefilter_p5_deg", "预筛 Param5 角度门限", "2 ~ 5"),
        ParamSpec("core_mht_params", "rhythm_gate_us", "节奏门限 us", "2.5 ~ 5"),
        ParamSpec("core_mht_params", "rhythm_gate_sigma", "节奏 sigma 门限", "1.0 ~ 2.0"),
        ParamSpec("core_mht_params", "max_beat_gap", "最大 beat 间隔", "2 ~ 6"),
    ),
    "recognition": (
        ParamSpec("template_match_params", "min_batch_pulses", "最小批次脉冲数", "5 ~ 50"),
        ParamSpec("template_match_params", "pri_gap_multiplier", "PRI 间隔倍数", "2.0 ~ 10.0"),
        ParamSpec("template_match_params", "pri_gap_quantile", "PRI 间隔分位数", "0.70 ~ 0.98"),
        ParamSpec("template_match_params", "threshold_scale", "匹配阈值缩放", "0.20 ~ 1.20"),
        ParamSpec("template_match_params", "label_threshold_scales", "类别阈值缩放", "0.2 ~ 1.2"),
        ParamSpec("template_match_params", "class_threshold_floor_scale", "类别阈值下限缩放", "0.1 ~ 1.0"),
        ParamSpec("template_match_params", "matching_mode", "匹配模式", "nearest / nearest_with_rescue / class_min_ratio"),
    ),
    "radar_attribute": (
        ParamSpec("时间窗口参数", "结果汇总时长秒", "结果汇总时长秒", "10 ~ 60"),
        ParamSpec("时间窗口参数", "显示保存间隔秒", "显示保存间隔秒", "10 ~ 60"),
        ParamSpec("数据过滤参数", "窗口特征最小脉冲数", "窗口特征最小脉冲数", "3 ~ 20"),
        ParamSpec("变化点检测参数", "变化点检测方法", "变化点检测方法", "自动 / custom / ruptures"),
        ParamSpec("变化点检测参数", "变化点最小间隔窗口数", "变化点最小间隔窗口数", "3 ~ 8"),
        ParamSpec("变化点检测参数", "变化强度阈值", "变化强度阈值", "2.0 ~ 3.5"),
        ParamSpec("工作模式识别参数", "已知模式最低得分", "已知模式最低得分", "0.40 ~ 0.60"),
        ParamSpec("功能属性识别参数", "属性最低融合得分", "属性最低融合得分", "0.30 ~ 0.50"),
    ),
}

GROUP_TITLES = {
    "streaming_sort_params": "预分选参数",
    "main5_params": "主分选参数",
    "core_mht_params": "细分选参数",
    "template_match_params": "信号识别参数",
}


class ConfigPage(QWidget):
    log_message = pyqtSignal(str)

    def __init__(self, project_root: Path, parent=None):
        super().__init__(parent)
        self.project_root = Path(project_root)
        self.sections = self._build_sections()
        self.current_section: Optional[ConfigSection] = None
        self.inputs: dict[tuple[str, str], QLineEdit] = {}
        self.original_config: dict[str, Any] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 8)
        outer.setSpacing(8)

        header = QFrame()
        header.setProperty("class", "ribbon")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 10, 12, 10)
        header_layout.setSpacing(10)
        title = QLabel("参数设置")
        title.setProperty("class", "sectionTitle")
        subtitle = QLabel("仅展示可修改参数；请在文本框中填写参数值")
        subtitle.setProperty("class", "subtle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        header_layout.addStretch(1)
        outer.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        left = QFrame()
        left.setProperty("class", "card")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)
        left_title = QLabel("流程配置")
        left_title.setProperty("class", "sectionTitle")
        left_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.section_list = QListWidget()
        list_font = QFont()
        list_font.setPointSize(13)
        self.section_list.setFont(list_font)
        self.section_list.setStyleSheet("QListWidget::item { padding: 8px 6px; }")
        self.section_list.currentRowChanged.connect(self._select_section)
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.section_list, 1)
        splitter.addWidget(left)

        right = QFrame()
        right.setProperty("class", "card")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)

        self.section_title = QLabel("-")
        self.section_title.setProperty("class", "sectionTitle")
        self.section_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        self.section_desc = QLabel("-")
        self.section_desc.setProperty("class", "subtle")
        self.section_desc.setWordWrap(True)
        self.section_desc.setVisible(False)
        self.path_label = QLabel("-")
        self.path_label.setProperty("class", "subtle")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.form_host = QWidget()
        self.form_layout = QVBoxLayout(self.form_host)
        self.form_layout.setContentsMargins(0, 0, 0, 0)
        self.form_layout.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form_host)

        button_row = QHBoxLayout()
        self.save_btn = QPushButton("保存")
        self.reset_btn = QPushButton("重置默认")
        self.open_btn = QPushButton("打开文件位置")
        self.save_btn.clicked.connect(self._save_current)
        self.reset_btn.clicked.connect(self._reset_current)
        self.open_btn.clicked.connect(self._open_current_location)
        for button in [self.save_btn, self.reset_btn, self.open_btn]:
            button_row.addWidget(button)
        button_row.addStretch(1)

        right_layout.addWidget(self.section_title)
        right_layout.addWidget(self.path_label)
        right_layout.addWidget(scroll, 1)
        right_layout.addLayout(button_row)
        splitter.addWidget(right)
        splitter.setSizes([260, 980])

        self._populate_sections()
        if self.sections:
            self.section_list.setCurrentRow(0)

    def _build_sections(self):
        config_dir = self.project_root / "configs"
        default_dir = config_dir / "defaults"
        return [
            ConfigSection("pre_sort", "预分选", "HDBSCAN/streaming 预分选可调参数。", config_dir / "pre_sort_config.json", default_dir / "pre_sort_config.json", PARAMS["pre_sort"]),
            ConfigSection("main_sort", "主分选", "Cycle-period main5 主分选可调参数。", config_dir / "main_sort_main5_config.json", default_dir / "main_sort_main5_config.json", PARAMS["main_sort"]),
            ConfigSection("fine_sort", "细分选", "Tracklet-level MHT 细分选可调参数。", config_dir / "fine_sort_mht_config.json", default_dir / "fine_sort_mht_config.json", PARAMS["fine_sort"]),
            ConfigSection("recognition", "信号识别", "模板匹配信号识别可调参数。", config_dir / "signal_recognition_config.json", default_dir / "signal_recognition_config.json", PARAMS["recognition"]),
            ConfigSection("radar_attribute", "工作模式/功能属性分析", "雷达工作模式与功能属性分析可调参数。", config_dir / "radar_attribute_config.json", default_dir / "radar_attribute_config.json", PARAMS["radar_attribute"]),
        ]

    def _populate_sections(self):
        self.section_list.clear()
        for section in self.sections:
            item = QListWidgetItem(section.title)
            item.setData(Qt.UserRole, section.key)
            if not self._is_config_available(section):
                item.setText(f"{section.title}（预留）")
            self.section_list.addItem(item)

    def _select_section(self, row: int):
        if row < 0 or row >= len(self.sections):
            return
        section = self.sections[row]
        self.current_section = section
        self.section_title.setText(section.title)
        self._load_section(section)

    def _load_section(self, section: ConfigSection):
        self._clear_form()
        if not self._is_config_available(section):
            self.original_config = {}
            self.path_label.setText("配置文件：预留")
            self.save_btn.setEnabled(False)
            self.reset_btn.setEnabled(False)
            self.open_btn.setEnabled(False)
            return

        self._ensure_current_config(section)
        self.original_config = self._read_json(section.current_path)
        self.path_label.setText(f"配置文件：{section.current_path}")
        self.save_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self._build_form(section)

    def _build_form(self, section: ConfigSection):
        grouped: dict[str, list[ParamSpec]] = {}
        for spec in section.params:
            grouped.setdefault(spec.group, []).append(spec)

        for group, specs in grouped.items():
            group_frame = QFrame()
            group_frame.setProperty("class", "card")
            grid = QGridLayout(group_frame)
            grid.setContentsMargins(12, 10, 12, 12)
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(8)

            group_title = QLabel(GROUP_TITLES.get(group, group))
            group_title.setProperty("class", "sectionTitle")
            group_title.setStyleSheet("font-size: 16px; font-weight: 700;")
            grid.addWidget(group_title, 0, 0, 1, 3)
            for column, text in enumerate(("参数", "当前值", "推荐范围")):
                header = QLabel(text)
                header.setStyleSheet("font-size: 14px; font-weight: 700;")
                grid.addWidget(header, 1, column)

            for row, spec in enumerate(specs, start=2):
                value = self._get_value(self.original_config, spec.group, spec.key)
                edit = QLineEdit(self._format_value(value))
                edit.setMinimumWidth(260)
                edit.setStyleSheet("font-size: 14px; padding: 4px 6px;")
                self.inputs[(spec.group, spec.key)] = edit
                range_label = QLabel(self._recommended_text(self.original_config, spec))
                range_label.setProperty("class", "subtle")
                range_label.setStyleSheet("font-size: 14px;")
                range_label.setWordWrap(True)
                label = QLabel(spec.label)
                label.setStyleSheet("font-size: 14px;")
                grid.addWidget(label, row, 0)
                grid.addWidget(edit, row, 1)
                grid.addWidget(range_label, row, 2)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(2, 1)
            self.form_layout.addWidget(group_frame)
        self.form_layout.addStretch(1)

    def _save_current(self):
        section = self.current_section
        if section is None or section.current_path is None:
            return
        config = dict(self.original_config)
        try:
            for spec in section.params:
                old_value = self._get_value(config, spec.group, spec.key)
                raw = self.inputs[(spec.group, spec.key)].text().strip()
                self._set_value(config, spec.group, spec.key, self._parse_value(raw, old_value))
        except ValueError as exc:
            QMessageBox.warning(self, "参数格式错误", str(exc))
            return

        section.current_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.original_config = config
        self.log_message.emit(f"参数配置已保存：{section.title}")

    def _reset_current(self):
        section = self.current_section
        if section is None or section.current_path is None or section.default_path is None:
            return
        reply = QMessageBox.question(
            self,
            "重置默认配置",
            f"确认将“{section.title}”恢复为默认配置？当前修改会被覆盖。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._ensure_current_config(section, overwrite=True)
        self._load_section(section)
        self.log_message.emit(f"参数配置已重置为默认值：{section.title}")

    def _open_current_location(self):
        section = self.current_section
        if section is None or section.current_path is None:
            return
        os.startfile(section.current_path.parent)

    def _clear_form(self):
        self.inputs = {}
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _read_json(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def _is_config_available(self, section: ConfigSection) -> bool:
        return section.current_path is not None and section.default_path is not None and section.default_path.exists()

    def _ensure_current_config(self, section: ConfigSection, overwrite: bool = False):
        if section.current_path is None or section.default_path is None:
            return
        if overwrite or not section.current_path.exists():
            section.current_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(section.default_path, section.current_path)

    def _get_value(self, config: dict, group: str, key: str):
        group_data = config.get(group, {})
        if isinstance(group_data, dict):
            return group_data.get(key, "")
        return ""

    def _set_value(self, config: dict, group: str, key: str, value):
        if not isinstance(config.get(group), dict):
            config[group] = {}
        config[group][key] = value

    def _format_value(self, value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return "" if value is None else str(value)

    def _recommended_text(self, config: dict, spec: ParamSpec) -> str:
        recommend_groups = {
            "streaming_sort_params": "recommend_sort_params",
            "template_match_params": "recommend_match_params",
            "main5_params": "recommended_ranges",
            "core_mht_params": "recommended_ranges",
        }
        group = recommend_groups.get(spec.group)
        if group:
            value = config.get(group, {}).get(spec.key) if isinstance(config.get(group), dict) else None
            if isinstance(value, list):
                return " / ".join(str(item) for item in value)
            if value not in (None, ""):
                return str(value)
        return spec.recommended

    def _parse_value(self, text: str, old_value):
        if isinstance(old_value, bool):
            value = text.lower()
            if value in {"true", "1", "yes", "y", "是"}:
                return True
            if value in {"false", "0", "no", "n", "否"}:
                return False
            raise ValueError(f"布尔参数只能填写 true/false：{text}")
        if isinstance(old_value, int) and not isinstance(old_value, bool):
            try:
                return int(text)
            except ValueError as exc:
                raise ValueError(f"整数参数格式错误：{text}") from exc
        if isinstance(old_value, float):
            try:
                return float(text)
            except ValueError as exc:
                raise ValueError(f"小数参数格式错误：{text}") from exc
        return text

import colorsys
from typing import Dict, Iterable

from PyQt5.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygon
from PyQt5.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.pipelines import (
    RECOGNITION_MODEL,
    SORTING_PIPELINE_ID,
    pipeline_definition,
)


TRACK_COLORS = ["#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#78909C", "#D81B60", "#00ACC1", "#7CB342"]


def track_color(track_id: int) -> str:
    track_id = int(track_id)
    if track_id <= 0:
        return "#78909C"
    if track_id <= len(TRACK_COLORS):
        return TRACK_COLORS[track_id - 1]
    value = (track_id * 2654435761) & 0xFFFFFFFF
    hue = (value % 360) / 360.0
    saturation = 0.58 + (((value >> 9) % 7) * 0.05)
    brightness = 0.68 + (((value >> 17) % 5) * 0.04)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return f"#{int(red * 255):02X}{int(green * 255):02X}{int(blue * 255):02X}"


class ColorDot(QWidget):
    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.color = QColor(color)
        self.setFixedSize(14, 14)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 10, 10)


def _ribbon_icon(kind: str) -> QIcon:
    pixmap = QPixmap(34, 34)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    blue = QColor("#0052b5")
    green = QColor("#18a957")
    gray = QColor("#5f6b76")
    dark = QColor("#111827")

    if kind == "import":
        painter.setPen(QPen(dark, 2.0))
        painter.setBrush(QColor("#ffffff"))
        painter.drawPath(_folder_path())
    elif kind == "truth":
        painter.setPen(QPen(blue, 2.0))
        painter.setBrush(QColor("#f7fbff"))
        painter.drawRoundedRect(8, 6, 18, 22, 2, 2)
        painter.drawLine(12, 14, 22, 14)
        painter.drawLine(12, 19, 22, 19)
        painter.drawLine(12, 24, 19, 24)
    elif kind == "run":
        painter.setPen(Qt.NoPen)
        painter.setBrush(green)
        painter.drawPolygon(QPolygon([QPoint(10, 6), QPoint(10, 28), QPoint(27, 17)]))
    elif kind == "stop":
        painter.setPen(QPen(gray, 1.5))
        painter.setBrush(QColor("#68717a"))
        painter.drawRect(10, 10, 14, 14)
    elif kind == "method":
        painter.setPen(QPen(blue, 2.2))
        for x, y in [(10, 13), (17, 21), (24, 11)]:
            painter.drawLine(x, 6, x, 28)
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QPoint(x, y), 2, 2)
    elif kind == "compare":
        painter.setPen(QPen(blue, 2.0))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRoundedRect(7, 9, 10, 16, 2, 2)
        painter.drawRoundedRect(17, 9, 10, 16, 2, 2)
        painter.drawLine(13, 17, 21, 17)
    elif kind == "template":
        painter.setPen(QPen(blue, 2.0))
        painter.setBrush(QColor("#f7fbff"))
        painter.drawRoundedRect(7, 7, 20, 20, 2, 2)
        painter.drawLine(12, 14, 22, 14)
        painter.drawLine(12, 20, 22, 20)
        painter.setBrush(green)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(24, 24), 5, 5)
    elif kind == "export":
        painter.setPen(QPen(blue, 2.0))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRect(9, 6, 16, 22)
        painter.drawLine(20, 6, 25, 11)
        painter.drawLine(20, 6, 20, 11)
        painter.drawLine(20, 11, 25, 11)
        painter.drawLine(17, 14, 17, 23)
        painter.drawLine(13, 19, 17, 23)
        painter.drawLine(21, 19, 17, 23)
    else:
        painter.setPen(QPen(blue, 2.0))
        painter.drawEllipse(7, 7, 20, 20)

    painter.end()
    return QIcon(pixmap)


def _folder_path():
    path = QPainterPath()
    path.moveTo(5, 25)
    path.lineTo(8, 11)
    path.lineTo(16, 11)
    path.lineTo(19, 15)
    path.lineTo(29, 15)
    path.lineTo(26, 27)
    path.lineTo(5, 27)
    path.closeSubpath()
    return path


class RibbonButton(QToolButton):
    def __init__(self, text: str, icon_kind: str = "default", primary: bool = False, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.set_icon_kind(icon_kind)
        self.setIconSize(QSize(34, 34))
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setProperty("class", "ribbonButton")
        self.setProperty("primary", primary)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_icon_kind(self, icon_kind: str):
        self.setIcon(_ribbon_icon(icon_kind))


class RibbonBar(QFrame):
    import_data = pyqtSignal()
    import_truth = pyqtSignal()
    run_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    method_clicked = pyqtSignal()
    compare_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    sorting_result_clicked = pyqtSignal()
    template_clicked = pyqtSignal()

    def __init__(self, run_text="开始运行", method_text="方案选择", export_text="导出结果", parent=None):
        super().__init__(parent)
        self.setProperty("class", "ribbon")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.import_btn = RibbonButton("导入数据", "import", primary=True)
        self.truth_btn = RibbonButton("导入Sorted", "truth")
        self.truth_btn.setVisible(False)
        self.run_btn = RibbonButton(run_text, "run", primary=True)
        self.stop_btn = RibbonButton("停止", "stop")
        self.method_btn = RibbonButton(method_text, "method")
        self.compare_btn = RibbonButton("对比显示", "compare")
        self.sorting_result_btn = RibbonButton("导入分选结果", "import")
        self.sorting_result_btn.setVisible(False)
        self.template_btn = RibbonButton("生成模板库", "template")
        self.template_btn.setVisible(False)
        self.export_btn = RibbonButton(export_text, "export")

        for button in [
            self.import_btn,
            self.truth_btn,
            self.run_btn,
            self.stop_btn,
            self.method_btn,
            self.compare_btn,
            self.sorting_result_btn,
            self.template_btn,
            self.export_btn,
        ]:
            layout.addWidget(button)
        layout.addStretch(1)

        self.import_btn.clicked.connect(self.import_data.emit)
        self.truth_btn.clicked.connect(self.import_truth.emit)
        self.run_btn.clicked.connect(self.run_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.method_btn.clicked.connect(self.method_clicked.emit)
        self.compare_btn.clicked.connect(self.compare_clicked.emit)
        self.sorting_result_btn.clicked.connect(self.sorting_result_clicked.emit)
        self.template_btn.clicked.connect(self.template_clicked.emit)
        self.export_btn.clicked.connect(self.export_clicked.emit)

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.import_btn.setEnabled(not running)
        self.truth_btn.setEnabled(not running)
        self.method_btn.setEnabled(not running)
        self.sorting_result_btn.setEnabled(not running)
        self.template_btn.setEnabled(not running)
        self.export_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)


class TrackPanel(QFrame):
    track_visibility_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.track_checks: Dict[int, QCheckBox] = {}
        self.content = QVBoxLayout(self)
        self.content.setContentsMargins(12, 12, 12, 12)
        self.content.setSpacing(10)

        title = QLabel("轨迹 / 样本列表")
        title.setProperty("class", "sectionTitle")
        self.track_title = title
        self.content.addWidget(title)
        self.set_tracks([])

    def set_tracks(self, track_ids: Iterable[int]):
        while self.content.count() > 1:
            item = self.content.takeAt(1)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.track_checks = {}
        track_ids = [int(track) for track in track_ids if int(track) > 0]
        if not track_ids:
            hint = QLabel("请先进行分选分析")
            hint.setProperty("class", "subtle")
            hint.setWordWrap(True)
            self.content.addWidget(hint)
            self.content.addStretch(1)
            self._emit_track_state()
            return
        self._add_track_row("全选", -1, TRACK_COLORS[-1], True, is_all=True)
        for track_id in track_ids:
            self._add_track_row(f"轨迹 {track_id}", track_id, track_color(track_id), True)
        self._add_track_row("未分配脉冲", 0, track_color(0), True)
        self.content.addStretch(1)
        self._emit_track_state()

    def _add_track_row(self, text: str, track_id: int, color: str, checked: bool, is_all: bool = False):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        check = QCheckBox(text)
        check.setChecked(checked)
        row_layout.addWidget(check, 1)
        row_layout.addWidget(ColorDot(color), 0, Qt.AlignRight)
        self.content.addWidget(row)
        if is_all:
            check.stateChanged.connect(self._toggle_all_tracks)
        else:
            self.track_checks[track_id] = check
            check.stateChanged.connect(self._emit_track_state)

    def _toggle_all_tracks(self, state):
        for check in self.track_checks.values():
            check.blockSignals(True)
            check.setChecked(state == Qt.Checked)
            check.blockSignals(False)
        self._emit_track_state()

    def _emit_track_state(self):
        self.track_visibility_changed.emit({track_id: check.isChecked() for track_id, check in self.track_checks.items()})


class MethodPanel(QFrame):
    sorting_method_changed = pyqtSignal(str)
    recognition_method_changed = pyqtSignal(str)
    pipeline_changed = pyqtSignal(str)
    display_options_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        options_title = QLabel("显示选项")
        options_title.setProperty("class", "sectionTitle")
        layout.addWidget(options_title)
        self.option_checks: Dict[str, QCheckBox] = {}
        for text in ["显示图例", "显示网格", "显示轨迹标签", "显示置信度"]:
            check = QCheckBox(text)
            check.setChecked(True)
            check.stateChanged.connect(self.display_options_changed.emit)
            layout.addWidget(check)
            self.option_checks[text] = check

        summary_title = QLabel("分析摘要 / 识别摘要")
        summary_title.setProperty("class", "sectionTitle")
        layout.addWidget(summary_title)
        self.summary_values: Dict[str, QLabel] = {}
        for left, right in [
            ("分选轨迹数", "-"),
            ("已分配脉冲数", "-"),
            ("未分配脉冲数", "-"),
            ("平均 PRI", "-"),
            ("平均脉宽", "-"),
            ("识别轨迹数", "-"),
            ("平均置信度", "-"),
            ("类别数", "-"),
        ]:
            layout.addWidget(self._summary_row(left, right))
        layout.addStretch(1)

    def selected_sorting_method(self) -> str:
        return pipeline_definition(SORTING_PIPELINE_ID).sorter

    def selected_recognition_method(self) -> str:
        return RECOGNITION_MODEL

    def selected_pipeline_id(self) -> str:
        return SORTING_PIPELINE_ID

    def set_pipeline_id(self, pipeline_id: str):
        return

    def show_sorting_selector(self):
        return

    def show_recognition_selector(self):
        return

    def display_options(self) -> Dict[str, bool]:
        return {name: check.isChecked() for name, check in self.option_checks.items()}

    def update_summary(self, values: Dict[str, object]):
        for key, value in values.items():
            if key in self.summary_values:
                self.summary_values[key].setText(str(value))

    def _summary_row(self, left, right):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(left)
        value = QLabel(str(right))
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value.setStyleSheet("font-weight: 700; color: #0052B5;")
        self.summary_values[left] = value
        layout.addWidget(label, 1)
        layout.addWidget(value)
        return row

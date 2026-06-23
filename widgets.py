from PyQt5.QtCore import QPoint, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygon
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


TRACK_COLORS = ["#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#78909C"]


class ColorDot(QWidget):
    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = QColor(color)
        self.setFixedSize(14, 14)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(2, 2, 10, 10)


def _ribbon_icon(kind):
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
    def __init__(self, text, icon_kind="default", primary=False, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.set_icon_kind(icon_kind)
        self.setIconSize(QSize(34, 34))
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setProperty("class", "ribbonButton")
        self.setProperty("primary", primary)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_icon_kind(self, icon_kind):
        self.setIcon(_ribbon_icon(icon_kind))


class RibbonBar(QFrame):
    import_data = pyqtSignal()
    run_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    method_clicked = pyqtSignal()
    compare_clicked = pyqtSignal()
    export_clicked = pyqtSignal()

    def __init__(self, run_text="开始运行", method_text="方法选择", export_text="导出结果", parent=None):
        super().__init__(parent)
        self.setProperty("class", "ribbon")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self.import_btn = RibbonButton("导入数据", "import", primary=True)
        self.run_btn = RibbonButton(run_text, "run", primary=True)
        self.stop_btn = RibbonButton("停止", "stop")
        self.method_btn = RibbonButton(method_text, "method")
        self.compare_btn = RibbonButton("对比显示", "compare")
        self.export_btn = RibbonButton(export_text, "export")

        for btn in [
            self.import_btn,
            self.run_btn,
            self.stop_btn,
            self.method_btn,
            self.compare_btn,
            self.export_btn,
        ]:
            layout.addWidget(btn)
        layout.addStretch(1)

        self.import_btn.clicked.connect(self.import_data.emit)
        self.run_btn.clicked.connect(self.run_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        self.method_btn.clicked.connect(self.method_clicked.emit)
        self.compare_btn.clicked.connect(self.compare_clicked.emit)
        self.export_btn.clicked.connect(self.export_clicked.emit)


class DataPanel(QFrame):
    track_visibility_changed = pyqtSignal(list)

    def __init__(self, show_tracks=False, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.track_checks = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        if show_tracks:
            track_title = QLabel("轨迹 / 样本列表")
            track_title.setProperty("class", "sectionTitle")
            layout.addWidget(track_title)
            self._add_track_row(layout, "全选", TRACK_COLORS[-1], True, is_all=True)
            for index, color in enumerate(TRACK_COLORS[:4], start=1):
                self._add_track_row(layout, f"轨迹 {index}", color, True)
            self._add_track_row(layout, "未分配脉冲", TRACK_COLORS[-1], True)
        else:
            empty_title = QLabel("数据已由顶部工具栏导入")
            empty_title.setProperty("class", "subtle")
            empty_title.setWordWrap(True)
            layout.addWidget(empty_title)

        layout.addStretch(1)

    def _add_track_row(self, layout, text, color, checked, is_all=False):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        check = QCheckBox(text)
        check.setChecked(checked)
        dot = ColorDot(color)
        row_layout.addWidget(check, 1)
        row_layout.addWidget(dot, 0, Qt.AlignRight)
        layout.addWidget(row)

        if is_all:
            check.stateChanged.connect(self._toggle_all_tracks)
        else:
            self.track_checks.append(check)
            check.stateChanged.connect(self._emit_track_state)

    def _toggle_all_tracks(self, state):
        for check in self.track_checks:
            check.blockSignals(True)
            check.setChecked(state == Qt.Checked)
            check.blockSignals(False)
        self._emit_track_state()

    def _emit_track_state(self):
        self.track_visibility_changed.emit([check.isChecked() for check in self.track_checks])


class MethodPanel(QFrame):
    sorting_method_changed = pyqtSignal(str)
    recognition_method_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.sort_combo = self._add_method_combo(layout, "分选方法", ["MHT", "DBSCAN", "K-means"])
        self.recognition_combo = self._add_method_combo(layout, "识别方法", ["SVM", "KNN", "随机森林", "CNN"])

        options_title = QLabel("显示选项")
        options_title.setProperty("class", "sectionTitle")
        layout.addWidget(options_title)
        self.option_checks = {}
        for text in ["显示图例", "显示网格", "显示轨迹标签", "显示置信度"]:
            check = QCheckBox(text)
            check.setChecked(True)
            layout.addWidget(check)
            self.option_checks[text] = check

        summary_title = QLabel("分析摘要 / 识别摘要")
        summary_title.setProperty("class", "sectionTitle")
        layout.addWidget(summary_title)
        self.summary_values = {}
        for left, right in [
            ("分选轨迹数", "4"),
            ("已分配脉冲数", "49,951"),
            ("未分配脉冲数", "49"),
            ("平均 PRI", "1.82 ms"),
            ("平均脉宽", "2.36 us"),
            ("识别准确率", "94.2%"),
            ("平均置信度", "92.5%"),
        ]:
            layout.addWidget(self._summary_row(left, right))
        layout.addStretch(1)

        self.sort_combo.currentTextChanged.connect(self.sorting_method_changed.emit)
        self.recognition_combo.currentTextChanged.connect(self.recognition_method_changed.emit)

    def _add_method_combo(self, layout, title_text, names):
        title = QLabel(title_text)
        title.setProperty("class", "sectionTitle")
        layout.addWidget(title)

        combo = QComboBox()
        combo.addItems(names)
        combo.setCursor(Qt.PointingHandCursor)
        combo.setMinimumHeight(32)
        layout.addWidget(combo)
        return combo

    def selected_sorting_method(self):
        return self.sort_combo.currentText()

    def selected_recognition_method(self):
        return self.recognition_combo.currentText()

    def show_sorting_selector(self):
        self.sort_combo.showPopup()

    def show_recognition_selector(self):
        self.recognition_combo.showPopup()

    def display_options(self):
        return {name: check.isChecked() for name, check in self.option_checks.items()}

    def update_summary(self, values):
        for key, value in values.items():
            if key in self.summary_values:
                self.summary_values[key].setText(str(value))

    def _summary_row(self, left, right):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(left)
        value = QLabel(right)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value.setStyleSheet("font-weight: 700; color: #0052B5;")
        self.summary_values[left] = value
        layout.addWidget(label, 1)
        layout.addWidget(value)
        return row

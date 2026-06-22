import os
import time
from datetime import datetime
from typing import Callable, Dict, Optional

import pandas as pd
from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QComboBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.data_loader import LoadedData, load_pdw_file
from core.exporter import export_full_csv, export_log, export_recognition_csv, export_report, export_sorting_csv
from core.external_model_adapters import run_zeng_template_training, zeng_template_library_exists, zeng_template_library_path
from core.pipelines import (
    FULL_PIPELINE_ID,
    PipelineRunResult,
    SortingPipelineResult,
    pipeline_label,
    run_cycle_period_pipeline_from_hdbscan,
    run_pipeline,
    run_pipeline_from_hdbscan,
    run_sorting_pipeline,
)
from core.recognition_algorithms import RecognitionOutput, run_recognition
from core.sorting_algorithms import SortingOutput
from utils.helpers import format_elapsed

from .charts import (
    ChartCard,
    TableCard,
    plot_class_stats,
    plot_feature_projection,
    plot_pa_scatter,
    plot_probability,
    plot_pw_scatter,
    plot_quality,
    plot_rf_scatter,
    render_full_detail_figure,
    plot_sort_scatter,
    plot_timeline,
    plot_track_bars,
)
from .widgets import MethodPanel, RibbonBar, TrackPanel


class Worker(QObject):
    progress = pyqtSignal(int, str)
    partial = pyqtSignal(object)
    finished = pyqtSignal(object, float)
    failed = pyqtSignal(str)

    def __init__(self, func: Callable, *args, stages=None, pass_progress=False, pass_stream=False):
        super().__init__()
        self.func = func
        self.args = args
        self.stages = stages or ("准备数据", "算法运行中", "结果刷新")
        self.pass_progress = pass_progress
        self.pass_stream = pass_stream
        self.cancelled = False

    def run(self):
        start = time.perf_counter()
        try:
            if not self.pass_progress:
                self.progress.emit(15, self.stages[0])
            if self.cancelled:
                self.failed.emit("任务已取消")
                return
            if self.pass_progress:
                kwargs = {
                    "progress_callback": self.progress.emit,
                    "should_cancel": lambda: self.cancelled,
                }
                if self.pass_stream:
                    kwargs["stream_callback"] = self.partial.emit
                result = self.func(*self.args, **kwargs)
            else:
                self.progress.emit(55, self.stages[1])
                result = self.func(*self.args)
            if self.cancelled:
                self.failed.emit("任务已取消")
                return
            self.progress.emit(100, self.stages[2])
            self.finished.emit(result, time.perf_counter() - start)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self.cancelled = True


def with_display_track_ids(data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if data is None or "Track_ID" not in data.columns:
        return data
    out = data.copy()
    tracks = pd.to_numeric(out["Track_ID"], errors="coerce").fillna(0).astype(int)
    mapping: Dict[int, int] = {}
    next_display_id = 1
    display_ids = []
    for track_id in tracks:
        if int(track_id) <= 0:
            display_ids.append(0)
            continue
        if int(track_id) not in mapping:
            mapping[int(track_id)] = next_display_id
            next_display_id += 1
        display_ids.append(mapping[int(track_id)])
    out["Display_Track_ID"] = display_ids
    return out


class AnalysisPage(QWidget):
    def __init__(
        self,
        ribbon: RibbonBar,
        chart_specs,
        show_tracks=False,
        table_card: Optional[TableCard] = None,
        show_stage_selector=False,
    ):
        super().__init__()
        self.data: Optional[pd.DataFrame] = None
        self.track_results: Optional[pd.DataFrame] = None
        self.track_visibility: Dict[int, bool] = {}
        self.chart_specs = chart_specs
        self.table_card = table_card
        self.stage_results = []
        self._current_track_ids = None
        self._refresh_index = 0
        self._refresh_options = {}
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._render_next_chart)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 8)
        outer.setSpacing(8)
        outer.addWidget(ribbon)

        self.stage_bar = None
        self.stage_combo = None
        if show_stage_selector:
            self.stage_bar = QFrame()
            self.stage_bar.setProperty("class", "card")
            stage_layout = QHBoxLayout(self.stage_bar)
            stage_layout.setContentsMargins(12, 8, 12, 8)
            stage_layout.setSpacing(10)
            stage_label = QLabel("阶段结果")
            stage_label.setProperty("class", "sectionTitle")
            self.stage_combo = QComboBox()
            self.stage_combo.setMinimumWidth(260)
            self.stage_combo.currentIndexChanged.connect(self._apply_stage_selection)
            self.stage_combo.addItem("暂无阶段结果")
            stage_layout.addWidget(stage_label)
            stage_layout.addWidget(self.stage_combo)
            stage_layout.addStretch(1)
            outer.addWidget(self.stage_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        self.track_panel = None
        if show_tracks:
            self.track_panel = TrackPanel()
            self.track_panel.setMinimumWidth(190)
            self.track_panel.setMaximumWidth(245)
            self.track_panel.track_visibility_changed.connect(self.set_track_visibility)
            splitter.addWidget(self.track_panel)

        center = QWidget()
        grid = QGridLayout(center)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        self.chart_cards = []
        for index, (title, plotter) in enumerate(chart_specs):
            card = ChartCard(title)
            self.chart_cards.append((card, plotter))
            if table_card is not None and index >= 2:
                grid.addWidget(card, 1, 1)
            else:
                grid.addWidget(card, index // 2, index % 2)
        if table_card is not None:
            grid.addWidget(table_card, 1, 0)
        row_count = 2 if table_card is not None else max(1, (len(chart_specs) + 1) // 2)
        for row in range(row_count):
            grid.setRowStretch(row, 1)
        splitter.addWidget(center)

        self.method_panel = MethodPanel()
        self.method_panel.setMinimumWidth(260)
        self.method_panel.setMaximumWidth(320)
        self.method_panel.display_options_changed.connect(self.refresh)
        splitter.addWidget(self.method_panel)
        splitter.setSizes([215, 870, 285] if show_tracks else [960, 285])
        self.clear()

    def set_data(self, data: Optional[pd.DataFrame], track_results: Optional[pd.DataFrame] = None):
        self.data = with_display_track_ids(data)
        self.track_results = track_results
        if self.track_panel is not None and self.data is not None and "Display_Track_ID" in self.data.columns:
            track_ids = tuple(sorted(int(value) for value in self.data["Display_Track_ID"].dropna().unique() if int(value) > 0))
            if track_ids != self._current_track_ids:
                self._current_track_ids = track_ids
                self.track_panel.blockSignals(True)
                self.track_panel.set_tracks(track_ids)
                self.track_panel.blockSignals(False)
        elif self.track_panel is not None:
            if self._current_track_ids != ():
                self._current_track_ids = ()
                self.track_panel.blockSignals(True)
                self.track_panel.set_tracks([])
                self.track_panel.blockSignals(False)
        self.refresh()
        self._update_data_summary()

    def set_stage_results(self, stage_results):
        self.stage_results = list(stage_results or [])
        if self.stage_combo is None:
            if self.stage_results:
                self.set_data(self.stage_results[-1].sorting.data)
            return

        self.stage_combo.blockSignals(True)
        self.stage_combo.clear()
        if self.stage_results:
            for index, stage in enumerate(self.stage_results, start=1):
                suffix = "（最终）" if index == len(self.stage_results) else ""
                self.stage_combo.addItem(f"{index}. {stage.definition.name}{suffix}")
            self.stage_combo.setCurrentIndex(max(0, len(self.stage_results) - 1))
        else:
            self.stage_combo.addItem("暂无阶段结果")
        self.stage_combo.blockSignals(False)
        if self.stage_bar is not None:
            self.stage_bar.setVisible(True)
        if self.stage_results:
            self._apply_stage_selection()

    def clear_stage_results(self):
        self.stage_results = []
        if self.stage_combo is not None:
            self.stage_combo.blockSignals(True)
            self.stage_combo.clear()
            self.stage_combo.addItem("暂无阶段结果")
            self.stage_combo.blockSignals(False)
        if self.stage_bar is not None:
            self.stage_bar.setVisible(True)

    def set_track_visibility(self, visibility: Dict[int, bool]):
        self.track_visibility = visibility
        self.refresh()

    def _apply_stage_selection(self):
        if not self.stage_results:
            return
        index = self.stage_combo.currentIndex() if self.stage_combo is not None else len(self.stage_results) - 1
        index = max(0, min(index, len(self.stage_results) - 1))
        self.set_data(self.stage_results[index].sorting.data)

    def _update_data_summary(self):
        if self.data is None:
            return
        tracks = self.data["Track_ID"] if "Track_ID" in self.data.columns else pd.Series([0] * len(self.data))
        tracks = pd.to_numeric(tracks, errors="coerce").fillna(0).astype(int)
        assigned = int((tracks > 0).sum())
        unassigned = int(len(self.data) - assigned)
        summary = {
            "分选轨迹数": int(tracks[tracks > 0].nunique()),
            "已分配脉冲数": f"{assigned:,}",
            "未分配脉冲数": f"{unassigned:,}",
            "平均 PRI": f"{self.data['PRI'].mean():.3f}" if "PRI" in self.data else "-",
            "平均脉宽": f"{self.data['PW'].mean():.3f}" if "PW" in self.data else "-",
        }
        self.method_panel.update_summary(summary)

    def refresh(self):
        if self.data is None:
            self._refresh_timer.stop()
            self.clear()
            return
        self._refresh_options = self.method_panel.display_options()
        self._refresh_index = 0
        self._refresh_timer.start(0)

    def _render_next_chart(self):
        if self.data is None:
            return
        if self._refresh_index >= len(self.chart_cards):
            if self.table_card is not None:
                self.table_card.set_dataframe(self.track_results)
            return
        card, plotter = self.chart_cards[self._refresh_index]
        self._refresh_index += 1
        if plotter in {plot_feature_projection, plot_probability, plot_class_stats}:
            plotter(card, self.track_results, self._refresh_options, self.track_visibility)
        else:
            plotter(card, self.data, self._refresh_options, self.track_visibility)
        card.set_detail_renderer(
            lambda figure,
            plotter=plotter,
            data=self.data,
            track_results=self.track_results,
            options=dict(self._refresh_options),
            visibility=dict(self.track_visibility): render_full_detail_figure(
                figure, plotter, data, track_results, options, visibility
            )
        )
        self._refresh_timer.start(20)

    def clear(self):
        for card, _ in self.chart_cards:
            card.show_empty()
        if self.table_card is not None:
            self.table_card.set_dataframe(None)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("信号分选与识别系统")
        self.resize(1500, 920)
        self.setMinimumSize(1180, 760)

        self.data: Optional[pd.DataFrame] = None
        self.loaded_file = "未导入"
        self.truth_file = "未导入"
        self.sorting_output: Optional[SortingOutput] = None
        self.sorting_pipeline_output: Optional[SortingPipelineResult] = None
        self.recognition_output: Optional[RecognitionOutput] = None
        self.pipeline_output: Optional[PipelineRunResult] = None
        self.pipeline_result_target_page = "recognition"
        self.sorting_import_target_page = "sorting"
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[Worker] = None
        self.log_lines = []
        self.compare_enabled = False
        self.pending_partial_sorting_data: Optional[pd.DataFrame] = None
        self.partial_sorting_started = False
        self.partial_sorting_timer = QTimer(self)
        self.partial_sorting_timer.setInterval(400)
        self.partial_sorting_timer.setSingleShot(True)
        self.partial_sorting_timer.timeout.connect(self._flush_partial_sorting_update)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._title_bar())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout.addWidget(self.tabs, 1)

        self.home_ribbon = RibbonBar("开始运行", "方法选择", "导出图表")
        self.sort_ribbon = RibbonBar("开始分析", "方法选择", "导出结果")
        self.recognition_ribbon = RibbonBar("开始识别", "模型选择", "导出结果")
        self.export_ribbon = RibbonBar("生成报告", "导出设置", "导出结果")
        self.home_ribbon.import_btn.setText("导入数据")
        self.home_ribbon.truth_btn.setVisible(False)
        self.home_ribbon.method_btn.setVisible(False)
        self.home_ribbon.compare_btn.setVisible(False)

        self.home_page = AnalysisPage(
            self.home_ribbon,
            [
                ("TOA 波形 / 脉冲时间轴", plot_timeline),
                ("原始 PRI vs TOA", plot_sort_scatter),
                ("TOA-脉宽", plot_pw_scatter),
                ("TOA-脉幅", plot_pa_scatter),
            ],
        )
        self.sort_page = AnalysisPage(
            self.sort_ribbon,
            [
                ("分选散点图：PRI vs TOA", plot_sort_scatter),
                ("频率分布：RF vs TOA", plot_rf_scatter),
                ("TOA-脉宽", plot_pw_scatter),
                ("TOA-脉幅", plot_pa_scatter),
                ("轨迹脉冲统计柱状图", plot_track_bars),
                ("已分配 / 未分配比例图", plot_quality),
            ],
            show_tracks=True,
            show_stage_selector=True,
        )
        self.recognition_page = AnalysisPage(
            self.recognition_ribbon,
            [
                ("特征二维投影散点图", plot_feature_projection),
                ("分类概率堆叠条形图", plot_probability),
                ("类别统计柱状图", plot_class_stats),
            ],
            show_tracks=True,
            table_card=TableCard("识别结果表格"),
        )
        self.sort_ribbon.template_btn.setText("导入HDBSCAN结果")
        self.sort_ribbon.method_btn.setVisible(False)
        self.sort_ribbon.compare_btn.setVisible(False)
        self.sort_ribbon.import_btn.setVisible(False)
        self.sort_ribbon.template_btn.setVisible(False)
        self.recognition_ribbon.import_btn.setText("导入训练数据")
        self.recognition_ribbon.method_btn.setVisible(False)
        self.recognition_ribbon.compare_btn.setVisible(False)
        self.recognition_ribbon.sorting_result_btn.setVisible(False)
        self.recognition_ribbon.template_btn.setText("生成模板库")
        self.export_ribbon.import_btn.setVisible(False)
        self.export_ribbon.method_btn.setVisible(False)
        self.export_ribbon.compare_btn.setVisible(False)
        self.export_page = self._export_page()

        self.tabs.addTab(self.home_page, "主页")
        self.tabs.addTab(self.sort_page, "分选分析")
        self.tabs.addTab(self.recognition_page, "信号识别")
        self.tabs.addTab(self.export_page, "结果导出")

        layout.addWidget(self._bottom_panel())
        self._status_bar()
        self._connect_actions()
        self._set_all_running(False)
        self.log("系统启动：等待导入数据")

        self.clock = QTimer(self)
        self.clock.timeout.connect(self._update_status_time)
        self.clock.start(1000)
        self._update_status_time()

        self.task_progress_timer = QTimer(self)
        self.task_progress_timer.timeout.connect(self._advance_task_progress)
        self.task_progress_cap = 90

    def _title_bar(self):
        frame = QFrame()
        frame.setObjectName("titleBar")
        frame.setFixedHeight(72)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(20, 8, 20, 8)
        title = QLabel("信号分选与识别系统")
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Signal Sorting and Recognition System")
        subtitle.setObjectName("appSubTitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame

    def _export_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(8)
        layout.addWidget(self.export_ribbon)
        body = QHBoxLayout()
        layout.addLayout(body, 1)
        card = QFrame()
        card.setProperty("class", "card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        title = QLabel("结果导出")
        title.setProperty("class", "sectionTitle")
        text = QLabel("支持导出：分选结果 CSV、识别结果 CSV、图表 PNG、运行日志 TXT、总结报告 TXT/HTML。")
        text.setWordWrap(True)
        card_layout.addWidget(title)
        card_layout.addWidget(text)
        for label, handler in [
            ("导出分选结果 CSV", self.export_sorting_file),
            ("导出识别结果 CSV", self.export_recognition_file),
            ("导出当前图表 PNG", self.export_current_charts),
            ("导出运行日志 TXT", self.export_log_file),
            ("生成总结报告", self.export_report_file),
        ]:
            button = QPushButton(label)
            button.clicked.connect(handler)
            card_layout.addWidget(button)
        card_layout.addStretch(1)
        body.addWidget(card, 1)
        self.export_method_panel = MethodPanel()
        self.export_method_panel.setMinimumWidth(260)
        self.export_method_panel.setMaximumWidth(320)
        body.addWidget(self.export_method_panel)
        return page

    def _bottom_panel(self):
        frame = QFrame()
        frame.setProperty("class", "card")
        frame.setFixedHeight(152)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        log_box = QVBoxLayout()
        title = QLabel("运行日志")
        title.setProperty("class", "sectionTitle")
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_box.addWidget(title)
        log_box.addWidget(self.log_edit, 1)
        layout.addLayout(log_box, 3)
        status_box = QVBoxLayout()
        status_title = QLabel("运行状态")
        status_title.setProperty("class", "sectionTitle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.step_label = QLabel("当前操作：等待操作")
        self.elapsed_label = QLabel("本次耗时：00:00:00.000")
        status_box.addWidget(status_title)
        status_box.addWidget(self.progress)
        status_box.addWidget(self.step_label)
        status_box.addWidget(self.elapsed_label)
        status_box.addStretch(1)
        layout.addLayout(status_box, 1)
        return frame

    def _status_bar(self):
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_ready = QLabel("就绪")
        self.status_project = QLabel("项目：Demo_Project")
        self.status_data = QLabel(f"数据：{self.loaded_file}")
        self.status_time = QLabel()
        for widget in [self.status_ready, self.status_project, self.status_data, self.status_time]:
            widget.setMinimumWidth(180)
            bar.addWidget(widget)

    def _connect_actions(self):
        self.home_ribbon.import_data.connect(self.import_home_data)
        self.recognition_ribbon.import_data.connect(self.import_zeng_training_data)
        for ribbon in [self.home_ribbon, self.sort_ribbon, self.recognition_ribbon, self.export_ribbon]:
            ribbon.stop_clicked.connect(self.stop_current_task)
            ribbon.method_clicked.connect(self.show_method_selector)
            ribbon.compare_clicked.connect(self.toggle_compare)
        self.sort_ribbon.template_clicked.connect(lambda: self.import_sorting_result("hdbscan"))
        self.recognition_ribbon.sorting_result_clicked.connect(lambda: self.import_sorting_result("recognition"))
        self.recognition_ribbon.template_clicked.connect(self.import_zeng_training_data)
        self.home_ribbon.run_clicked.connect(self.start_home_pipeline)
        self.sort_ribbon.run_clicked.connect(self.start_sorting)
        self.recognition_ribbon.run_clicked.connect(self.start_recognition)
        self.export_ribbon.run_clicked.connect(self.export_report_file)
        self.home_ribbon.export_clicked.connect(self.export_current_charts)
        self.sort_ribbon.export_clicked.connect(self.export_sorting_file)
        self.recognition_ribbon.export_clicked.connect(self.export_recognition_file)
        self.export_ribbon.export_clicked.connect(self.export_full_result_file)

        for panel in self._method_panels():
            panel.pipeline_changed.connect(self._sync_pipeline_selection)
            panel.sorting_method_changed.connect(lambda m: self.log(f"分选方法切换为：{m}"))
            panel.recognition_method_changed.connect(lambda m: self.log(f"识别模型切换为：{m}"))
        self._update_template_button_visibility()
        self._update_home_import_button()

    def _method_panels(self):
        return [
            self.home_page.method_panel,
            self.sort_page.method_panel,
            self.recognition_page.method_panel,
            self.export_method_panel,
        ]

    def _sync_pipeline_selection(self, pipeline_id: str):
        for panel in self._method_panels():
            if panel.selected_pipeline_id() != pipeline_id:
                panel.set_pipeline_id(pipeline_id)
        self._update_template_button_visibility()

    def _update_template_button_visibility(self):
        self.recognition_ribbon.template_btn.setVisible(True)

    def _update_home_import_button(self):
        if self.data is None:
            self.home_ribbon.import_btn.setText("导入数据")
        elif self.truth_file == "未导入":
            self.home_ribbon.import_btn.setText("导入Sorted")
        else:
            self.home_ribbon.import_btn.setText("重新导入数据")

    def import_home_data(self):
        if self.data is None:
            self.import_data()
            return
        if self.truth_file == "未导入":
            self.import_truth_data()
            return
        self.import_data()

    def import_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 Merge 数据文件", "", "Data Files (*.csv *.txt);;All Files (*)")
        if not path:
            return
        self.log(f"Merge 数据导入开始：{path}")
        self._run_background(
            load_pdw_file,
            path,
            done=self._on_import_done,
            stages=("准备导入Merge", "正在读取Merge数据文件", "刷新数据视图"),
        )

    def _on_import_done(self, loaded, elapsed: float):
        self.data = loaded.data
        self.loaded_file = loaded.filename
        self.truth_file = "Merge内置真值" if "Original_Track_ID" in self.data.columns else "未导入"
        self.sorting_output = None
        self.sorting_pipeline_output = None
        self.recognition_output = None
        self.pipeline_output = None
        self.status_data.setText(f"数据：{self.loaded_file}")
        self._refresh_pages()
        self._update_home_import_button()
        self._finish_status("Merge 数据导入", elapsed)
        truth_text = f"，真值：{self.truth_file}" if self.truth_file != "未导入" else ""
        self.log(f"导入 Merge 完成：{loaded.filename}，脉冲数：{len(self.data):,}，字段：{', '.join(loaded.fields)}{truth_text}")

    def import_truth_data(self):
        if self.data is None:
            QMessageBox.information(self, "提示", "请先导入 Merge 数据文件。")
            return
        path, _ = QFileDialog.getOpenFileName(self, "选择 Sorted 真值文件", "", "Data Files (*.csv *.txt);;All Files (*)")
        if not path:
            return
        self.log(f"Sorted 真值导入开始：{path}")
        self._run_background(
            self._load_sorted_truth_file,
            path,
            self.data.copy(),
            done=self._on_truth_import_done,
            stages=("准备导入Sorted", "正在读取Sorted真值文件", "刷新真值视图"),
        )

    def _read_table_auto(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        for sep in [None, ",", "\t", r"\s+"]:
            try:
                table = pd.read_csv(path, sep=sep, engine="python")
                if table.shape[1] > 1:
                    return table
            except Exception:
                continue
        raise ValueError("无法识别文件分隔符，请检查 Sorted 文件格式")

    def _canonical_truth_table(self, table: pd.DataFrame) -> pd.DataFrame:
        normalized = []
        for index, column in enumerate(table.columns):
            name = str(column).strip().lower().replace(" ", "_").replace("-", "_")
            normalized.append((index, name))

        def pick(candidates):
            for candidate in candidates:
                for index, name in normalized:
                    if name == candidate:
                        return index
            return None

        pulse_index = pick(["pulse_id", "pulseid", "id"])
        toa_index = pick(["toa", "toa(s)", "toa_s", "toa_us", "time", "time_us"])
        track_truth_index = pick(["sigidx", "sig_idx", "true_track_id", "track_id", "trackid", "track"])
        label_truth_index = pick(["label", "true_label", "true_label_id", "truelabel", "class", "class_label", "target"])
        if track_truth_index is None and label_truth_index is None:
            raise ValueError("Sorted truth file is missing SigIdx or LABEL")

        truth = pd.DataFrame(index=table.index)
        if track_truth_index is not None:
            truth["True_Track_ID"] = table.iloc[:, track_truth_index]
        if label_truth_index is not None:
            truth["True_Label"] = table.iloc[:, label_truth_index]
        if pulse_index is not None:
            truth["Pulse_ID"] = table.iloc[:, pulse_index]
        if toa_index is not None:
            truth["TOA"] = table.iloc[:, toa_index]
        if "TOA" in truth.columns:
            truth["TOA"] = pd.to_numeric(truth["TOA"], errors="coerce")
        if "Pulse_ID" in truth.columns:
            truth["Pulse_ID"] = pd.to_numeric(truth["Pulse_ID"], errors="coerce")
        truth_columns = [column for column in ("True_Track_ID", "True_Label") if column in truth.columns]
        truth = truth.dropna(subset=truth_columns, how="all").reset_index(drop=True)
        if truth.empty:
            raise ValueError("Sorted truth file has no usable truth rows")
        return truth
        truth_index = pick([
            "sigidx",
            "sig_idx",
            "true_track_id",
            "true_label",
            "true_label_id",
            "truelabel",
            "target",
            "label",
            "class",
            "class_label",
            "track_id",
            "trackid",
            "track",
        ])
        if truth_index is None:
            raise ValueError("Sorted 真值文件缺少 SigIdx/Track_ID/Label 真值字段")

        truth = pd.DataFrame({"True_Track_ID": table.iloc[:, truth_index]})
        if pulse_index is not None:
            truth["Pulse_ID"] = table.iloc[:, pulse_index]
        if toa_index is not None:
            truth["TOA"] = table.iloc[:, toa_index]
        if "TOA" in truth.columns:
            truth["TOA"] = pd.to_numeric(truth["TOA"], errors="coerce")
        if "Pulse_ID" in truth.columns:
            truth["Pulse_ID"] = pd.to_numeric(truth["Pulse_ID"], errors="coerce")
        truth = truth.dropna(subset=["True_Track_ID"]).reset_index(drop=True)
        if truth.empty:
            raise ValueError("Sorted 真值文件没有可用真值行")
        return truth

    def _toa_for_match(self, values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        if numeric.notna().any() and float(numeric.max()) > 1_000.0:
            return numeric / 1_000_000.0
        return numeric

    def _load_sorted_truth_file(self, path: str, current_data: pd.DataFrame) -> LoadedData:
        truth = self._canonical_truth_table(self._read_table_auto(path))
        data = current_data.copy()
        data.attrs.update(current_data.attrs)

        labels = pd.Series(index=data.index, dtype=object)
        if "Pulse_ID" in data.columns and "Pulse_ID" in truth.columns:
            truth_by_id = truth.dropna(subset=["Pulse_ID"]).drop_duplicates("Pulse_ID").set_index("Pulse_ID")["True_Track_ID"]
            labels = pd.to_numeric(data["Pulse_ID"], errors="coerce").map(truth_by_id)
        elif len(truth) == len(data):
            data_order = data.sort_values("TOA").index if "TOA" in data.columns else data.index
            truth_sorted = truth.sort_values("TOA") if "TOA" in truth.columns else truth
            labels.loc[data_order] = truth_sorted["True_Track_ID"].to_numpy(dtype=object).reshape(-1)
        elif "TOA" in data.columns and "TOA" in truth.columns:
            left = pd.DataFrame({"_idx": data.index, "_toa": self._toa_for_match(data["TOA"])}).dropna(subset=["_toa"]).sort_values("_toa")
            right = pd.DataFrame({"_toa": self._toa_for_match(truth["TOA"]), "True_Track_ID": truth["True_Track_ID"]}).dropna(subset=["_toa"]).sort_values("_toa")
            matched = pd.merge_asof(left, right, on="_toa", direction="nearest")
            labels.loc[matched["_idx"].to_numpy()] = matched["True_Track_ID"].to_numpy(dtype=object).reshape(-1)
        else:
            raise ValueError("Sorted 真值无法与 Merge 对齐：需要 Pulse_ID、相同行数或 TOA 字段")

        matched_count = int(labels.notna().sum())
        if matched_count == 0:
            raise ValueError("Sorted 真值与当前 Merge 没有匹配到任何脉冲")
        data["Original_Track_ID"] = labels
        data["True_Track_ID"] = labels
        data.attrs["truth_path"] = os.path.abspath(path)
        data.attrs["truth_match_count"] = matched_count
        data.attrs["truth_total_count"] = len(data)
        return LoadedData(path=path, filename=os.path.basename(path), data=data, fields=list(data.columns))

    def _load_sorted_truth_file(self, path: str, current_data: pd.DataFrame) -> LoadedData:
        truth = self._canonical_truth_table(self._read_table_auto(path))
        data = current_data.copy()
        data.attrs.update(current_data.attrs)
        truth_value_columns = [column for column in ("True_Track_ID", "True_Label") if column in truth.columns]
        aligned_truth = pd.DataFrame(index=data.index)

        if "Pulse_ID" in data.columns and "Pulse_ID" in truth.columns:
            truth_by_id = truth.dropna(subset=["Pulse_ID"]).drop_duplicates("Pulse_ID").set_index("Pulse_ID")
            pulse_ids = pd.to_numeric(data["Pulse_ID"], errors="coerce")
            for column in truth_value_columns:
                aligned_truth[column] = pulse_ids.map(truth_by_id[column])
        elif len(truth) == len(data):
            data_order = data.sort_values("TOA").index if "TOA" in data.columns else data.index
            truth_sorted = truth.sort_values("TOA") if "TOA" in truth.columns else truth
            for column in truth_value_columns:
                aligned_truth.loc[data_order, column] = truth_sorted[column].to_numpy(dtype=object).reshape(-1)
        elif "TOA" in data.columns and "TOA" in truth.columns:
            left = pd.DataFrame({"_idx": data.index, "_toa": self._toa_for_match(data["TOA"])}).dropna(subset=["_toa"]).sort_values("_toa")
            right = truth[truth_value_columns].copy()
            right.insert(0, "_toa", self._toa_for_match(truth["TOA"]))
            right = right.dropna(subset=["_toa"]).sort_values("_toa")
            matched = pd.merge_asof(left, right, on="_toa", direction="nearest")
            for column in truth_value_columns:
                aligned_truth.loc[matched["_idx"].to_numpy(), column] = matched[column].to_numpy(dtype=object).reshape(-1)
        else:
            raise ValueError("Sorted truth cannot align with Merge: need Pulse_ID, same row count, or TOA")

        match_column = "True_Track_ID" if "True_Track_ID" in aligned_truth.columns else truth_value_columns[0]
        matched_count = int(aligned_truth[match_column].notna().sum())
        if matched_count == 0:
            raise ValueError("Sorted truth did not match any pulse in current Merge")
        if "True_Track_ID" in aligned_truth.columns:
            data["Original_Track_ID"] = aligned_truth["True_Track_ID"]
            data["True_Track_ID"] = aligned_truth["True_Track_ID"]
        if "True_Label" in aligned_truth.columns:
            data["Original_Label"] = aligned_truth["True_Label"]
            data["True_Label"] = aligned_truth["True_Label"]
        data.attrs["truth_path"] = os.path.abspath(path)
        data.attrs["truth_match_count"] = matched_count
        data.attrs["truth_total_count"] = len(data)
        return LoadedData(path=path, filename=os.path.basename(path), data=data, fields=list(data.columns))

    def _on_truth_import_done(self, loaded: LoadedData, elapsed: float):
        self.data = loaded.data
        self.truth_file = loaded.filename
        self._attach_truth_to_outputs()
        self._refresh_pages()
        self._update_home_import_button()
        self._finish_status("Sorted 真值导入", elapsed)
        fallback_column = next((column for column in ("Original_Track_ID", "True_Track_ID", "True_Label") if column in self.data.columns), None)
        fallback_matched = int(self.data[fallback_column].notna().sum()) if fallback_column is not None else 0
        matched = int(self.data.attrs.get("truth_match_count", fallback_matched))
        total = int(self.data.attrs.get("truth_total_count", len(self.data)))
        self.log(f"导入 Sorted 真值完成：{loaded.filename}，匹配脉冲：{matched:,}/{total:,}")

    def _attach_truth_to_outputs(self):
        if self.data is None or not any(column in self.data.columns for column in ("Original_Track_ID", "True_Track_ID", "Original_Label", "True_Label")):
            return
        truth_columns = [col for col in ("Original_Track_ID", "True_Track_ID", "Original_Label", "True_Label") if col in self.data.columns]

        def attach(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if frame is None or len(frame) != len(self.data):
                return frame
            out = frame.copy()
            for column in truth_columns:
                out[column] = self.data[column].to_numpy()
            out.attrs.update(frame.attrs)
            out.attrs["truth_path"] = self.data.attrs.get("truth_path", "")
            return out

        if self.sorting_output is not None:
            self.sorting_output.data = attach(self.sorting_output.data)
        if self.recognition_output is not None:
            self.recognition_output.data = attach(self.recognition_output.data)
        if self.sorting_pipeline_output is not None:
            self.sorting_pipeline_output.sorting.data = attach(self.sorting_pipeline_output.sorting.data)
            for stage in self.sorting_pipeline_output.stage_results:
                stage.sorting.data = attach(stage.sorting.data)
        if self.pipeline_output is not None:
            self.pipeline_output.sorting.data = attach(self.pipeline_output.sorting.data)
            self.pipeline_output.recognition.data = attach(self.pipeline_output.recognition.data)
            for stage in self.pipeline_output.stage_results:
                stage.sorting.data = attach(stage.sorting.data)

    def import_sorting_result(self, target_page: str = "sorting"):
        title = "选择HDBSCAN第一阶段结果文件" if target_page == "hdbscan" else "选择分选结果文件"
        path, _ = QFileDialog.getOpenFileName(self, title, "", "CSV/TXT Files (*.csv *.txt);;All Files (*)")
        if not path:
            return
        self.sorting_import_target_page = target_page
        self.tabs.setCurrentWidget(self.recognition_page if target_page == "recognition" else self.sort_page)
        self.log(f"分选结果导入开始：{path}")
        loader = self._load_hdbscan_stage_file if target_page == "hdbscan" else load_pdw_file
        self._run_background(
            loader,
            path,
            done=self._on_sorting_result_loaded,
            stages=("准备导入", "正在读取分选结果", "恢复分选轨迹"),
        )

    def _load_hdbscan_stage_file(self, path: str):
        table = None
        for sep in [None, ",", "\t", r"\s+"]:
            try:
                candidate = pd.read_csv(path, sep=sep, engine="python")
                if candidate.shape[1] > 1:
                    table = candidate
                    break
            except Exception:
                continue
        if table is None or table.empty:
            raise ValueError("无法读取HDBSCAN第一阶段结果文件")

        rename = {}
        for column in table.columns:
            text = str(column).strip()
            lower = text.lower().replace(" ", "_")
            if lower in {"toa", "toa(s)", "toa_s", "time"}:
                rename[column] = "TOA"
            elif lower in {"sigidx", "sig_idx", "track_id", "trackid", "track"}:
                rename[column] = "Track_ID"
            elif lower in {"rf", "param1"}:
                rename[column] = "RF"
            elif lower in {"pw", "param2"}:
                rename[column] = "PW"
            elif lower in {"pri", "param3"}:
                rename[column] = "PRI"
            elif lower in {"pa", "param4"}:
                rename[column] = "PA"
            elif lower in {"doa", "param5"}:
                rename[column] = "DOA"
        data = table.rename(columns=rename).copy()
        if "TOA" not in data.columns:
            raise ValueError("HDBSCAN第一阶段结果缺少 TOA/TOA(s) 字段")
        if "Track_ID" not in data.columns:
            raise ValueError("HDBSCAN第一阶段结果缺少 SigIdx/Track_ID 字段")

        for name in ["TOA", "Track_ID", "RF", "PW", "PRI", "PA", "DOA"]:
            if name in data.columns:
                data[name] = pd.to_numeric(data[name], errors="coerce")
        data = data.dropna(subset=["TOA", "Track_ID"]).sort_values("TOA").reset_index(drop=True)
        if data.empty:
            raise ValueError("HDBSCAN第一阶段结果没有可用数据")
        if "Pulse_ID" not in data.columns:
            data.insert(0, "Pulse_ID", range(1, len(data) + 1))
        if "RF" not in data.columns:
            data["RF"] = 0.0
        if "PW" not in data.columns:
            data["PW"] = 0.0
        if "PA" not in data.columns:
            data["PA"] = 0.0
        if "DOA" not in data.columns:
            data["DOA"] = float("nan")
        if "PRI" not in data.columns:
            pri = data["TOA"].diff().fillna(data["TOA"].diff().median())
            data["PRI"] = pri.replace([float("inf"), float("-inf")], 0.0).fillna(0.0).clip(lower=0)

        data.attrs["source_path"] = os.path.abspath(path)
        truth_path = os.path.join(os.path.dirname(os.path.abspath(path)), "Sorted_PDW.txt")
        if os.path.exists(truth_path):
            data.attrs["truth_path"] = truth_path

        return LoadedData(path=path, filename=os.path.basename(path), data=data, fields=list(data.columns))

    def _on_sorting_result_loaded(self, loaded, elapsed: float):
        data = loaded.data.copy()
        if "Track_ID" not in data.columns:
            if "SigIdx" in data.columns:
                data["Track_ID"] = pd.to_numeric(data["SigIdx"], errors="coerce").fillna(0).astype(int)
            else:
                QMessageBox.warning(self, "导入失败", "分选结果文件必须包含 Track_ID 或 SigIdx 字段。")
                self._finish_status("分选结果导入", elapsed)
                return

        tracks = pd.to_numeric(data["Track_ID"], errors="coerce").fillna(0).astype(int)
        data["Track_ID"] = tracks
        data["Assigned"] = tracks > 0
        if self.sorting_import_target_page == "hdbscan":
            data["Sorting_Method"] = "HDBSCAN"
        elif "Sorting_Method" not in data.columns:
            data["Sorting_Method"] = "导入分选结果"
        if self.sorting_import_target_page == "hdbscan":
            data["HDBSCAN_Track_ID"] = data["Track_ID"]
            data["HDBSCAN_Assigned"] = data["Assigned"]
            data["HDBSCAN_Sorting_Method"] = "HDBSCAN"

        assigned = int(data["Assigned"].sum())
        unassigned = int(len(data) - assigned)
        track_count = int(data.loc[data["Track_ID"] > 0, "Track_ID"].nunique())
        output = SortingOutput(
            data=data,
            method="HDBSCAN" if self.sorting_import_target_page == "hdbscan" else "导入分选结果",
            elapsed=elapsed,
            track_count=track_count,
            assigned_count=assigned,
            unassigned_count=unassigned,
            summary={
                "分选轨迹数": track_count,
                "已分配脉冲数": assigned,
                "未分配脉冲数": unassigned,
                "平均 PRI": float(data["PRI"].mean()) if "PRI" in data else 0.0,
                "平均脉宽": float(data["PW"].mean()) if "PW" in data else 0.0,
            },
        )
        self.sorting_output = output
        self.sorting_pipeline_output = None
        self.recognition_output = None
        self.pipeline_output = None
        self.data = output.data
        self.loaded_file = loaded.filename
        self.status_data.setText(f"数据：{self.loaded_file}")
        self._refresh_pages()
        self.tabs.setCurrentWidget(self.recognition_page if self.sorting_import_target_page == "recognition" else self.sort_page)
        self._finish_status("分选结果导入", elapsed)
        source_text = "HDBSCAN第一阶段结果" if self.sorting_import_target_page == "hdbscan" else "分选结果"
        self.log(
            f"{source_text}导入完成：{loaded.filename}，"
            f"轨迹数：{track_count}，已分配：{assigned:,}，未分配：{unassigned:,}"
        )

    def import_zeng_training_data(self):
        train_dir = QFileDialog.getExistingDirectory(self, "选择 zeng 训练数据目录（包含 Class_*.txt）")
        if not train_dir:
            return
        self.tabs.setCurrentWidget(self.recognition_page)
        self.log(f"zeng 模板库生成开始：训练数据目录 {train_dir}")
        self._run_background(
            run_zeng_template_training,
            train_dir,
            done=self._on_zeng_template_training_done,
            stages=("检查训练数据", "生成 zeng 模板库", "更新模板库文件"),
            pass_progress=True,
            animate_progress=False,
        )

    def _on_zeng_template_training_done(self, result: Dict[str, object], elapsed: float):
        self._finish_status("zeng 模板库生成", elapsed)
        self.log(
            "zeng 模板库生成完成："
            f"类别文件 {result.get('class_count')} 个，"
            f"模板库 {result.get('template_library')}，"
            f"调参文件 {result.get('tuned_parameters')}，"
            f"耗时 {format_elapsed(elapsed)}"
        )

    def start_sorting(self):
        if not self._ensure_data():
            return
        if self.sorting_output is not None and str(self.sorting_output.method).lower() == "hdbscan":
            self.log("从导入的 HDBSCAN 第一阶段结果开始调试：仅运行 cycle_period")
            self._run_background(
                lambda df, progress_callback=None, should_cancel=None: run_cycle_period_pipeline_from_hdbscan(
                    df,
                    progress_callback=progress_callback,
                    should_cancel=should_cancel,
                ),
                self.data.copy(),
                done=self._on_sorting_pipeline_done,
                stages=("准备HDBSCAN导入结果", "cycle_period分选流程运行中", "刷新cycle_period分选结果"),
                pass_progress=True,
                animate_progress=False,
            )
            return
        self.log("两阶段分选开始：HDBSCAN → cycle_period")
        self._run_background(
            lambda df, progress_callback=None, should_cancel=None, stream_callback=None: run_sorting_pipeline(
                df,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
                stream_callback=stream_callback,
            ),
            self.data.copy(),
            done=self._on_sorting_pipeline_done,
            stages=("准备分选数据", "两阶段分选流程运行中", "刷新两阶段分选结果"),
            pass_progress=True,
            animate_progress=False,
            stream_updates=True,
        )

    def start_home_pipeline(self):
        self.pipeline_result_target_page = "recognition"
        self.start_pipeline()

    def start_recognition(self):
        self.pipeline_result_target_page = "recognition"
        if self._can_recognize_current_sorting_result():
            self.start_recognition_from_current_sorting()
            return
        if self.sorting_output is not None and str(self.sorting_output.method).lower() == "hdbscan":
            self.start_pipeline_from_imported_hdbscan()
            return
        if self.sorting_output is not None:
            self.log("当前分选结果不是 cycle_period，按完整流程重新生成 cycle_period 分选结果后进行 zeng 识别")
        self.start_pipeline()

    def _can_recognize_current_sorting_result(self) -> bool:
        if self.data is None or self.sorting_output is None or "Track_ID" not in self.data.columns:
            return False
        method = str(self.sorting_output.method).lower()
        return method == "cycle_period" or self.sorting_output.method == "导入分选结果"

    def start_recognition_from_current_sorting(self):
        model = self.recognition_page.method_panel.selected_recognition_method()
        if model.lower() == "zeng" and not self._ensure_zeng_template_ready():
            return
        self.log(f"使用当前分选结果进行识别：模型 {model}")
        self._run_background(
            lambda df, m, progress_callback=None, should_cancel=None: run_recognition(
                df,
                m,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            ),
            self.data.copy(),
            model,
            done=self._on_recognition_done,
            stages=("准备识别数据", "识别算法运行中", "刷新识别结果"),
            pass_progress=model.lower() == "zeng",
            animate_progress=model.lower() != "zeng",
        )

    def start_pipeline(self):
        if not self._ensure_data():
            return
        if not self._ensure_zeng_template_ready():
            return
        self.log(f"完整流程开始：{pipeline_label(FULL_PIPELINE_ID)}")
        self._run_background(
            lambda df, p, progress_callback=None, should_cancel=None, stream_callback=None: run_pipeline(
                df,
                p,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
                stream_callback=stream_callback,
            ),
            self.data.copy(),
            FULL_PIPELINE_ID,
            done=self._on_pipeline_done,
            stages=("准备流程数据", "分选与识别流程运行中", "刷新分选与识别结果"),
            pass_progress=True,
            animate_progress=False,
            stream_updates=True,
        )

    def start_pipeline_from_imported_hdbscan(self):
        if not self._ensure_data():
            return
        if not self._ensure_zeng_template_ready():
            return
        self.log("使用导入的 HDBSCAN 第一阶段结果运行 cycle_period 后进行 zeng 识别")
        self._run_background(
            lambda df, progress_callback=None, should_cancel=None: run_pipeline_from_hdbscan(
                df,
                progress_callback=progress_callback,
                should_cancel=should_cancel,
            ),
            self.data.copy(),
            done=self._on_pipeline_done,
            stages=("准备HDBSCAN导入结果", "cycle_period与识别流程运行中", "刷新分选与识别结果"),
            pass_progress=True,
            animate_progress=False,
        )

    def _selected_pipeline_id(self) -> str:
        page = self.tabs.currentWidget()
        if isinstance(page, AnalysisPage):
            return page.method_panel.selected_pipeline_id()
        return self.recognition_page.method_panel.selected_pipeline_id()

    def _ensure_zeng_template_ready(self) -> bool:
        if zeng_template_library_exists():
            return True
        path = zeng_template_library_path()
        message = (
            "zeng 识别需要模板库。\n\n"
            "请先在“信号识别”页点击“生成zeng模板库”，"
            f"生成后再运行。\n\n缺少文件：{path}"
        )
        QMessageBox.information(self, "需要生成模板库", message)
        self.log(f"运行已暂停：zeng 模板库未生成，缺少 {path}")
        self.tabs.setCurrentWidget(self.recognition_page)
        return False

    def _run_background(self, func, *args, done, stages=None, pass_progress=False, animate_progress=True, stream_updates=False):
        self._clear_partial_sorting_update()
        self._set_all_running(True)
        self.progress.setValue(5)
        self.step_label.setText("当前操作：任务启动")
        self.task_progress_cap = 90
        if animate_progress:
            self.task_progress_timer.start(650)
        else:
            self.task_progress_timer.stop()
        self.worker_thread = QThread()
        self.worker = Worker(func, *args, stages=stages, pass_progress=pass_progress, pass_stream=stream_updates)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_worker_progress)
        if stream_updates:
            self.worker.partial.connect(self._on_worker_partial_sorting)
        self.worker.finished.connect(done)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(lambda: self._set_all_running(False))
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _on_worker_progress(self, value: int, text: str):
        if value >= 100:
            self.task_progress_timer.stop()
        self.progress.setValue(max(self.progress.value(), value))
        self.step_label.setText(f"当前操作：{text}")

    def _on_worker_partial_sorting(self, data: pd.DataFrame):
        if data is None or data.empty:
            return
        self.pending_partial_sorting_data = data
        if not self.partial_sorting_timer.isActive():
            self.partial_sorting_timer.start()

    def _flush_partial_sorting_update(self):
        data = self.pending_partial_sorting_data
        self.pending_partial_sorting_data = None
        if data is None or data.empty:
            return
        if not self.partial_sorting_started:
            self.sort_page.clear_stage_results()
            self.tabs.setCurrentWidget(self.sort_page)
            self.partial_sorting_started = True
        self.sort_page.set_data(data)
        self.sort_page.method_panel.update_summary(self._summary_for_dataframe(data))

    def _clear_partial_sorting_update(self):
        if hasattr(self, "partial_sorting_timer"):
            self.partial_sorting_timer.stop()
        self.pending_partial_sorting_data = None
        self.partial_sorting_started = False

    def _advance_task_progress(self):
        value = self.progress.value()
        if value < 15:
            self.progress.setValue(value + 1)
        elif value < 55:
            self.progress.setValue(min(55, value + 2))
        elif value < self.task_progress_cap:
            self.progress.setValue(value + 1)

    def _on_pipeline_done(self, output: PipelineRunResult, elapsed: float):
        self._clear_partial_sorting_update()
        self.pipeline_output = output
        self.sorting_pipeline_output = None
        self.sorting_output = output.sorting
        self.recognition_output = output.recognition
        self.data = output.recognition.data
        self._autosave_sorting_results(output.sorting, output.stage_results)
        self._refresh_pages()
        if self.pipeline_result_target_page == "sorting":
            self.tabs.setCurrentWidget(self.sort_page)
        else:
            self.tabs.setCurrentWidget(self.recognition_page)
        self._finish_status(output.pipeline_name, elapsed)
        stage_text = f"，阶段数：{len(output.stage_results)}" if len(output.stage_results) > 1 else ""
        self.log(
            f"完整流程完成：{output.pipeline_name}{stage_text}，"
            f"分选轨迹数：{output.sorting.track_count}，"
            f"平均置信度：{output.recognition.mean_confidence * 100:.1f}%"
        )

    def _on_sorting_pipeline_done(self, output: SortingPipelineResult, elapsed: float):
        self._clear_partial_sorting_update()
        self.sorting_pipeline_output = output
        self.pipeline_output = None
        self.sorting_output = output.sorting
        self.recognition_output = None
        self.data = output.sorting.data
        self._autosave_sorting_results(output.sorting, output.stage_results)
        self._refresh_pages()
        self.tabs.setCurrentWidget(self.sort_page)
        self._finish_status(output.pipeline_name, elapsed)
        stages = "、".join(stage.definition.name for stage in output.stage_results)
        self.log(
            f"两阶段分选完成：{stages}，"
            f"最终方法 {output.sorting.method}，轨迹数：{output.sorting.track_count}，"
            f"耗时：{format_elapsed(elapsed)}"
        )

    def _on_sorting_done(self, output: SortingOutput, elapsed: float):
        self._clear_partial_sorting_update()
        self.sorting_output = output
        self.sorting_pipeline_output = None
        self.recognition_output = None
        self.pipeline_output = None
        self.data = output.data
        self._refresh_pages()
        self.tabs.setCurrentWidget(self.sort_page)
        self._finish_status("分选分析", elapsed)
        self.log(f"分选完成：方法 {output.method}，轨迹数：{output.track_count}，耗时：{format_elapsed(elapsed)}")

    def _on_recognition_done(self, output: RecognitionOutput, elapsed: float):
        self.recognition_output = output
        self.data = output.data
        self._refresh_pages()
        self.tabs.setCurrentWidget(self.recognition_page)
        self._finish_status("信号识别", elapsed)
        self.log(f"识别完成：模型 {output.model}，平均置信度：{output.mean_confidence * 100:.1f}%")

    def _on_worker_failed(self, message: str):
        self._clear_partial_sorting_update()
        self.task_progress_timer.stop()
        self.progress.setValue(0)
        self.step_label.setText("当前操作：任务失败")
        self.log(f"任务失败：{message}")
        QMessageBox.warning(self, "任务失败", message)

    def stop_current_task(self):
        self._clear_partial_sorting_update()
        if self.worker is not None:
            self.worker.cancel()
        self.task_progress_timer.stop()
        self.progress.setValue(0)
        self.step_label.setText("当前操作：已停止")
        self.elapsed_label.setText("本次耗时：00:00:00.000")
        self.log("任务停止请求已发送")

    def _refresh_pages(self):
        self.home_page.set_data(self.data)
        stage_results = None
        if self.pipeline_output is not None:
            stage_results = self.pipeline_output.stage_results
        elif self.sorting_pipeline_output is not None:
            stage_results = self.sorting_pipeline_output.stage_results

        if stage_results:
            self.sort_page.set_stage_results(stage_results)
        else:
            self.sort_page.clear_stage_results()
            self.sort_page.set_data(self.data)
        track_results = self.recognition_output.track_results if self.recognition_output else None
        self.recognition_page.set_data(self.data, track_results)
        self._update_summary()

    def _update_summary(self):
        final_summary = self._summary_for_dataframe(self.data)
        sort_summary = self._summary_for_dataframe(self.sort_page.data)
        recognition_summary = dict(final_summary)
        if self.recognition_output is not None:
            recognition_summary.update(
                {
                    "识别轨迹数": len(self.recognition_output.track_results),
                    "平均置信度": f"{self.recognition_output.mean_confidence * 100:.1f}%",
                    "类别数": self.recognition_output.class_count,
                }
            )
            recognition_accuracy = self.recognition_output.data.attrs.get(
                "recognition_accuracy",
                self.recognition_output.track_results.attrs.get("recognition_accuracy"),
            )
            if recognition_accuracy is not None:
                recognition_summary["识别准确率"] = f"{float(recognition_accuracy) * 100:.1f}%"
        self.home_page.method_panel.update_summary(final_summary)
        self.sort_page.method_panel.update_summary(sort_summary)
        self.recognition_page.method_panel.update_summary(recognition_summary)
        self.export_method_panel.update_summary(recognition_summary)

    def _summary_for_dataframe(self, df: Optional[pd.DataFrame]) -> Dict[str, object]:
        if df is None:
            return {}
        tracks = df["Track_ID"] if "Track_ID" in df.columns else pd.Series([0] * len(df))
        tracks = pd.to_numeric(tracks, errors="coerce").fillna(0).astype(int)
        assigned = int((tracks > 0).sum())
        unassigned = int(len(df) - assigned)
        return {
            "分选轨迹数": int(tracks[tracks > 0].nunique()),
            "已分配脉冲数": f"{assigned:,}",
            "未分配脉冲数": f"{unassigned:,}",
            "平均 PRI": f"{df['PRI'].mean():.3f}" if "PRI" in df else "-",
            "平均脉宽": f"{df['PW'].mean():.3f}" if "PW" in df else "-",
        }

    def show_method_selector(self):
        page = self.tabs.currentWidget()
        if isinstance(page, AnalysisPage):
            if page is self.recognition_page:
                page.method_panel.show_recognition_selector()
            else:
                page.method_panel.show_sorting_selector()

    def toggle_compare(self):
        self.compare_enabled = not self.compare_enabled
        self.log(f"对比显示已{'开启' if self.compare_enabled else '关闭'}")

    def export_sorting_file(self):
        if not self._ensure_data():
            return
        data = self.sort_page.data if self.sort_page.data is not None else self.data
        path, _ = QFileDialog.getSaveFileName(self, "导出分选结果", "sorting_result.csv", "CSV Files (*.csv)")
        if path:
            export_sorting_csv(data, path)
            self.log(f"导出分选结果：{path}")

    def export_sorting_file(self):
        if not self._ensure_data():
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出分选结果", "sorting_result.csv", "CSV Files (*.csv)")
        if path:
            data = self._final_sorting_dataframe()
            export_sorting_csv(data, path)
            exported_stages = self._export_sorting_stage_files(path)
            stage_text = f"，阶段结果 {exported_stages} 个" if exported_stages else ""
            self.log(f"导出分选最终结果：{path}{stage_text}")

    def _final_sorting_dataframe(self) -> pd.DataFrame:
        if self.sorting_pipeline_output is not None:
            return self.sorting_pipeline_output.sorting.data
        if self.pipeline_output is not None:
            return self.pipeline_output.sorting.data
        if self.sorting_output is not None:
            return self.sorting_output.data
        return self.data

    def _export_sorting_stage_files(self, final_path: str) -> int:
        if self.sorting_pipeline_output is not None:
            stages = self.sorting_pipeline_output.stage_results
        elif self.pipeline_output is not None:
            stages = self.pipeline_output.stage_results
        else:
            stages = []
        if not stages:
            return 0
        directory = os.path.dirname(os.path.abspath(final_path))
        stem, ext = os.path.splitext(os.path.basename(final_path))
        ext = ext or ".csv"
        exported = 0
        for index, stage in enumerate(stages, start=1):
            name = str(stage.definition.name).replace("+", "_").replace(" ", "_")
            stage_path = os.path.join(directory, f"{stem}_stage{index}_{name}{ext}")
            export_sorting_csv(stage.sorting.data, stage_path)
            exported += 1
        return exported

    def _autosave_sorting_results(self, final_sorting: SortingOutput, stages) -> Optional[str]:
        data = final_sorting.data
        directory = self._sorting_run_directory(data)
        if directory is None:
            return None
        os.makedirs(directory, exist_ok=True)
        export_sorting_csv(data, os.path.join(directory, "sorting_final_result.csv"))
        for index, stage in enumerate(stages or [], start=1):
            name = str(stage.definition.name).replace("+", "_").replace(" ", "_")
            stage_path = os.path.join(directory, f"sorting_stage{index}_{name}.csv")
            export_sorting_csv(stage.sorting.data, stage_path)
        return directory

    def _sorting_run_directory(self, data: pd.DataFrame) -> Optional[str]:
        for column in ("CyclePeriod_Run_Dir", "HDBSCAN_Run_Dir"):
            if column not in data.columns:
                continue
            values = [str(value) for value in data[column].dropna().unique() if str(value).strip()]
            if values:
                return values[0]
        return None

    def export_recognition_file(self):
        if self.recognition_output is None:
            QMessageBox.information(self, "提示", "请先完成信号识别。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出识别结果", "recognition_result.csv", "CSV Files (*.csv)")
        if path:
            export_recognition_csv(self.recognition_output.track_results, path)
            self.log(f"导出识别结果：{path}")

    def export_full_result_file(self):
        if not self._ensure_data():
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出完整结果", "full_result.csv", "CSV Files (*.csv)")
        if path:
            export_full_csv(self.data, path)
            self.log(f"导出完整结果：{path}")

    def export_current_charts(self):
        page = self.tabs.currentWidget()
        if not isinstance(page, AnalysisPage):
            page = self.home_page
        directory = QFileDialog.getExistingDirectory(self, "选择图表导出目录")
        if not directory:
            return
        for index, (card, _) in enumerate(page.chart_cards, start=1):
            filename = f"{index:02d}_{card.title.text().replace('/', '_')}.svg"
            card.figure.savefig(os.path.join(directory, filename), format="svg", bbox_inches="tight")
        self.log(f"导出图表完成：{directory}")

    def export_report_file(self):
        if not self._ensure_data():
            return
        path, _ = QFileDialog.getSaveFileName(self, "生成总结报告", "summary_report.txt", "Text/HTML Files (*.txt *.html)")
        if path:
            results = self.recognition_output.track_results if self.recognition_output else pd.DataFrame()
            export_report(self.data, results, self.log_lines, path)
            self.log(f"生成总结报告：{path}")

    def export_log_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出运行日志", "run_log.txt", "Text Files (*.txt)")
        if path:
            export_log(self.log_lines, path)
            self.log(f"导出运行日志：{path}")

    def _ensure_data(self) -> bool:
        if self.data is not None:
            return True
        QMessageBox.information(self, "提示", "请先导入 Merge 数据文件。")
        self.log("操作取消：未导入 Merge 数据")
        return False

    def _set_all_running(self, running: bool):
        for ribbon in [self.home_ribbon, self.sort_ribbon, self.recognition_ribbon, self.export_ribbon]:
            ribbon.set_running(running)

    def _finish_status(self, operation: str, elapsed: float):
        self.task_progress_timer.stop()
        self.progress.setValue(100)
        self.step_label.setText(f"当前操作：{operation}完成")
        self.elapsed_label.setText(f"本次耗时：{format_elapsed(elapsed)}")

    def log(self, text: str):
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {text}"
        self.log_lines.append(line)
        self.log_edit.append(line)

    def _update_status_time(self):
        self.status_time.setText(f"时间：{QTimer().remainingTime() if False else datetime.now():%Y-%m-%d %H:%M:%S}")


def load_stylesheet(app: QApplication):
    style_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "style.qss")
    if os.path.exists(style_path):
        with open(style_path, "r", encoding="utf-8") as file:
            app.setStyleSheet(file.read())

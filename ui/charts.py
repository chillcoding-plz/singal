from typing import Dict, Optional
import pickle

import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QFrame, QHBoxLayout, QLabel, QSizePolicy, QTableWidget, QTableWidgetItem, QVBoxLayout

from .widgets import TRACK_COLORS, track_color


rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

TRACK_BAR_PREVIEW_LIMIT = 12


class InteractiveFigureDialog(QDialog):
    def __init__(self, source_figure: Figure, title: str, parent=None, render_callback=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1180, 760)
        self._drag_origin = None
        if render_callback is None:
            self.figure = pickle.loads(pickle.dumps(source_figure))
            self.figure.set_size_inches(10, 6, forward=False)
            self.figure.set_dpi(100)
        else:
            self.figure = Figure(figsize=(10, 6), dpi=100, facecolor="white")
            render_callback(self.figure)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _on_scroll(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        ax = event.inaxes
        scale = 1 / 1.18 if event.button == "up" else 1.18
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_span = (x_max - x_min) * scale
        y_span = (y_max - y_min) * scale
        rel_x = (x_max - event.xdata) / max(x_max - x_min, 1e-12)
        rel_y = (y_max - event.ydata) / max(y_max - y_min, 1e-12)
        ax.set_xlim(event.xdata - x_span * (1 - rel_x), event.xdata + x_span * rel_x)
        ax.set_ylim(event.ydata - y_span * (1 - rel_y), event.ydata + y_span * rel_y)
        self.canvas.draw_idle()

    def _on_press(self, event):
        if self.toolbar.mode or event.button != 1 or event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        self._drag_origin = {
            "axes": event.inaxes,
            "x": event.xdata,
            "y": event.ydata,
            "xlim": event.inaxes.get_xlim(),
            "ylim": event.inaxes.get_ylim(),
        }

    def _on_motion(self, event):
        if self._drag_origin is None or event.inaxes is not self._drag_origin["axes"] or event.xdata is None or event.ydata is None:
            return
        ax = self._drag_origin["axes"]
        dx = event.xdata - self._drag_origin["x"]
        dy = event.ydata - self._drag_origin["y"]
        x_min, x_max = self._drag_origin["xlim"]
        y_min, y_max = self._drag_origin["ylim"]
        ax.set_xlim(x_min - dx, x_max - dx)
        ax.set_ylim(y_min - dy, y_max - dy)
        self.canvas.draw_idle()

    def _on_release(self, _event):
        self._drag_origin = None


class ChartCard(QFrame):
    def __init__(self, title, parent=None, compact=False):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.compact = compact
        self.figure = Figure(figsize=(4, 3), dpi=100, facecolor="white")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111)
        self._dialog = None
        self._detail_renderer = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame()
        header.setProperty("class", "chartHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 7, 10, 7)
        self.title = QLabel(title)
        self.title.setProperty("class", "chartTitle")
        header_layout.addWidget(self.title)
        header_layout.addStretch(1)
        layout.addWidget(header)
        layout.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._open_dialog)

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("white")

    def _open_dialog(self, event):
        if event.button != 1 or not event.dblclick or event.inaxes is None:
            return
        self._dialog = InteractiveFigureDialog(self.figure, self.title.text(), self, self._detail_renderer)
        self._dialog.show()

    def set_detail_renderer(self, renderer):
        self._detail_renderer = renderer

    def show_empty(self):
        self.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.5, 0.5, "暂无数据", ha="center", va="center", color="#94a3b8", transform=self.ax.transAxes)
        self.canvas.draw_idle()

    def finish(self, grid=True, legend=True):
        if grid:
            self.ax.grid(True, linestyle="--", linewidth=0.55, color="#d8e1ec", alpha=0.9)
        for spine in self.ax.spines.values():
            spine.set_color("#aebdcc")
        self.ax.tick_params(colors="#334155", labelsize=8)
        if legend:
            self.ax.legend(frameon=True, fontsize=7 if self.compact else 8, loc="upper left")
        if self.compact:
            self.figure.subplots_adjust(left=0.10, right=0.985, top=0.965, bottom=0.16)
        else:
            self.figure.tight_layout(pad=1.0)
        self.canvas.draw_idle()


class TableCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame()
        header.setProperty("class", "chartHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 7, 10, 7)
        label = QLabel(title)
        label.setProperty("class", "chartTitle")
        header_layout.addWidget(label)
        header_layout.addStretch(1)
        layout.addWidget(header)
        self.table = QTableWidget(0, 7)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

    def set_dataframe(self, df: Optional[pd.DataFrame]):
        if df is None or df.empty:
            self.table.setRowCount(0)
            return
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        self.table.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            for col_idx, value in enumerate(row):
                item = QTableWidgetItem(f"{value:.4g}" if isinstance(value, float) else str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_idx, col_idx, item)
        self.table.resizeColumnsToContents()


def _sample(df: pd.DataFrame, max_points=2500) -> pd.DataFrame:
    if df is None or len(df) <= max_points:
        return df
    return df.iloc[np.linspace(0, len(df) - 1, max_points).astype(int)]


def _visible(df: pd.DataFrame, visibility: Dict[int, bool]) -> pd.DataFrame:
    track_column = _track_column(df)
    if df is None or track_column not in df.columns:
        return df
    hidden = [int(track_id) for track_id, visible in (visibility or {}).items() if not visible]
    if not hidden:
        return df
    tracks = pd.to_numeric(df[track_column], errors="coerce").fillna(0).astype(int)
    return df[~tracks.isin(hidden)]


def _track_column(df: Optional[pd.DataFrame]) -> str:
    if df is not None and "Display_Track_ID" in df.columns:
        return "Display_Track_ID"
    return "Track_ID"


def _plot_track_scatter(card: ChartCard, data: pd.DataFrame, x_column: str, y_column: str, y_label: str, show_legend: bool):
    track_column = _track_column(data)
    if track_column not in data:
        card.ax.scatter(data[x_column], data[y_column], s=10, color="#1E88E5", alpha=0.72, label="脉冲")
    else:
        tracks = pd.to_numeric(data[track_column], errors="coerce").fillna(0).astype(int)
        colors = [track_color(track_id) for track_id in tracks]
        card.ax.scatter(data[x_column], data[y_column], s=10, c=colors, alpha=0.72, linewidths=0)
        if show_legend:
            legend_tracks = tracks.value_counts().head(8).index.tolist()
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=track_color(track_id),
                    markersize=5,
                    label="未分配" if int(track_id) == 0 else f"轨迹 {int(track_id)}",
                )
                for track_id in legend_tracks
            ]
            card.ax.legend(handles=handles, frameon=True, fontsize=8, loc="best")
    card.ax.set_xlabel(x_column)
    card.ax.set_ylabel(y_label)


def _plot_full_track_scatter(ax, data: pd.DataFrame, x_column: str, y_column: str, y_label: str):
    track_column = _track_column(data)
    if track_column not in data:
        ax.scatter(data[x_column], data[y_column], s=4, color="#1E88E5", alpha=0.58, linewidths=0, label="脉冲")
        ax.legend(frameon=True, fontsize=8, loc="best")
    else:
        tracks = pd.to_numeric(data[track_column], errors="coerce").fillna(0).astype(int)
        colors = [track_color(track_id) for track_id in tracks]
        ax.scatter(data[x_column], data[y_column], s=4, c=colors, alpha=0.58, linewidths=0)
        legend_tracks = sorted(tracks.unique().tolist())
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=track_color(track_id),
                markersize=5,
                label="未分配" if int(track_id) == 0 else f"轨迹 {int(track_id)}",
            )
            for track_id in legend_tracks
        ]
        if handles:
            ncol = max(1, min(4, int(np.ceil(len(handles) / 35))))
            ax.legend(
                handles=handles,
                frameon=True,
                fontsize=7,
                title="轨迹",
                title_fontsize=8,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                ncol=ncol,
                borderaxespad=0.0,
            )
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_label)


def _finish_full_figure(figure: Figure, ax, grid=True):
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.55, color="#d8e1ec", alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#aebdcc")
    ax.tick_params(colors="#334155", labelsize=9)
    figure.tight_layout(pad=1.0, rect=(0, 0, 0.82, 1))


def plot_timeline(card: ChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    card.clear()
    data = _sample(df, 1200)
    y = data["PA"] if "PA" in data else np.ones(len(data))
    y = (y - y.min()) / (y.max() - y.min() + 1e-9)
    card.ax.vlines(data["TOA"], 0, y, color="#1E88E5", alpha=0.72, linewidth=0.8, label="脉冲")
    card.ax.plot(data["TOA"], y, color="#0052B5", linewidth=0.9, alpha=0.65, label="幅度趋势")
    card.ax.set_xlabel("TOA")
    card.ax.set_ylabel("归一化幅度")
    card.finish(options.get("显示网格", True), options.get("显示图例", True))


def plot_sort_scatter(card: ChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    card.clear()
    data = _sample(_visible(df, visibility or {}), 1200)
    _plot_track_scatter(card, data, "TOA", "PRI", "PRI", options.get("鏄剧ず鍥句緥", True))
    card.finish(options.get("鏄剧ず缃戞牸", True), False)
    return
    track_column = _track_column(data)
    if track_column not in data:
        card.ax.scatter(data["TOA"], data["PRI"], s=12, color="#1E88E5", alpha=0.75, label="原始脉冲")
    else:
        for track_id, group in data.groupby(track_column):
            label = "未分配" if int(track_id) == 0 else f"轨迹 {int(track_id)}"
            color = track_color(track_id)
            card.ax.scatter(group["TOA"], group["PRI"], s=13, color=color, alpha=0.75, label=label)
    card.ax.set_xlabel("TOA")
    card.ax.set_ylabel("PRI")
    card.finish(options.get("显示网格", True), options.get("显示图例", True))


def plot_rf_scatter(card: ChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    card.clear()
    data = _sample(_visible(df, visibility or {}), 1200)
    _plot_track_scatter(card, data, "TOA", "RF", "RF", options.get("鏄剧ず鍥句緥", True))
    card.finish(options.get("鏄剧ず缃戞牸", True), False)
    return
    track_column = _track_column(data)
    if track_column not in data:
        card.ax.scatter(data["TOA"], data["RF"], s=12, color="#1E88E5", alpha=0.75, label="原始脉冲")
    else:
        for track_id, group in data.groupby(track_column):
            label = "未分配" if int(track_id) == 0 else f"轨迹 {int(track_id)}"
            color = track_color(track_id)
            card.ax.scatter(group["TOA"], group["RF"], s=13, color=color, alpha=0.75, label=label)
    card.ax.set_xlabel("TOA")
    card.ax.set_ylabel("RF")
    card.finish(options.get("显示网格", True), options.get("显示图例", True))


def _plot_toa_parameter_scatter(
    card: ChartCard,
    df: pd.DataFrame,
    options: Dict[str, bool],
    visibility,
    y_column: str,
    y_label: str,
):
    card.clear()
    if y_column not in df:
        card.show_empty()
        return
    data = _sample(_visible(df, visibility or {}), 1200)
    _plot_track_scatter(card, data, "TOA", y_column, y_label, options.get("鏄剧ず鍥句緥", True))
    card.finish(options.get("鏄剧ず缃戞牸", True), False)
    return
    track_column = _track_column(data)
    if track_column not in data:
        card.ax.scatter(data["TOA"], data[y_column], s=12, color="#1E88E5", alpha=0.75, label="原始脉冲")
    else:
        for track_id, group in data.groupby(track_column):
            label = "未分配" if int(track_id) == 0 else f"轨迹 {int(track_id)}"
            color = track_color(track_id)
            card.ax.scatter(group["TOA"], group[y_column], s=13, color=color, alpha=0.75, label=label)
    card.ax.set_xlabel("TOA")
    card.ax.set_ylabel(y_label)
    card.finish(options.get("显示网格", True), options.get("显示图例", True))


def plot_pw_scatter(card: ChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    _plot_toa_parameter_scatter(card, df, options, visibility, "PW", "脉宽 PW")


def plot_pa_scatter(card: ChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    _plot_toa_parameter_scatter(card, df, options, visibility, "PA", "脉幅 PA")


def _track_counts(df: Optional[pd.DataFrame], visibility=None) -> pd.Series:
    track_column = _track_column(df)
    if df is None or track_column not in df:
        return pd.Series(dtype=int)
    visible_df = _visible(df, visibility or {})
    tracks = pd.to_numeric(visible_df[track_column], errors="coerce").fillna(0).astype(int)
    return tracks.value_counts().sort_values(ascending=False)


def plot_track_bars(card: ChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    card.clear()
    counts = _track_counts(df, visibility)
    if counts.empty:
        card.show_empty()
        return
    preview = counts.head(TRACK_BAR_PREVIEW_LIMIT).copy()
    labels = ["未分配" if int(i) == 0 else f"轨迹{int(i)}" for i in preview.index]
    values = [int(value) for value in preview.values]
    colors = [track_color(track_id) for track_id in preview.index]
    bars = card.ax.bar(labels, values, color=colors)
    card.ax.bar_label(bars, fontsize=8, padding=2)
    card.ax.set_ylabel("脉冲数")
    card.ax.set_title(f"Top {len(preview)} / 共 {len(counts)} 项", fontsize=9)
    card.ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    card.finish(options.get("显示网格", True), False)


def _plot_full_track_bars(figure: Figure, ax, df: Optional[pd.DataFrame], visibility=None):
    counts = _track_counts(df, visibility)
    if counts.empty:
        ax.text(0.5, 0.5, "暂无轨迹脉冲统计", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    labels = ["未分配" if int(i) == 0 else f"轨迹{int(i)}" for i in counts.index]
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=[track_color(track_id) for track_id in counts.index])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7 if len(labels) <= 80 else 5)
    ax.invert_yaxis()
    ax.set_xlabel("脉冲数")
    ax.set_ylabel("轨迹")
    ax.set_title(f"全部轨迹脉冲统计（{len(labels)}项）")
    figure.set_size_inches(12, max(6, min(36, 0.18 * len(labels))), forward=False)


def _truth_column(df: pd.DataFrame) -> Optional[str]:
    for column in (
        "Original_Track_ID",
        "True_Track_ID",
    ):
        if column in df.columns:
            return column
    return None


def _cycle_period_prediction_column(df: pd.DataFrame) -> Optional[str]:
    if "CyclePeriod_OurPredID" not in df.columns:
        return None
    if "Sorting_Method" not in df.columns:
        return "CyclePeriod_OurPredID"
    methods = df["Sorting_Method"].dropna().astype(str).str.lower()
    return "CyclePeriod_OurPredID" if methods.str.contains("cycle").any() else None


def _majority_match_accuracy(df: pd.DataFrame) -> Optional[float]:
    truth_column = _truth_column(df)
    prediction_column = _cycle_period_prediction_column(df) or "Track_ID"
    if truth_column is None or prediction_column not in df.columns:
        return None
    data = df[[prediction_column, truth_column]].copy()
    data[prediction_column] = pd.to_numeric(data[prediction_column], errors="coerce").fillna(0).astype(int)
    data = data.dropna(subset=[truth_column])
    if prediction_column == "CyclePeriod_OurPredID" and "Track_ID" in df.columns:
        assigned = pd.to_numeric(df.loc[data.index, "Track_ID"], errors="coerce").fillna(0).astype(int) > 0
        data = data[assigned]
    else:
        data = data[data[prediction_column] > 0]
    if data.empty:
        return None
    correct = 0
    for _, group in data.groupby(prediction_column, sort=False):
        correct += int(group[truth_column].astype(str).value_counts().iloc[0])
    return correct / max(len(df.dropna(subset=[truth_column])), 1)


def plot_quality(card: ChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    card.clear()
    if "Assigned" not in df:
        card.show_empty()
        return
    assigned = int(df["Assigned"].sum())
    unassigned = int(len(df) - assigned)
    values = [assigned, unassigned]
    wedges, _ = card.ax.pie(
        values,
        colors=["#1E88E5", "#78909C"],
        startangle=90,
        center=(-0.48, 0),
        radius=0.82,
        wedgeprops={"width": 0.32, "edgecolor": "white"},
    )
    total = max(sum(values), 1)
    card.ax.text(-0.48, 0, f"{assigned / total * 100:.1f}%\n已分配", ha="center", va="center", color="#0052B5", fontweight="bold")
    accuracy = _majority_match_accuracy(df)
    accuracy_text = f"{accuracy * 100:.1f}%" if accuracy is not None else "暂无真值"
    card.ax.text(
        0.73,
        0.60,
        "准确率",
        ha="center",
        va="center",
        transform=card.ax.transAxes,
        color="#334155",
        fontsize=9,
    )
    card.ax.text(
        0.73,
        0.43,
        accuracy_text,
        ha="center",
        va="center",
        transform=card.ax.transAxes,
        color="#0F766E" if accuracy is not None else "#64748B",
        fontsize=13 if accuracy is not None else 10,
        fontweight="bold",
    )
    if options.get("显示图例", True):
        card.ax.legend(wedges, ["已分配", "未分配"], loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, frameon=False)
    card.ax.set_aspect("equal")
    card.ax.set_xlim(-1.45, 1.35)
    card.ax.set_ylim(-1.05, 1.05)
    card.figure.tight_layout(pad=0.8)
    card.canvas.draw_idle()


def plot_feature_projection(card: ChartCard, data: pd.DataFrame, options: Dict[str, bool], visibility=None):
    card.clear()
    if data is None or data.empty:
        card.show_empty()
        return
    if {"RF", "PRI"}.issubset(data.columns):
        work = _sample(_visible(data, visibility)).copy()
        x_column, y_column = "RF", "PRI"
    elif {"Mean_RF", "Mean_PRI"}.issubset(data.columns):
        work = data.copy()
        x_column, y_column = "Mean_RF", "Mean_PRI"
    else:
        card.show_empty()
        return
    labels = work.get("Predicted_Label", pd.Series(["未知"] * len(work), index=work.index)).fillna("未知")
    for idx, label in enumerate(sorted(labels.unique())):
        group = work[labels == label]
        legend_label = "未知" if str(label).lower() == "unknown" else str(label)
        card.ax.scatter(group[x_column], group[y_column], s=10, alpha=0.72, color=TRACK_COLORS[idx % len(TRACK_COLORS)], linewidths=0, label=legend_label)
    card.ax.set_xlabel("RF" if x_column == "RF" else "Mean RF")
    card.ax.set_ylabel("PRI" if y_column == "PRI" else "Mean PRI")
    card.finish(options.get("显示网格", True), options.get("显示图例", True))


def plot_probability(card: ChartCard, data: pd.DataFrame, options: Dict[str, bool], visibility=None):
    card.clear()
    if data is None or data.empty or "Confidence" not in data.columns:
        card.show_empty()
        return
    if "TOA" in data.columns:
        work = _sample(_visible(data, visibility)).copy()
        x = pd.to_numeric(work["TOA"], errors="coerce")
        confidence = pd.to_numeric(work["Confidence"], errors="coerce").fillna(0.0).clip(0, 1)
        card.ax.scatter(x, confidence, s=10, color="#1E88E5", alpha=0.72, linewidths=0, label="置信度")
        card.ax.set_xlabel("TOA")
    else:
        work = data.head(40).copy()
        labels = [f"T{int(t)}" for t in work["Track_ID"]]
        confidence = pd.to_numeric(work["Confidence"], errors="coerce").fillna(0.0).clip(0, 1).to_numpy(float)
        card.ax.bar(labels, confidence, color="#1E88E5", label="置信度")
        card.ax.bar(labels, 1 - confidence, bottom=confidence, color="#D7E3F0", label="不确定度")
    card.ax.set_ylim(0, 1)
    card.ax.set_ylabel("置信度")
    card.finish(options.get("显示网格", True), options.get("显示图例", True))


def plot_class_stats(card: ChartCard, data: pd.DataFrame, options: Dict[str, bool], visibility=None):
    card.clear()
    if data is None or data.empty or "Predicted_Label" not in data.columns:
        card.show_empty()
        return
    counts = data["Predicted_Label"].replace("", pd.NA).dropna().value_counts()
    if counts.empty:
        card.show_empty()
        return
    colors = [TRACK_COLORS[index % len(TRACK_COLORS)] for index in range(len(counts))]
    bars = card.ax.bar(counts.index.astype(str), counts.values, color=colors)
    card.ax.bar_label(bars, fontsize=8, padding=2)
    card.ax.set_ylabel("轨迹数")
    card.finish(options.get("显示网格", True), False)


def render_full_detail_figure(
    figure: Figure,
    plotter,
    data: Optional[pd.DataFrame],
    track_results: Optional[pd.DataFrame],
    options: Dict[str, bool],
    visibility=None,
):
    ax = figure.add_subplot(111)
    visibility = visibility or {}
    if plotter is plot_sort_scatter:
        full = data
        _plot_full_track_scatter(ax, full, "TOA", "PRI", "PRI")
        _finish_full_figure(figure, ax)
        return
    if plotter is plot_rf_scatter:
        full = data
        _plot_full_track_scatter(ax, full, "TOA", "RF", "RF")
        _finish_full_figure(figure, ax)
        return
    if plotter is plot_pw_scatter:
        full = data
        if full is not None and "PW" in full:
            _plot_full_track_scatter(ax, full, "TOA", "PW", "PW")
        _finish_full_figure(figure, ax)
        return
    if plotter is plot_pa_scatter:
        full = data
        if full is not None and "PA" in full:
            _plot_full_track_scatter(ax, full, "TOA", "PA", "PA")
        _finish_full_figure(figure, ax)
        return
    if plotter is plot_track_bars:
        _plot_full_track_bars(figure, ax, data, visibility)
        _finish_full_figure(figure, ax)
        return
    ax.text(0.5, 0.5, "Preview chart", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    _finish_full_figure(figure, ax, grid=False)

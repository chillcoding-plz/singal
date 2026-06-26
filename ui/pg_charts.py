"""pyqtgraph 版图表组件 — 替代 matplotlib 用于散点图/时间轴等高频绘图场景。
接口与 ui/charts.py 的 ChartCard 兼容，可在 AnalysisPage 中混用。
"""
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .widgets import TRACK_COLORS, track_color

# ---- pyqtgraph 全局配置 ----
pg.setConfigOptions(
    antialias=True,
    background="white",
    foreground=(51, 65, 85),
)

_CN_FONT = QFont("Microsoft YaHei", 8)
_CN_FONT_SMALL = QFont("Microsoft YaHei", 7)

TRACK_BAR_PREVIEW_LIMIT = 12


def _hex_to_rgba(hex_color: str, alpha: int = 180) -> tuple:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def _sample(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
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


def _track_counts(df: Optional[pd.DataFrame], visibility=None) -> pd.Series:
    track_column = _track_column(df)
    if df is None or track_column not in df:
        return pd.Series(dtype=int)
    visible_df = _visible(df, visibility or {})
    tracks = pd.to_numeric(visible_df[track_column], errors="coerce").fillna(0).astype(int)
    return tracks.value_counts().sort_values(ascending=False)


def _class_counts(data: Optional[pd.DataFrame]) -> pd.Series:
    if data is None or data.empty or "Predicted_Label" not in data:
        return pd.Series(dtype=int)
    return data["Predicted_Label"].replace("", pd.NA).dropna().astype(str).value_counts()


def _class_color_map(label_order, present_labels) -> Dict[str, str]:
    ordered = [str(label) for label in (label_order or [])]
    present = [str(label) for label in present_labels]
    for label in present:
        if label not in ordered:
            ordered.append(label)
    return {
        label: TRACK_COLORS[index % len(TRACK_COLORS)]
        for index, label in enumerate(ordered)
    }


def _build_scatter_brushes(data: pd.DataFrame):
    """从 DataFrame 逐行构建 RGBA 颜色列表"""
    track_column = _track_column(data)
    if track_column not in data:
        return [_hex_to_rgba("#1E88E5")] * len(data)
    tracks = pd.to_numeric(data[track_column], errors="coerce").fillna(0).astype(int)
    return [_hex_to_rgba(track_color(int(t))) for t in tracks]


def _state_hash(options: Dict[str, bool], visibility) -> int:
    """对显示状态做哈希，用于判断是否需要重建缓存"""
    vis_tuple = tuple(sorted((k, v) for k, v in (visibility or {}).items()))
    grid = options.get("显示网格", True)
    legend = options.get("显示图例", True)
    return hash((grid, legend, vis_tuple))


class PgChartCard(QFrame):
    """pyqtgraph 版图表卡片 — 对外 API 与 matplotlib ChartCard 兼容"""

    def __init__(self, title: str, parent=None, compact: bool = False):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.compact = compact
        self._detail_renderer: Optional[Callable] = None
        self._initialized: bool = False
        self._render_key: Optional[str] = None
        self._render_items: dict = {}

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_item: pg.PlotItem = self._plot_widget.getPlotItem()
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)
        self._plot_item.getAxis("bottom").setTickFont(_CN_FONT)
        self._plot_item.getAxis("left").setTickFont(_CN_FONT)

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
        layout.addWidget(self._plot_widget, 1)
        self._bottom_legend = QWidget()
        self._bottom_legend.setFixedHeight(44)
        self._bottom_legend_layout = QGridLayout(self._bottom_legend)
        self._bottom_legend_layout.setContentsMargins(8, 2, 8, 4)
        self._bottom_legend_layout.setHorizontalSpacing(12)
        self._bottom_legend_layout.setVerticalSpacing(2)
        self._bottom_legend.setVisible(False)
        layout.addWidget(self._bottom_legend)

        self._plot_widget.scene().sigMouseClicked.connect(self._on_double_click)

    def clear(self):
        self._plot_item.clear()
        self.set_bottom_legend([])
        self._initialized = False
        self._render_key = None
        self._render_items = {}

    def set_inline_legend(self, items):
        self.set_bottom_legend(items)

    def set_bottom_legend(self, items):
        while self._bottom_legend_layout.count():
            item = self._bottom_legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for index, (label, color) in enumerate(items):
            row = index % 2
            column = index // 2
            self._bottom_legend_layout.addWidget(self._legend_item(str(label), color), row, column)
        for column in range(max(1, (len(items) + 1) // 2)):
            self._bottom_legend_layout.setColumnStretch(column, 0)
        self._bottom_legend_layout.setColumnStretch(max(1, (len(items) + 1) // 2), 1)
        self._bottom_legend.setVisible(bool(items))

    def _legend_item(self, label: str, color):
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background: {self._legend_color(color)}; border-radius: 5px;")
        text = QLabel(label)
        text.setProperty("class", "subtle")
        text.setFont(_CN_FONT_SMALL)
        layout.addWidget(dot)
        layout.addWidget(text)
        return item

    def _legend_color(self, color):
        if isinstance(color, tuple):
            r, g, b = color[:3]
            return f"rgb({int(r)}, {int(g)}, {int(b)})"
        return str(color)

    def show_empty(self):
        self.clear()
        self._plot_item.setLabel("bottom", "")
        self._plot_item.setLabel("left", "")
        empty = pg.TextItem("暂无数据", color=(148, 163, 184), anchor=(0.5, 0.5))
        empty.setFont(_CN_FONT)
        self._plot_item.addItem(empty)

    def finish(self, grid: bool = True, legend: bool = True):
        self._plot_item.showGrid(x=grid, y=grid, alpha=0.3)
        if legend:
            legend_item = self._plot_item.addLegend(offset=(-10, 10))
            if legend_item is not None:
                legend_item.setLabelTextSize("8pt" if self.compact else "9pt")
        if self.compact:
            self._plot_item.setContentsMargins(2, 2, 2, 2)
        self._initialized = True

    @property
    def figure(self):
        return None

    def set_detail_renderer(self, renderer: Callable):
        self._detail_renderer = renderer

    def _on_double_click(self, event):
        if not event.double():
            return
        pos = event.scenePos()
        if self._plot_widget.sceneBoundingRect().contains(pos):
            PgInteractiveDialog(
                self.title.text(),
                self._detail_renderer,
                self,
            ).exec_()


class PgInteractiveDialog(QDialog):
    """pyqtgraph 版交互详情弹窗 — 支持全量数据查看、滚轮缩放、拖拽平移"""

    def __init__(self, title: str, renderer: Optional[Callable], parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1180, 760)

        self._view = pg.GraphicsLayoutWidget()
        self._view.setBackground("white")
        self._plot = self._view.addPlot()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.getAxis("bottom").setTickFont(_CN_FONT)
        self._plot.getAxis("left").setTickFont(_CN_FONT)
        self._plot.setMouseEnabled(x=True, y=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._view, 1)

        if renderer is not None:
            renderer(self._plot)


# ---- 绘图函数（pyqtgraph 版） ----


def _plot_track_scatter_pg(
    plot_item: pg.PlotItem,
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    y_label: str,
    show_legend: bool,
):
    """全量散点绘制到裸 plot_item（用于 PgInteractiveDialog，不走卡片缓存）"""
    if data is None or data.empty or x_column not in data or y_column not in data:
        return
    track_column = _track_column(data)
    if track_column not in data:
        scatter = pg.ScatterPlotItem(
            x=data[x_column].values, y=data[y_column].values,
            size=6, brush=_hex_to_rgba("#1E88E5"), pen=None,
        )
        plot_item.addItem(scatter)
    else:
        tracks = pd.to_numeric(data[track_column], errors="coerce").fillna(0).astype(int)
        counts = tracks.value_counts()
        for track_id in counts.index:
            track_id = int(track_id)
            group = data.loc[tracks == track_id]
            plot_item.addItem(pg.ScatterPlotItem(
                x=group[x_column].values,
                y=group[y_column].values,
                size=5,
                brush=_hex_to_rgba(track_color(track_id)),
                pen=None,
            ))

    if show_legend:
        legend_tracks = pd.Series(tracks).value_counts().head(8).index.tolist() if track_column in data else []
        legend = plot_item.addLegend(offset=(-10, 10))
        for tid in legend_tracks:
            label = "未分配" if int(tid) == 0 else f"轨迹 {int(tid)}"
            color = _hex_to_rgba(track_color(int(tid)), alpha=255)
            plot_item.addItem(pg.ScatterPlotItem(size=8, brush=color, pen=None, name=label))

    plot_item.setLabel("bottom", x_column)
    plot_item.setLabel("left", y_label)


# ---- 散点图（核心优化目标） ----


def _plot_toa_scatter_pg(
    card: PgChartCard,
    df: pd.DataFrame,
    options: Dict[str, bool],
    visibility,
    y_column: str,
    y_label: str,
):
    if y_column not in df:
        card.show_empty()
        return
    data = _sample(_visible(df, visibility or {}), 5000)
    show_legend = options.get("显示图例", True)
    render_key = f"scatter_{y_column}_{_state_hash(options, visibility)}"

    if card._render_key == render_key:
        # 就地更新 — 不清除不重建，只替换顶点数据
        scatter_item = card._render_items.get("scatter")
        if scatter_item is not None:
            scatter_item.setData(
                x=data["TOA"].values,
                y=data[y_column].values,
                brush=_build_scatter_brushes(data),
            )
    else:
        # 首次渲染或显示状态变化 — 完整重绘
        card.clear()
        _plot_track_scatter_pg(card._plot_item, data, "TOA", y_column, y_label, show_legend)
        for item in card._plot_item.items:
            if isinstance(item, pg.ScatterPlotItem):
                card._render_items["scatter"] = item
                break
        card._render_key = render_key

    card.finish(options.get("显示网格", True), False)


def plot_sort_scatter(card: PgChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    _plot_toa_scatter_pg(card, df, options, visibility, "PRI", "PRI")


def plot_rf_scatter(card: PgChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    _plot_toa_scatter_pg(card, df, options, visibility, "RF", "RF")


def plot_pw_scatter(card: PgChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    _plot_toa_scatter_pg(card, df, options, visibility, "PW", "脉宽 PW")


def plot_pa_scatter(card: PgChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    _plot_toa_scatter_pg(card, df, options, visibility, "PA", "脉幅 PA")


# ---- 时间轴 ----


def plot_timeline(card: PgChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    data = _sample(df, 5000)
    if "PA" in data:
        y = data["PA"].values.astype(float)
    else:
        y = np.ones(len(data), dtype=float)
    y_min, y_max = y.min(), y.max()
    y = (y - y_min) / (y_max - y_min + 1e-9)
    toa = data["TOA"].values
    bar_width = max((toa[-1] - toa[0]) / max(len(toa), 1) * 0.8, 1e-6)
    render_key = f"timeline_{_state_hash(options, visibility)}"

    if card._render_key == render_key:
        bar_item = card._render_items.get("bar")
        line_item = card._render_items.get("line")
        if bar_item is not None:
            bar_item.setOpts(x=toa, height=y, width=bar_width)
        if line_item is not None:
            line_item.setData(toa, y)
    else:
        card.clear()
        bar_item = pg.BarGraphItem(
            x=toa, height=y, width=bar_width, brush=_hex_to_rgba("#1E88E5"),
        )
        card._plot_item.addItem(bar_item)
        line_item = pg.PlotDataItem(
            toa, y, pen=pg.mkPen(color=(0, 82, 181, 165), width=0.9),
        )
        card._plot_item.addItem(line_item)
        card._render_items = {"bar": bar_item, "line": line_item}
        card._render_key = render_key
        card._plot_item.setLabel("bottom", "TOA")
        card._plot_item.setLabel("left", "归一化幅度")

    card.finish(options.get("显示网格", True), False)


def _plot_full_timeline_pg(plot_item: pg.PlotItem, data: pd.DataFrame):
    if data is None or data.empty or "TOA" not in data:
        return
    toa = pd.to_numeric(data["TOA"], errors="coerce")
    if "PA" in data:
        values = pd.to_numeric(data["PA"], errors="coerce")
        y_label = "归一化幅度"
    else:
        values = pd.Series(np.ones(len(data), dtype=float), index=data.index)
        y_label = "脉冲"
    work = pd.DataFrame({"TOA": toa, "value": values}).dropna()
    if work.empty:
        return
    y = work["value"].to_numpy(dtype=float)
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        y = (y - y_min) / (y_max - y_min)
    else:
        y = np.ones(len(work), dtype=float)
    x = work["TOA"].to_numpy(dtype=float)
    plot_item.addItem(pg.PlotDataItem(
        x,
        y,
        pen=pg.mkPen(color=(0, 82, 181, 175), width=1.0),
    ))
    plot_item.setLabel("bottom", "TOA")
    plot_item.setLabel("left", y_label)


# ---- 特征投影 ----


def plot_feature_projection(card: PgChartCard, data: pd.DataFrame, options: Dict[str, bool], visibility=None):
    if data is None or data.empty:
        card.show_empty()
        return
    if {"RF", "PRI"}.issubset(data.columns):
        work = _sample(_visible(data, visibility), 5000).copy()
        x_column, y_column = "RF", "PRI"
    elif {"Mean_RF", "Mean_PRI"}.issubset(data.columns):
        work = data.copy()
        x_column, y_column = "Mean_RF", "Mean_PRI"
    else:
        card.show_empty()
        return

    labels = work.get("Predicted_Label", pd.Series(["Unknown"] * len(work), index=work.index)).fillna("Unknown").astype(str)
    present_labels = _class_counts(work).index.tolist() or labels.value_counts().index.tolist()
    label_order = getattr(card, "_class_color_order", None) or present_labels
    color_map = _class_color_map(label_order, present_labels)
    unique_labels = [label for label in color_map if label in set(labels)]
    label_key = tuple(unique_labels)
    render_key = f"featproj_{x_column}_{y_column}_{label_key}_{_state_hash(options, visibility)}"
    show_legend = options.get("显示图例", True)

    if card._render_key == render_key and len(card._render_items.get("scatters", [])) == len(unique_labels):
        scatter_items = card._render_items.get("scatters", [])
        for idx, label in enumerate(unique_labels):
            group = work[labels == label]
            if idx < len(scatter_items):
                scatter_items[idx].setData(
                    x=group[x_column].values, y=group[y_column].values,
                )
    else:
        card.clear()
        scatter_items = []
        for idx, label in enumerate(unique_labels):
            group = work[labels == label]
            color = color_map[label]
            scatter = pg.ScatterPlotItem(
                x=group[x_column].values, y=group[y_column].values,
                size=7, brush=_hex_to_rgba(color),
                pen=None,
            )
            card._plot_item.addItem(scatter)
            scatter_items.append(scatter)
        card._render_items = {"scatters": scatter_items}
        card._render_key = render_key
        card._plot_item.setLabel("bottom", x_column)
        card._plot_item.setLabel("left", y_column)

    legend_items = [
        (label, color_map[label])
        for label in unique_labels
    ]
    card.set_bottom_legend(legend_items if show_legend else [])
    card.finish(options.get("显示网格", True), False)


# ---- 置信度 ----


def plot_probability(card: PgChartCard, data: pd.DataFrame, options: Dict[str, bool], visibility=None):
    if data is None or data.empty or "Confidence" not in data.columns:
        card.show_empty()
        return
    render_key = f"prob_{_state_hash(options, visibility)}"

    if "TOA" in data.columns:
        work = _sample(_visible(data, visibility), 5000).copy()
        x = pd.to_numeric(work["TOA"], errors="coerce").values
        confidence = pd.to_numeric(work["Confidence"], errors="coerce").fillna(0.0).clip(0, 1).values

        if card._render_key == render_key:
            scatter = card._render_items.get("scatter")
            if scatter is not None:
                scatter.setData(x=x, y=confidence)
        else:
            card.clear()
            scatter = pg.ScatterPlotItem(
                x=x, y=confidence, size=6,
                brush=_hex_to_rgba("#1E88E5"), pen=None,
            )
            card._plot_item.addItem(scatter)
            card._render_items = {"scatter": scatter}
            card._render_key = render_key
            card._plot_item.setLabel("bottom", "TOA")
    else:
        work = data.head(40).copy()
        labels = [f"T{int(t)}" for t in work["Track_ID"]]
        confidence = pd.to_numeric(work["Confidence"], errors="coerce").fillna(0.0).clip(0, 1).values
        x = np.arange(len(labels), dtype=float)

        if card._render_key == render_key:
            bar_bg = card._render_items.get("bar_bg")
            bar_fg = card._render_items.get("bar_fg")
            if bar_bg is not None:
                bar_bg.setOpts(x=x, height=np.ones(len(labels)))
            if bar_fg is not None:
                bar_fg.setOpts(x=x, height=confidence)
            card._plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
        else:
            card.clear()
            bar_bg = pg.BarGraphItem(
                x=x, height=np.ones(len(labels)), width=0.6,
                brush=(215, 227, 240, 180),
            )
            bar_fg = pg.BarGraphItem(
                x=x, height=confidence, width=0.6,
                brush=_hex_to_rgba("#1E88E5"),
            )
            card._plot_item.addItem(bar_bg)
            card._plot_item.addItem(bar_fg)
            card._render_items = {"bar_bg": bar_bg, "bar_fg": bar_fg}
            card._render_key = render_key
            card._plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])

    card._plot_item.setLabel("left", "置信度")
    card._plot_item.setYRange(0, 1)
    card.finish(options.get("显示网格", True), False)


# ---- 柱状图 ----


def plot_track_bars(card: PgChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    counts = _track_counts(df, visibility)
    if counts.empty:
        card.show_empty()
        return
    preview = counts.head(TRACK_BAR_PREVIEW_LIMIT)
    labels = ["未分配" if int(i) == 0 else f"轨迹{int(i)}" for i in preview.index]
    x = np.arange(len(labels), dtype=float)
    heights = np.array([int(v) for v in preview.values], dtype=float)
    brushes = [_hex_to_rgba(track_color(int(t)), alpha=220) for t in preview.index]
    render_key = f"trackbars_{_state_hash(options, visibility)}"

    if card._render_key == render_key:
        bar = card._render_items.get("bar")
        if bar is not None:
            bar.setOpts(x=x, height=heights, brushes=brushes)
        # 更新文本标签
        text_items = card._render_items.get("texts", [])
        for i, (h, txt) in enumerate(zip(heights, text_items)):
            txt.setText(str(int(h)))
            txt.setPos(x[i], h)
        card._plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
    else:
        card.clear()
        bar = pg.BarGraphItem(x=x, height=heights, width=0.6, brushes=brushes)
        card._plot_item.addItem(bar)
        text_items = []
        for i, h in enumerate(heights):
            txt = pg.TextItem(str(int(h)), color=(51, 65, 85), anchor=(0.5, 0))
            txt.setFont(_CN_FONT_SMALL)
            txt.setPos(x[i], h)
            card._plot_item.addItem(txt)
            text_items.append(txt)
        card._render_items = {"bar": bar, "texts": text_items}
        card._render_key = render_key
        card._plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
        card._plot_item.setLabel("left", "脉冲数")
        card._plot_item.getAxis("bottom").setStyle(tickTextOffset=8)

    card.finish(options.get("显示网格", True), False)


def plot_class_stats(card: PgChartCard, data: pd.DataFrame, options: Dict[str, bool], visibility=None):
    if data is None or data.empty or "Predicted_Label" not in data.columns:
        card.show_empty()
        return
    counts = data["Predicted_Label"].replace("", pd.NA).dropna().value_counts()
    if counts.empty:
        card.show_empty()
        return
    labels = counts.index.astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    heights = counts.values.astype(float)
    brushes = [_hex_to_rgba(TRACK_COLORS[i % len(TRACK_COLORS)], alpha=220) for i in range(len(labels))]
    render_key = f"classstats_{_state_hash(options, visibility)}"

    if card._render_key == render_key:
        bar = card._render_items.get("bar")
        if bar is not None:
            bar.setOpts(x=x, height=heights, brushes=brushes)
        text_items = card._render_items.get("texts", [])
        for i, (h, txt) in enumerate(zip(heights, text_items)):
            txt.setText(str(int(h)))
            txt.setPos(x[i], h)
        card._plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
    else:
        card.clear()
        bar = pg.BarGraphItem(x=x, height=heights, width=0.6, brushes=brushes)
        card._plot_item.addItem(bar)
        text_items = []
        for i, h in enumerate(heights):
            txt = pg.TextItem(str(int(h)), color=(51, 65, 85), anchor=(0.5, 0))
            txt.setFont(_CN_FONT_SMALL)
            txt.setPos(x[i], h)
            card._plot_item.addItem(txt)
            text_items.append(txt)
        card._render_items = {"bar": bar, "texts": text_items}
        card._render_key = render_key
        card._plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
        card._plot_item.setLabel("left", "轨迹数")

    card.finish(options.get("显示网格", True), False)


def _plot_full_probability_pg(plot_item: pg.PlotItem, data: pd.DataFrame, visibility=None):
    if data is None or data.empty or "Confidence" not in data:
        return
    full_data = _visible(data, visibility or {})
    if full_data is None or full_data.empty:
        return
    confidence = pd.to_numeric(full_data["Confidence"], errors="coerce").fillna(0.0).clip(0, 1)
    if "TOA" in full_data:
        x = pd.to_numeric(full_data["TOA"], errors="coerce")
        work = pd.DataFrame({"x": x, "confidence": confidence}).dropna()
        if work.empty:
            return
        plot_item.addItem(pg.ScatterPlotItem(
            x=work["x"].to_numpy(dtype=float),
            y=work["confidence"].to_numpy(dtype=float),
            size=5,
            brush=_hex_to_rgba("#1E88E5"),
            pen=None,
        ))
        plot_item.setLabel("bottom", "TOA")
    else:
        if "Track_ID" in full_data:
            labels = [f"T{int(t)}" for t in pd.to_numeric(full_data["Track_ID"], errors="coerce").fillna(0)]
        else:
            labels = [str(index + 1) for index in range(len(full_data))]
        x = np.arange(len(labels), dtype=float)
        plot_item.addItem(pg.BarGraphItem(
            x=x,
            height=np.ones(len(labels), dtype=float),
            width=0.6,
            brush=(215, 227, 240, 180),
        ))
        plot_item.addItem(pg.BarGraphItem(
            x=x,
            height=confidence.to_numpy(dtype=float),
            width=0.6,
            brush=_hex_to_rgba("#1E88E5"),
        ))
        plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
    plot_item.setLabel("left", "置信度")
    plot_item.setYRange(0, 1)


def _plot_full_class_stats_pg(plot_item: pg.PlotItem, data: pd.DataFrame):
    if data is None or data.empty or "Predicted_Label" not in data:
        return
    counts = data["Predicted_Label"].replace("", pd.NA).dropna().value_counts()
    if counts.empty:
        return
    labels = counts.index.astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    heights = counts.to_numpy(dtype=float)
    brushes = [_hex_to_rgba(TRACK_COLORS[i % len(TRACK_COLORS)], alpha=220) for i in range(len(labels))]
    plot_item.addItem(pg.BarGraphItem(x=x, height=heights, width=0.6, brushes=brushes))
    for i, h in enumerate(heights):
        txt = pg.TextItem(str(int(h)), color=(51, 65, 85), anchor=(0.5, 0))
        txt.setFont(_CN_FONT_SMALL)
        txt.setPos(x[i], h)
        plot_item.addItem(txt)
    plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
    plot_item.setLabel("left", "轨迹数")


def _truth_column(df: pd.DataFrame) -> Optional[str]:
    for column in ("Original_Track_ID", "True_Track_ID"):
        if column in df.columns:
            return column
    return None


def _cycle_period_prediction_column(df: pd.DataFrame) -> Optional[str]:
    if "Sorting_Method" in df.columns:
        methods = df["Sorting_Method"].dropna().astype(str).str.lower()
        if methods.str.contains("mht").any():
            if "MHT_MHTId" in df.columns:
                return "MHT_MHTId"
            if "MHT_Track_ID" in df.columns:
                return "MHT_Track_ID"
    if "CyclePeriod_OurPredID" not in df.columns:
        return None
    if "Sorting_Method" not in df.columns:
        return "CyclePeriod_OurPredID"
    methods = df["Sorting_Method"].dropna().astype(str).str.lower()
    return "CyclePeriod_OurPredID" if methods.str.contains("cycle").any() else None


def _majority_match_accuracy(df: pd.DataFrame) -> Optional[float]:
    truth_col = _truth_column(df)
    prediction_col = _cycle_period_prediction_column(df) or "Track_ID"
    if truth_col is None or prediction_col not in df.columns:
        return None
    data = df[[prediction_col, truth_col]].copy()
    data[prediction_col] = pd.to_numeric(data[prediction_col], errors="coerce").fillna(0).astype(int)
    data = data.dropna(subset=[truth_col])
    if prediction_col in {"CyclePeriod_OurPredID", "MHT_MHTId", "MHT_Track_ID"} and "Track_ID" in df.columns:
        assigned = pd.to_numeric(df.loc[data.index, "Track_ID"], errors="coerce").fillna(0).astype(int) > 0
        data = data[assigned]
    else:
        data = data[data[prediction_col] > 0]
    if data.empty:
        return None
    correct = 0
    for _, group in data.groupby(prediction_col, sort=False):
        correct += int(group[truth_col].astype(str).value_counts().iloc[0])
    return correct / max(len(df.dropna(subset=[truth_col])), 1)


# ---- 质量柱状图（替代 matplotlib 饼图） ----


def plot_quality(card: PgChartCard, df: pd.DataFrame, options: Dict[str, bool], visibility=None):
    if df is None or df.empty or "Assigned" not in df:
        card.show_empty()
        return
    assigned = int(df["Assigned"].sum())
    unassigned = int(len(df) - assigned)
    accuracy = _majority_match_accuracy(df)
    labels = ["已分配", "未分配"]
    x = np.array([0, 1], dtype=float)
    heights = np.array([assigned, unassigned], dtype=float)
    brushes = [_hex_to_rgba("#1E88E5", 220), _hex_to_rgba("#78909C", 220)]
    accuracy_text = f"准确率 {accuracy * 100:.1f}%" if accuracy is not None else "暂无真值"
    render_key = f"quality_{_state_hash(options, visibility)}_{accuracy}"

    if card._render_key == render_key:
        bar = card._render_items.get("bar")
        if bar is not None:
            bar.setOpts(x=x, height=heights, brushes=brushes)
        text_items = card._render_items.get("texts", [])
        for i, (h, txt) in enumerate(zip(heights, text_items)):
            total = max(assigned + unassigned, 1)
            pct = f"{h / total * 100:.1f}%"
            txt.setText(f"{int(h):,}\n{pct}")
            txt.setPos(x[i], h)
        acc_txt = card._render_items.get("accuracy")
        if acc_txt is not None:
            acc_txt.setText(accuracy_text)
    else:
        card.clear()
        bar = pg.BarGraphItem(x=x, height=heights, width=0.5, brushes=brushes)
        card._plot_item.addItem(bar)
        text_items = []
        total = max(assigned + unassigned, 1)
        for i, h in enumerate(heights):
            pct = f"{h / total * 100:.1f}%"
            txt = pg.TextItem(f"{int(h):,}\n{pct}", color=(51, 65, 85), anchor=(0.5, 0))
            txt.setFont(_CN_FONT_SMALL)
            txt.setPos(x[i], h)
            card._plot_item.addItem(txt)
            text_items.append(txt)
        # 准确率文字标注
        acc_color = (15, 118, 110) if accuracy is not None else (100, 116, 139)  # #0F766E / #64748B
        acc_txt = pg.TextItem(accuracy_text, color=acc_color, anchor=(0.5, 0.5))
        acc_txt.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        acc_txt.setPos(0.5, max(heights) * 1.15)
        card._plot_item.addItem(acc_txt)
        card._render_items = {"bar": bar, "texts": text_items, "accuracy": acc_txt}
        card._render_key = render_key
        card._plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
        card._plot_item.setLabel("left", "脉冲数")

    card.finish(options.get("显示网格", True), False)


# ---- 全量详情渲染 ----

def render_full_detail_pg(
    plot_item: pg.PlotItem,
    plotter: Callable,
    data: Optional[pd.DataFrame],
    track_results: Optional[pd.DataFrame],
    options: Dict[str, bool],
    visibility=None,
):
    """在给定 plot_item 上渲染全量数据（用于 PgInteractiveDialog）"""
    visibility = visibility or {}
    full_data = _visible(data, visibility) if data is not None else None

    if plotter is plot_timeline:
        if full_data is not None:
            _plot_full_timeline_pg(plot_item, full_data)
        return

    if plotter is plot_sort_scatter:
        if full_data is not None:
            _plot_track_scatter_pg(plot_item, full_data, "TOA", "PRI", "PRI", True)
        return

    if plotter is plot_rf_scatter:
        if full_data is not None:
            _plot_track_scatter_pg(plot_item, full_data, "TOA", "RF", "RF", True)
        return

    if plotter is plot_pw_scatter:
        if full_data is not None and "PW" in full_data:
            _plot_track_scatter_pg(plot_item, full_data, "TOA", "PW", "PW", True)
        return

    if plotter is plot_pa_scatter:
        if full_data is not None and "PA" in full_data:
            _plot_track_scatter_pg(plot_item, full_data, "TOA", "PA", "PA", True)
        return

    if plotter is plot_track_bars:
        counts = _track_counts(data, visibility)
        if counts.empty:
            return
        labels_full = ["未分配" if int(i) == 0 else f"轨迹{int(i)}" for i in counts.index]
        y = np.arange(len(counts))
        bar = pg.BarGraphItem(
            x0=np.zeros(len(counts)),
            y=y,
            width=counts.values,
            height=0.7,
            brushes=[_hex_to_rgba(track_color(int(t)), alpha=220) for t in counts.index],
        )
        plot_item.addItem(bar)
        plot_item.getAxis("left").setTicks([list(zip(y, labels_full))])
        plot_item.setLabel("bottom", "脉冲数")
        plot_item.setLabel("left", "轨迹")
        return

    if plotter is plot_quality:
        if data is not None and "Assigned" in data:
            assigned = int(data["Assigned"].sum())
            unassigned = int(len(data) - assigned)
            accuracy = _majority_match_accuracy(data)
            labels = ["已分配", "未分配"]
            x = np.array([0, 1], dtype=float)
            heights = np.array([assigned, unassigned], dtype=float)
            brushes = [_hex_to_rgba("#1E88E5", 220), _hex_to_rgba("#78909C", 220)]
            bar = pg.BarGraphItem(x=x, height=heights, width=0.5, brushes=brushes)
            plot_item.addItem(bar)
            total = max(assigned + unassigned, 1)
            for i, h in enumerate(heights):
                pct = f"{h / total * 100:.1f}%"
                txt = pg.TextItem(f"{int(h):,}\n{pct}", color=(51, 65, 85), anchor=(0.5, 0))
                txt.setFont(_CN_FONT_SMALL)
                txt.setPos(x[i], h)
                plot_item.addItem(txt)
            accuracy_text = f"准确率 {accuracy * 100:.1f}%" if accuracy is not None else "暂无真值"
            acc_color = (15, 118, 110) if accuracy is not None else (100, 116, 139)
            acc_txt = pg.TextItem(accuracy_text, color=acc_color, anchor=(0.5, 0.5))
            acc_txt.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            acc_txt.setPos(0.5, max(heights) * 1.15)
            plot_item.addItem(acc_txt)
            plot_item.getAxis("bottom").setTicks([list(zip(x, labels))])
            plot_item.setLabel("left", "脉冲数")
        return

    if plotter is plot_probability:
        _plot_full_probability_pg(plot_item, data, visibility)
        return

    if plotter is plot_class_stats:
        _plot_full_class_stats_pg(plot_item, data)
        return

    if plotter is plot_feature_projection:
        if full_data is not None:
            if {"RF", "PRI"}.issubset(full_data.columns):
                x_col, y_col = "RF", "PRI"
            elif {"Mean_RF", "Mean_PRI"}.issubset(full_data.columns):
                x_col, y_col = "Mean_RF", "Mean_PRI"
            else:
                return
            labels = full_data.get("Predicted_Label", pd.Series(["Unknown"] * len(full_data), index=full_data.index)).fillna("Unknown").astype(str)
            present_labels = _class_counts(full_data).index.tolist() or labels.value_counts().index.tolist()
            label_order = _class_counts(track_results).index.tolist() or present_labels
            color_map = _class_color_map(label_order, present_labels)
            for label in [label for label in color_map if label in set(labels)]:
                group = full_data[labels == label]
                plot_item.addItem(pg.ScatterPlotItem(
                    x=group[x_col].values, y=group[y_col].values,
                    size=4, brush=_hex_to_rgba(color_map[label]),
                    pen=None, name=str(label),
                ))
            plot_item.setLabel("bottom", x_col)
            plot_item.setLabel("left", y_col)
        return

    text = pg.TextItem("Preview chart", color=(148, 163, 184), anchor=(0.5, 0.5))
    plot_item.addItem(text)

import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout


BLUE = "#1E88E5"
DEEP_BLUE = "#0052B5"
TRACK_COLORS = ["#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#78909C"]
CLASS_NAME_MAP = {
    "Radar_A": "雷达 A",
    "Radar_B": "雷达 B",
    "Comm_Pulse": "通信脉冲",
    "Unknown": "未知信号",
}

rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
rcParams["axes.unicode_minus"] = False


def has_data(data):
    return data is not None and getattr(data, "size", 0) > 0 and getattr(data.dtype, "names", None)


def col(data, name, default=None):
    if not has_data(data) or name not in data.dtype.names:
        return default
    values = data[name]
    if values.shape == ():
        values = np.array([values.item()])
    return values


def numeric_col(data, name, default=None):
    values = col(data, name, default)
    if values is default:
        return default
    return np.asarray(values, dtype=float)


def text_col(data, name, default=None):
    values = col(data, name, default)
    if values is default:
        return default
    return np.asarray(values).astype(str)


def downsample_indices(length, max_points=1800):
    if length <= max_points:
        return np.arange(length)
    return np.linspace(0, length - 1, max_points).astype(int)


class ChartCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")
        self.figure = Figure(figsize=(4, 3), dpi=100, facecolor="white")
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

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

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("white")

    def show_empty(self):
        self.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_color("#d7e0ea")
        self.ax.text(
            0.5,
            0.5,
            "暂无数据",
            ha="center",
            va="center",
            color="#94a3b8",
            fontsize=12,
            transform=self.ax.transAxes,
        )
        self.canvas.draw_idle()

    def finish(self, grid=True, legend=False):
        if grid:
            self.ax.grid(True, linestyle="--", linewidth=0.55, color="#d8e1ec", alpha=0.9)
        for spine in self.ax.spines.values():
            spine.set_color("#aebdcc")
        self.ax.tick_params(colors="#334155", labelsize=8)
        self.ax.xaxis.label.set_color("#334155")
        self.ax.yaxis.label.set_color("#334155")
        if legend:
            self.ax.legend(frameon=True, fontsize=8, loc="best")
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

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["样本", "类别", "置信度", "PRI/ms", "RF/MHz"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)

    def clear(self):
        self.table.setRowCount(0)

    def refresh(self, seed=None, data=None):
        if has_data(data):
            pulse_id = col(data, "pulse_id", np.arange(1, len(data) + 1))
            classes = text_col(data, "class_label", np.array(["Unknown"] * len(data)))
            confidence = numeric_col(data, "confidence", np.zeros(len(data)))
            pri = numeric_col(data, "pri_us", np.zeros(len(data))) / 1000.0
            rf = numeric_col(data, "rf_mhz", np.zeros(len(data)))
            count = min(12, len(data))
            rows = np.linspace(0, len(data) - 1, count).astype(int)
            self.table.setRowCount(count)
            for row, idx in enumerate(rows):
                raw_class = str(classes[idx])
                values = [
                    f"S-{int(pulse_id[idx]):03d}",
                    CLASS_NAME_MAP.get(raw_class, raw_class),
                    f"{confidence[idx] * 100:.1f}%",
                    f"{pri[idx]:.2f}",
                    f"{rf[idx]:.1f}",
                ]
                for col_index, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(row, col_index, item)
            self.table.resizeColumnsToContents()
            return

        rng = np.random.default_rng(seed)
        classes = ["雷达 A", "雷达 B", "通信脉冲", "未知信号"]
        self.table.setRowCount(12)
        for row in range(12):
            values = [
                f"S-{row + 1:03d}",
                rng.choice(classes),
                f"{rng.uniform(78, 98):.1f}%",
                f"{rng.uniform(0.7, 3.2):.2f}",
                f"{rng.uniform(810, 1220):.1f}",
            ]
            for col_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col_index, item)
        self.table.resizeColumnsToContents()


def pulse_timeline(card, seed=None, visible=None, data=None):
    card.clear()
    if has_data(data):
        toa = numeric_col(data, "toa_us", np.arange(len(data))) / 1000.0
        pa = numeric_col(data, "pa_db", np.zeros(len(data)))
        idx = downsample_indices(len(toa), 1600)
        norm_pa = (pa[idx] - np.min(pa[idx])) / (np.ptp(pa[idx]) + 1e-9)
        card.ax.vlines(toa[idx], 0, norm_pa, color=BLUE, alpha=0.72, linewidth=0.8, label="脉冲幅度")
        card.ax.plot(toa[idx], norm_pa, color=DEEP_BLUE, linewidth=0.9, alpha=0.65, label="包络趋势")
        card.ax.set_xlabel("TOA / ms")
        card.ax.set_ylabel("归一化幅度")
        card.finish(legend=True)
        return

    rng = np.random.default_rng(seed)
    toa = np.linspace(0, 10, 460)
    amp = np.sin(toa * 7.0) * 0.25 + rng.normal(0, 0.08, toa.size)
    pulse_positions = np.sort(rng.uniform(0, 10, 80))
    pulse_heights = rng.uniform(0.55, 1.0, pulse_positions.size)
    card.ax.plot(toa, amp, color=DEEP_BLUE, linewidth=1.25, label="背景波形")
    card.ax.vlines(pulse_positions, 0, pulse_heights, color=BLUE, alpha=0.78, linewidth=1.0, label="脉冲")
    card.ax.set_xlabel("TOA / s")
    card.ax.set_ylabel("幅度")
    card.finish(legend=True)


def sorting_scatter(card, seed=None, visible=None, data=None):
    card.clear()
    visible = visible or [True, True, True, True, True]
    if has_data(data):
        toa = numeric_col(data, "toa_us", np.arange(len(data))) / 1000.0
        pri = numeric_col(data, "pri_us", np.zeros(len(data))) / 1000.0
        tracks = numeric_col(data, "track_id", np.ones(len(data))).astype(int)
        idx = downsample_indices(len(toa), 2200)
        for track_id, color in zip([1, 2, 3, 4], TRACK_COLORS[:4]):
            if not visible[track_id - 1]:
                continue
            mask = tracks[idx] == track_id
            card.ax.scatter(toa[idx][mask], pri[idx][mask], s=14, color=color, alpha=0.78, label=f"轨迹 {track_id}")
        if len(visible) > 4 and visible[4]:
            mask = tracks[idx] == 0
            card.ax.scatter(toa[idx][mask], pri[idx][mask], s=14, color=TRACK_COLORS[-1], alpha=0.55, label="未分配")
        card.ax.set_xlabel("TOA / ms")
        card.ax.set_ylabel("PRI / ms")
        card.finish(legend=True)
        return

    rng = np.random.default_rng(seed)
    for i, color in enumerate(TRACK_COLORS[:4]):
        if not visible[i]:
            continue
        x = np.linspace(0, 10, 75) + rng.normal(0, 0.08, 75)
        y = (i + 1) * 0.55 + np.sin(x * (0.6 + i * 0.13)) * 0.12 + rng.normal(0, 0.035, 75)
        card.ax.scatter(x, y, s=18, color=color, alpha=0.82, label=f"轨迹 {i + 1}")
    if len(visible) > 4 and visible[4]:
        card.ax.scatter(rng.uniform(0, 10, 35), rng.uniform(0.25, 2.8, 35), s=16, color=TRACK_COLORS[-1], alpha=0.55, label="未分配")
    card.ax.set_xlabel("TOA / s")
    card.ax.set_ylabel("PRI / ms")
    card.finish(legend=True)


def rf_distribution(card, seed=None, visible=None, data=None):
    card.clear()
    visible = visible or [True, True, True, True, True]
    if has_data(data):
        toa = numeric_col(data, "toa_us", np.arange(len(data))) / 1000.0
        rf = numeric_col(data, "rf_mhz", np.zeros(len(data)))
        tracks = numeric_col(data, "track_id", np.ones(len(data))).astype(int)
        idx = downsample_indices(len(toa), 2200)
        for track_id, color in zip([1, 2, 3, 4], TRACK_COLORS[:4]):
            if not visible[track_id - 1]:
                continue
            mask = tracks[idx] == track_id
            card.ax.scatter(toa[idx][mask], rf[idx][mask], s=14, color=color, alpha=0.78, label=f"轨迹 {track_id}")
        if len(visible) > 4 and visible[4]:
            mask = tracks[idx] == 0
            card.ax.scatter(toa[idx][mask], rf[idx][mask], s=14, color=TRACK_COLORS[-1], alpha=0.55, label="未分配")
        card.ax.set_xlabel("TOA / ms")
        card.ax.set_ylabel("RF / MHz")
        card.finish(legend=True)
        return

    rng = np.random.default_rng(seed)
    centers = [860, 940, 1040, 1160]
    for i, color in enumerate(TRACK_COLORS[:4]):
        if not visible[i]:
            continue
        x = np.linspace(0, 10, 80)
        y = centers[i] + np.sin(x * (0.9 + i * 0.2)) * 18 + rng.normal(0, 8, x.size)
        card.ax.scatter(x, y, s=17, color=color, alpha=0.80, label=f"轨迹 {i + 1}")
    card.ax.set_xlabel("TOA / s")
    card.ax.set_ylabel("RF / MHz")
    card.finish(legend=True)


def trajectory_bars(card, seed=None, visible=None, data=None):
    card.clear()
    labels = [f"轨迹{i}" for i in range(1, 5)] + ["未分配"]
    if has_data(data):
        tracks = numeric_col(data, "track_id", np.ones(len(data))).astype(int)
        values = [int(np.sum(tracks == i)) for i in [1, 2, 3, 4]] + [int(np.sum(tracks == 0))]
    else:
        rng = np.random.default_rng(seed)
        values = np.array([12600, 12180, 13040, 12131, 49]) + rng.integers(-450, 450, 5)
    bars = card.ax.bar(labels, values, color=TRACK_COLORS, alpha=0.9)
    card.ax.bar_label(bars, labels=[f"{int(v):,}" for v in values], fontsize=8, padding=2)
    card.ax.set_ylabel("脉冲数")
    card.finish()


def quality_donut(card, seed=None, visible=None, data=None):
    card.clear()
    if has_data(data):
        tracks = numeric_col(data, "track_id", np.ones(len(data))).astype(int)
        confidence = numeric_col(data, "sorting_confidence", None)
        if confidence is None:
            confidence = numeric_col(data, "confidence", np.zeros(len(data)))
        assigned = np.sum(tracks > 0)
        unknown = np.sum(tracks == 0)
        low_conf = np.sum((tracks > 0) & (confidence < 0.86))
        correct = max(assigned - low_conf, 0)
        values = [correct, low_conf, unknown]
    else:
        rng = np.random.default_rng(seed)
        values = [rng.uniform(93, 97), rng.uniform(2.0, 4.0), rng.uniform(0.5, 1.5)]
    labels = ["高置信分配", "低置信分配", "未分配"]
    colors = [BLUE, "#FB8C00", "#78909C"]
    total = max(sum(values), 1)
    wedges, _ = card.ax.pie(values, colors=colors, startangle=90, wedgeprops={"width": 0.38, "edgecolor": "white"})
    card.ax.text(0, 0, f"{values[0] / total * 100:.1f}%\n质量", ha="center", va="center", fontsize=13, color=DEEP_BLUE, fontweight="bold")
    card.ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8, frameon=False)
    card.ax.set_aspect("equal")
    card.figure.tight_layout(pad=0.8)
    card.canvas.draw_idle()


def feature_projection(card, seed=None, visible=None, data=None):
    card.clear()
    if has_data(data):
        pri = numeric_col(data, "pri_us", np.zeros(len(data))) / 1000.0
        rf = numeric_col(data, "rf_mhz", np.zeros(len(data)))
        tracks = numeric_col(data, "track_id", np.ones(len(data))).astype(int)
        idx = downsample_indices(len(pri), 1800)
        x = (pri[idx] - np.mean(pri[idx])) / (np.std(pri[idx]) + 1e-9)
        y = (rf[idx] - np.mean(rf[idx])) / (np.std(rf[idx]) + 1e-9)
        for track_id, color in zip([1, 2, 3, 4], TRACK_COLORS[:4]):
            mask = tracks[idx] == track_id
            card.ax.scatter(x[mask], y[mask], s=14, color=color, alpha=0.76, label=f"类别 {track_id}")
        mask = tracks[idx] == 0
        if np.any(mask):
            card.ax.scatter(x[mask], y[mask], s=14, color=TRACK_COLORS[-1], alpha=0.55, label="未知")
        card.ax.set_xlabel("PRI 标准化特征")
        card.ax.set_ylabel("RF 标准化特征")
        card.finish(legend=True)
        return

    rng = np.random.default_rng(seed)
    centers = [(-1.4, 0.8), (1.1, 1.1), (-0.5, -1.0), (1.5, -0.7)]
    for i, (cx, cy) in enumerate(centers):
        pts = rng.normal((cx, cy), (0.35, 0.28), (70, 2))
        card.ax.scatter(pts[:, 0], pts[:, 1], s=18, color=TRACK_COLORS[i], alpha=0.78, label=f"类别 {i + 1}")
    card.ax.set_xlabel("特征 1")
    card.ax.set_ylabel("特征 2")
    card.finish(legend=True)


def probability_stacked(card, seed=None, visible=None, data=None):
    card.clear()
    if has_data(data):
        classes = text_col(data, "class_label", np.array(["Unknown"] * len(data)))
        confidence = numeric_col(data, "confidence", np.zeros(len(data)))
        rows = np.linspace(0, len(data) - 1, min(8, len(data))).astype(int)
        samples = [f"S{idx + 1}" for idx in rows]
        class_order = ["Radar_A", "Radar_B", "Comm_Pulse", "Unknown"]
        bottoms = np.zeros(len(rows))
        for class_index, class_name in enumerate(class_order):
            values = np.full(len(rows), 0.05)
            mask = classes[rows] == class_name
            values[mask] = confidence[rows][mask]
            values[~mask] = (1.0 - confidence[rows][~mask]) / 3.0
            card.ax.bar(samples, values, bottom=bottoms, color=TRACK_COLORS[class_index], label=CLASS_NAME_MAP[class_name], alpha=0.9)
            bottoms += values
        card.ax.set_ylim(0, 1.05)
        card.ax.set_ylabel("概率")
        card.finish(legend=True)
        return

    rng = np.random.default_rng(seed)
    samples = [f"S{i}" for i in range(1, 7)]
    raw = rng.uniform(0.08, 1.0, (4, len(samples)))
    probs = raw / raw.sum(axis=0)
    bottom = np.zeros(len(samples))
    for i, color in enumerate(TRACK_COLORS[:4]):
        card.ax.bar(samples, probs[i], bottom=bottom, color=color, label=f"类别 {i + 1}", alpha=0.9)
        bottom += probs[i]
    card.ax.set_ylim(0, 1)
    card.ax.set_ylabel("概率")
    card.finish(legend=True)


def class_stats(card, seed=None, visible=None, data=None):
    card.clear()
    if has_data(data):
        classes = text_col(data, "class_label", np.array(["Unknown"] * len(data)))
        labels = ["雷达 A", "雷达 B", "通信", "未知"]
        values = [
            int(np.sum(classes == "Radar_A")),
            int(np.sum(classes == "Radar_B")),
            int(np.sum(classes == "Comm_Pulse")),
            int(np.sum(classes == "Unknown")),
        ]
    else:
        rng = np.random.default_rng(seed)
        labels = ["雷达 A", "雷达 B", "通信", "未知"]
        values = rng.integers(8, 28, len(labels))
    bars = card.ax.bar(labels, values, color=TRACK_COLORS[:4], alpha=0.9)
    card.ax.bar_label(bars, fontsize=8, padding=2)
    card.ax.set_ylabel("样本数")
    card.finish()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualization helpers for the tracklet-graph sorting pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection


_AXIS_STYLE = {
    "labelpad": 8,
}


def _prepare_out(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sample_rows(df: pd.DataFrame, max_points: int, seed: int = 1234) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed).sort_index()


def _sig_colors(sigidx: np.ndarray):
    sigidx = np.asarray(sigidx, dtype=np.int64)
    positive = sorted(int(v) for v in np.unique(sigidx) if int(v) > 0)
    cmap = plt.get_cmap("tab10")
    mapping = {v: cmap(i % 10) for i, v in enumerate(positive)}
    colors = np.array([mapping.get(int(v), (0.34, 0.38, 0.44, 0.86)) for v in sigidx])
    return colors, positive


def _set_compact_limits(ax, x, y, z, q_low: float = 1.0, q_high: float = 99.0) -> None:
    values = [np.asarray(v, dtype=float) for v in (x, y, z)]
    limits = []
    for data in values:
        data = data[np.isfinite(data)]
        if len(data) == 0:
            limits.append(None)
            continue
        lo, hi = np.percentile(data, [q_low, q_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(data)), float(np.max(data))
        pad = 0.06 * max(hi - lo, 1e-9)
        limits.append((lo - pad, hi + pad))
    if limits[0] is not None:
        ax.set_xlim(*limits[0])
    if limits[1] is not None:
        ax.set_ylim(*limits[1])
    if limits[2] is not None:
        ax.set_zlim(*limits[2])


def _set_compact_limits_2d(ax, x, y, q_low: float = 1.0, q_high: float = 99.0) -> None:
    for setter, data in [(ax.set_xlim, x), (ax.set_ylim, y)]:
        values = np.asarray(data, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        lo, hi = np.percentile(values, [q_low, q_high])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(values)), float(np.max(values))
        pad = 0.06 * max(hi - lo, 1e-9)
        setter(lo - pad, hi + pad)


def _setup_2d(ax, title: str, ylabel: str = "RF / Param1") -> None:
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_xlabel("TOA (s)", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.grid(True, alpha=0.20, linewidth=0.8)
    ax.set_facecolor("#fbfcfe")
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _setup_3d(ax, title: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("TOA (s)", **_AXIS_STYLE)
    ax.set_ylabel("RF / Param1", **_AXIS_STYLE)
    ax.set_zlabel("DOA / Param5", **_AXIS_STYLE)
    ax.view_init(elev=24, azim=135)
    ax.grid(True, alpha=0.22)
    ax.xaxis.pane.set_facecolor((0.98, 0.99, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((0.98, 0.99, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((0.98, 0.99, 1.0, 1.0))
    ax.tick_params(axis="both", which="major", labelsize=8, pad=2)


def _finish(fig, output_path: Path, dpi: int) -> None:
    output_path = _prepare_out(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_raw_pdw_3d(pdw: pd.DataFrame, output_path: Path, max_points: int = 12000, dpi: int = 300) -> None:
    view = _sample_rows(pdw, max_points)
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        view["TOA(s)"], view["Param1"], view["Param5"],
        c="#263f5f", s=12, alpha=0.82, linewidths=0,
    )
    _setup_3d(ax, "Stage 1  Raw PDW Distribution")
    _set_compact_limits(ax, view["TOA(s)"], view["Param1"], view["Param5"])
    _finish(fig, output_path, dpi)


def plot_sigidx_3d(
    pdw: pd.DataFrame,
    sigidx: Iterable[int],
    output_path: Path,
    title: str,
    max_points: int = 12000,
    dpi: int = 300,
) -> None:
    sigidx = np.asarray(sigidx, dtype=np.int64)
    work = pdw.copy()
    work["_sigidx"] = sigidx[: len(work)]
    view = _sample_rows(work, max_points)
    colors, positive = _sig_colors(view["_sigidx"].to_numpy(dtype=np.int64))

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        view["TOA(s)"], view["Param1"], view["Param5"],
        c=colors, s=14, alpha=0.90, linewidths=0,
    )
    _setup_3d(ax, title)
    _set_compact_limits(ax, view["TOA(s)"], view["Param1"], view["Param5"])
    ax.text2D(
        0.02, 0.96,
        f"batches: {len(positive)} | pulses shown: {len(view):,}",
        transform=ax.transAxes,
        fontsize=10,
        color="#344054",
    )
    _finish(fig, output_path, dpi)


def plot_raw_pdw_2d(pdw: pd.DataFrame, output_path: Path, max_points: int = 12000, dpi: int = 300) -> None:
    view = _sample_rows(pdw, max_points)
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.scatter(view["TOA(s)"], view["Param1"], c="#183b63", s=18, alpha=0.88, linewidths=0)
    _setup_2d(ax, "Stage 1  Raw PDW Distribution")
    _set_compact_limits_2d(ax, view["TOA(s)"], view["Param1"])
    ax.text(0.015, 0.96, f"pulses shown: {len(view):,}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_sigidx_2d(
    pdw: pd.DataFrame,
    sigidx: Iterable[int],
    output_path: Path,
    title: str,
    max_points: int = 12000,
    dpi: int = 300,
) -> None:
    sigidx = np.asarray(sigidx, dtype=np.int64)
    work = pdw.copy()
    work["_sigidx"] = sigidx[: len(work)]
    view = _sample_rows(work, max_points)
    colors, positive = _sig_colors(view["_sigidx"].to_numpy(dtype=np.int64))

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.scatter(view["TOA(s)"], view["Param1"], c=colors, s=20, alpha=0.92, linewidths=0)
    _setup_2d(ax, title)
    _set_compact_limits_2d(ax, view["TOA(s)"], view["Param1"])
    ax.text(0.015, 0.96, f"batches: {len(positive)} | pulses shown: {len(view):,}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_tracklets_3d(
    pdw: pd.DataFrame,
    sigidx: Iterable[int],
    nodes: pd.DataFrame,
    output_path: Path,
    max_tracklets: int = 120,
    dpi: int = 300,
) -> None:
    sigidx = np.asarray(sigidx, dtype=np.int64)
    work = pdw.copy()
    work["_sigidx"] = sigidx[: len(work)]
    if len(nodes) > max_tracklets:
        nodes_view = nodes.nlargest(max_tracklets, "num_pulses")
    else:
        nodes_view = nodes

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    segments = []
    seg_colors = []
    centers = []
    center_colors = []

    for i, row in enumerate(nodes_view.itertuples(index=False)):
        sid = int(getattr(row, "pred_sigidx"))
        sub = work[work["_sigidx"] == sid]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("TOA(s)")
        pts = np.column_stack([
            sub["TOA(s)"].to_numpy(dtype=float),
            sub["Param1"].to_numpy(dtype=float),
            sub["Param5"].to_numpy(dtype=float),
        ])
        if len(pts) > 1:
            step = max(1, len(pts) // 18)
            coarse = pts[::step]
            if len(coarse) > 1:
                for start in range(len(coarse) - 1):
                    segments.append(coarse[start : start + 2])
                    seg_colors.append(cmap(i % 20))
        centers.append([
            (float(row.start_toa) + float(row.end_toa)) / 2.0,
            float(row.median_param1),
            float(row.median_param5),
        ])
        center_colors.append(cmap(i % 20))

    if segments:
        collection = Line3DCollection(segments, colors=seg_colors, linewidths=1.45, alpha=0.78)
        ax.add_collection3d(collection)
    if centers:
        centers = np.asarray(centers, dtype=float)
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=center_colors, s=62, alpha=0.98, edgecolors="white", linewidths=0.7)
        _set_compact_limits(ax, centers[:, 0], centers[:, 1], centers[:, 2])
    _setup_3d(ax, "Stage 3  Tracklet Construction")
    ax.text2D(0.02, 0.96, f"tracklets shown: {len(nodes_view):,} / {len(nodes):,}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_tracklets_2d(
    pdw: pd.DataFrame,
    sigidx: Iterable[int],
    nodes: pd.DataFrame,
    output_path: Path,
    max_tracklets: int = 120,
    dpi: int = 300,
) -> None:
    sigidx = np.asarray(sigidx, dtype=np.int64)
    work = pdw.copy()
    work["_sigidx"] = sigidx[: len(work)]
    nodes_view = nodes.nlargest(max_tracklets, "num_pulses") if len(nodes) > max_tracklets else nodes

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    cmap = plt.get_cmap("tab10")
    xs_all, ys_all = [], []
    for i, row in enumerate(nodes_view.itertuples(index=False)):
        sid = int(getattr(row, "pred_sigidx"))
        sub = work[work["_sigidx"] == sid].sort_values("TOA(s)")
        if len(sub) == 0:
            continue
        step = max(1, len(sub) // 24)
        sub = sub.iloc[::step]
        color = cmap(i % 10)
        ax.plot(sub["TOA(s)"], sub["Param1"], color=color, linewidth=1.7, alpha=0.82)
        center_x = (float(row.start_toa) + float(row.end_toa)) / 2.0
        center_y = float(row.median_param1)
        ax.scatter([center_x], [center_y], c=[color], s=58, alpha=0.98, edgecolors="white", linewidths=0.8, zorder=4)
        xs_all.extend(sub["TOA(s)"].to_numpy(dtype=float).tolist())
        ys_all.extend(sub["Param1"].to_numpy(dtype=float).tolist())
    _setup_2d(ax, "Stage 3  Tracklet Construction")
    if xs_all:
        _set_compact_limits_2d(ax, xs_all, ys_all)
    ax.text(0.015, 0.96, f"tracklets shown: {len(nodes_view):,} / {len(nodes):,}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_tracklet_graph_3d(nodes: pd.DataFrame, edges: pd.DataFrame, output_path: Path, max_edges: int = 900, dpi: int = 300) -> None:
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _setup_3d(ax, "Stage 4  Tracklet Physics-Consistency Graph")

    if len(nodes) == 0:
        _finish(fig, output_path, dpi)
        return

    centers = pd.DataFrame({
        "pred_sigidx": nodes["pred_sigidx"].astype(int),
        "x": (nodes["start_toa"].astype(float) + nodes["end_toa"].astype(float)) / 2.0,
        "y": nodes["median_param1"].astype(float),
        "z": nodes["median_param5"].astype(float),
        "size": nodes["num_pulses"].astype(float),
    })
    id_to_point = {int(r.pred_sigidx): np.array([r.x, r.y, r.z], dtype=float) for r in centers.itertuples(index=False)}

    accepted = edges
    if len(accepted) and "accepted" in accepted.columns:
        accepted = accepted[accepted["accepted"].astype(bool)]
    if len(accepted) > max_edges:
        if "score" in accepted.columns:
            accepted = accepted.nsmallest(max_edges, "score")
        else:
            accepted = accepted.iloc[:max_edges]

    segs = []
    for row in accepted.itertuples(index=False):
        left = int(getattr(row, "src_sigidx", getattr(row, "left", getattr(row, "a", -1))))
        right = int(getattr(row, "dst_sigidx", getattr(row, "right", getattr(row, "b", -1))))
        if left in id_to_point and right in id_to_point:
            segs.append(np.vstack([id_to_point[left], id_to_point[right]]))
    if segs:
        ax.add_collection3d(Line3DCollection(segs, colors="#1d4ed8", linewidths=1.05, alpha=0.38))

    size = 34 + 88 * np.sqrt(centers["size"].to_numpy() / max(float(centers["size"].max()), 1.0))
    ax.scatter(centers["x"], centers["y"], centers["z"], c="#065f5b", s=size, alpha=0.98, edgecolors="white", linewidths=0.7)
    _set_compact_limits(ax, centers["x"], centers["y"], centers["z"])
    ax.text2D(0.02, 0.96, f"nodes: {len(nodes):,} | accepted edges shown: {len(segs):,}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def _component_lookup(components: pd.DataFrame) -> Dict[int, int]:
    comp_lookup: Dict[int, int] = {}
    for comp_id, row in enumerate(components.itertuples(index=False), start=1):
        for field in ("merged_sigidx", "members", "member_sigidx", "pred_sigidx_list"):
            if hasattr(row, field):
                raw = getattr(row, field)
                if isinstance(raw, str):
                    values = [int(v) for v in raw.replace("[", "").replace("]", "").replace(",", " ").split() if v.strip().lstrip("-").isdigit()]
                else:
                    try:
                        values = [int(v) for v in raw]
                    except TypeError:
                        values = []
                for v in values:
                    comp_lookup[v] = comp_id
                break
        if hasattr(row, "component_id") and hasattr(row, "pred_sigidx"):
            comp_lookup[int(getattr(row, "pred_sigidx"))] = int(getattr(row, "component_id"))
    return comp_lookup


def plot_tracklet_graph_2d(nodes: pd.DataFrame, edges: pd.DataFrame, output_path: Path, max_edges: int = 900, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    _setup_2d(ax, "Stage 4  Tracklet Physics-Consistency Graph")
    if len(nodes) == 0:
        _finish(fig, output_path, dpi)
        return

    centers = pd.DataFrame({
        "pred_sigidx": nodes["pred_sigidx"].astype(int),
        "x": (nodes["start_toa"].astype(float) + nodes["end_toa"].astype(float)) / 2.0,
        "y": nodes["median_param1"].astype(float),
        "size": nodes["num_pulses"].astype(float),
    })
    id_to_point = {int(r.pred_sigidx): (float(r.x), float(r.y)) for r in centers.itertuples(index=False)}
    accepted = edges
    if len(accepted) and "accepted" in accepted.columns:
        accepted = accepted[accepted["accepted"].astype(bool)]
    if len(accepted) > max_edges:
        accepted = accepted.nsmallest(max_edges, "distance") if "distance" in accepted.columns else accepted.iloc[:max_edges]

    shown_edges = 0
    for row in accepted.itertuples(index=False):
        left = int(getattr(row, "src_sigidx", getattr(row, "left", getattr(row, "a", -1))))
        right = int(getattr(row, "dst_sigidx", getattr(row, "right", getattr(row, "b", -1))))
        if left in id_to_point and right in id_to_point:
            x1, y1 = id_to_point[left]
            x2, y2 = id_to_point[right]
            ax.plot([x1, x2], [y1, y2], color="#1d4ed8", linewidth=1.05, alpha=0.30, zorder=1)
            shown_edges += 1

    size = 48 + 110 * np.sqrt(centers["size"].to_numpy() / max(float(centers["size"].max()), 1.0))
    ax.scatter(centers["x"], centers["y"], c="#065f5b", s=size, alpha=0.98, edgecolors="white", linewidths=0.8, zorder=3)
    _set_compact_limits_2d(ax, centers["x"], centers["y"])
    ax.text(0.015, 0.96, f"nodes: {len(nodes):,} | accepted edges shown: {shown_edges:,}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_components_3d(nodes: pd.DataFrame, components: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    _setup_3d(ax, "Stage 5  Connected Component Merging")

    if len(nodes) == 0:
        _finish(fig, output_path, dpi)
        return

    comp_lookup = _component_lookup(components)

    xs = (nodes["start_toa"].astype(float) + nodes["end_toa"].astype(float)) / 2.0
    ys = nodes["median_param1"].astype(float)
    zs = nodes["median_param5"].astype(float)
    comps = np.array([comp_lookup.get(int(v), 0) for v in nodes["pred_sigidx"]], dtype=int)
    colors = [plt.get_cmap("tab10")((int(v) - 1) % 10) if int(v) > 0 else (0.34, 0.38, 0.44, 0.86) for v in comps]
    size = 34 + 88 * np.sqrt(nodes["num_pulses"].astype(float).to_numpy() / max(float(nodes["num_pulses"].max()), 1.0))
    ax.scatter(xs, ys, zs, c=colors, s=size, alpha=0.98, edgecolors="white", linewidths=0.7)
    _set_compact_limits(ax, xs, ys, zs)
    ax.text2D(0.02, 0.96, f"components: {len(components):,} | tracklets: {len(nodes):,}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_components_2d(nodes: pd.DataFrame, components: pd.DataFrame, output_path: Path, dpi: int = 300) -> None:
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    _setup_2d(ax, "Stage 5  Connected Component Merging")
    if len(nodes) == 0:
        _finish(fig, output_path, dpi)
        return

    comp_lookup = _component_lookup(components)
    xs = (nodes["start_toa"].astype(float) + nodes["end_toa"].astype(float)) / 2.0
    ys = nodes["median_param1"].astype(float)
    comps = np.array([comp_lookup.get(int(v), 0) for v in nodes["pred_sigidx"]], dtype=int)
    colors = [plt.get_cmap("tab10")((int(v) - 1) % 10) if int(v) > 0 else (0.34, 0.38, 0.44, 0.86) for v in comps]
    size = 48 + 110 * np.sqrt(nodes["num_pulses"].astype(float).to_numpy() / max(float(nodes["num_pulses"].max()), 1.0))
    ax.scatter(xs, ys, c=colors, s=size, alpha=0.98, edgecolors="white", linewidths=0.8)
    _set_compact_limits_2d(ax, xs, ys)
    ax.text(0.015, 0.96, f"components: {len(components):,} | tracklets: {len(nodes):,}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_before_after_comparison(
    pdw: pd.DataFrame,
    before_sigidx: Iterable[int],
    after_sigidx: Iterable[int],
    output_path: Path,
    max_points: int = 12000,
    dpi: int = 300,
) -> None:
    before = np.asarray(before_sigidx, dtype=np.int64)
    after = np.asarray(after_sigidx, dtype=np.int64)
    work = pdw.copy()
    work["_before"] = before[: len(work)]
    work["_after"] = after[: len(work)]
    view = _sample_rows(work, max_points)

    fig = plt.figure(figsize=(16, 7.8), facecolor="white")
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
    for ax, col, title in [
        (axes[0], "_before", "Initial Sorting"),
        (axes[1], "_after", "Tracklet-Graph Enhanced Sorting"),
    ]:
        colors, positive = _sig_colors(view[col].to_numpy(dtype=np.int64))
        ax.scatter(view["TOA(s)"], view["Param1"], view["Param5"], c=colors, s=13, alpha=0.90, linewidths=0)
        _setup_3d(ax, title)
        _set_compact_limits(ax, view["TOA(s)"], view["Param1"], view["Param5"])
        ax.text2D(0.02, 0.95, f"batches: {len(positive)}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_before_after_comparison_2d(
    pdw: pd.DataFrame,
    before_sigidx: Iterable[int],
    after_sigidx: Iterable[int],
    output_path: Path,
    max_points: int = 12000,
    dpi: int = 300,
) -> None:
    before = np.asarray(before_sigidx, dtype=np.int64)
    after = np.asarray(after_sigidx, dtype=np.int64)
    work = pdw.copy()
    work["_before"] = before[: len(work)]
    work["_after"] = after[: len(work)]
    view = _sample_rows(work, max_points)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), facecolor="white")
    for ax, col, title in [
        (axes[0], "_before", "Initial Sorting"),
        (axes[1], "_after", "Tracklet-Graph Enhanced Sorting"),
    ]:
        colors, positive = _sig_colors(view[col].to_numpy(dtype=np.int64))
        ax.scatter(view["TOA(s)"], view["Param1"], c=colors, s=18, alpha=0.92, linewidths=0)
        _setup_2d(ax, title)
        _set_compact_limits_2d(ax, view["TOA(s)"], view["Param1"])
        ax.text(0.015, 0.96, f"batches: {len(positive)}", transform=ax.transAxes, fontsize=10, color="#344054")
    _finish(fig, output_path, dpi)


def plot_metric_comparison(metrics_before: Optional[Dict[str, float]], metrics_final: Optional[Dict[str, float]], output_path: Path, dpi: int = 300) -> None:
    metrics_before = metrics_before or {}
    metrics_final = metrics_final or {}
    keys = [
        ("sample_sort_acc", "Accuracy"),
        ("sample_extra_batch_rate", "Extra Batch"),
        ("sample_wrong_batch_rate", "Wrong Batch"),
        ("sample_signal_tracking_stability", "Tracking Stability"),
    ]
    before_vals = [float(metrics_before.get(k, np.nan)) for k, _ in keys]
    final_vals = [float(metrics_final.get(k, np.nan)) for k, _ in keys]
    labels = [label for _, label in keys]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6), facecolor="white")
    ax.bar(x - width / 2, before_vals, width, label="Before graph", color="#93c5fd")
    ax.bar(x + width / 2, final_vals, width, label="Final", color="#0f766e")
    ax.set_title("Sorting Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _finish(fig, output_path, dpi)


def render_sorting_visualizations(
    pdw: pd.DataFrame,
    seed_sigidx: Iterable[int],
    graph_sigidx: Iterable[int],
    final_sigidx: Iterable[int],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    components: pd.DataFrame,
    output_dir: Path,
    metrics_before: Optional[Dict[str, float]] = None,
    metrics_final: Optional[Dict[str, float]] = None,
    dpi: int = 300,
    max_points: int = 12000,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_raw_pdw_2d(pdw, output_dir / "stage1_raw_pdw_2d.png", max_points=max_points, dpi=dpi)
    plot_raw_pdw_3d(pdw, output_dir / "stage1_raw_pdw_3d.png", max_points=max_points, dpi=dpi)
    plot_sigidx_2d(pdw, seed_sigidx, output_dir / "stage2_initial_sort_2d.png", "Stage 2  Initial Sorting", max_points=max_points, dpi=dpi)
    plot_sigidx_3d(pdw, seed_sigidx, output_dir / "stage2_initial_sort_3d.png", "Stage 2  Initial Sorting", max_points=max_points, dpi=dpi)
    plot_tracklets_2d(pdw, seed_sigidx, nodes, output_dir / "stage3_tracklets_2d.png", dpi=dpi)
    plot_tracklets_3d(pdw, seed_sigidx, nodes, output_dir / "stage3_tracklets_3d.png", dpi=dpi)
    plot_tracklet_graph_2d(nodes, edges, output_dir / "stage4_tracklet_graph_2d.png", dpi=dpi)
    plot_tracklet_graph_3d(nodes, edges, output_dir / "stage4_tracklet_graph_3d.png", dpi=dpi)
    plot_components_2d(nodes, components, output_dir / "stage5_connected_components_2d.png", dpi=dpi)
    plot_components_3d(nodes, components, output_dir / "stage5_connected_components_3d.png", dpi=dpi)
    plot_sigidx_2d(pdw, final_sigidx, output_dir / "stage6_final_sort_2d.png", "Stage 6  Final Sorting Result", max_points=max_points, dpi=dpi)
    plot_sigidx_3d(pdw, final_sigidx, output_dir / "stage6_final_sort_3d.png", "Stage 6  Final Sorting Result", max_points=max_points, dpi=dpi)
    plot_before_after_comparison_2d(pdw, seed_sigidx, final_sigidx, output_dir / "comparison_initial_vs_final_2d.png", max_points=max_points, dpi=dpi)
    plot_before_after_comparison(pdw, seed_sigidx, final_sigidx, output_dir / "comparison_initial_vs_final.png", max_points=max_points, dpi=dpi)
    plot_before_after_comparison_2d(pdw, seed_sigidx, graph_sigidx, output_dir / "comparison_initial_vs_tracklet_graph_2d.png", max_points=max_points, dpi=dpi)
    plot_before_after_comparison(pdw, seed_sigidx, graph_sigidx, output_dir / "comparison_initial_vs_tracklet_graph.png", max_points=max_points, dpi=dpi)
    plot_metric_comparison(metrics_before, metrics_final, output_dir / "metric_comparison.png", dpi=dpi)




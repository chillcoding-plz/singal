#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best sorting pipeline with tracklet-graph sorting inserted before fusion.

Pipeline:
    PA-TSR/PA-TGR initial sorting
        -> tracklet physics-consistency graph sorting
        -> original Best agile/strict fusion and suppression
        -> final SigIdx evaluation

This script is sorting-only. It does not run recognition.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

import Best
import tracklet_graph_sort as graph

try:
    from visualization_utils import render_sorting_visualizations
except Exception as exc:  # matplotlib may be unavailable in a minimal environment.
    render_sorting_visualizations = None
    VISUALIZATION_IMPORT_ERROR = exc
else:
    VISUALIZATION_IMPORT_ERROR = None


def make_best_args(cli: argparse.Namespace) -> argparse.Namespace:
    args = Best.build_parser().parse_args([])
    args.sample = cli.sample
    if cli.sample == "custom":
        args.pdw_file = cli.pdw_file
        args.truth_file = cli.truth_file
        args.name = cli.name
    args.output_root = Path(cli.output_root)
    args.max_pulses = int(cli.max_pulses)
    args.seed = int(cli.seed)
    args.n_jobs = int(cli.n_jobs)
    args.window_seconds = float(cli.window_seconds)
    args.progress_every = int(cli.progress_every)
    args.sort_backend = str(cli.sort_backend)
    args.skip_recognition = True
    args.agile_fusion = str(cli.agile_fusion)
    args.skip_initial_sort = bool(cli.skip_initial_sort)
    args.visualize = bool(cli.visualize)
    args.visualize_output = Path(cli.visualize_output)
    args.visualize_dpi = int(cli.visualize_dpi)
    args.visualize_max_points = int(cli.visualize_max_points)
    return args


def graph_args_from_cli(cli: argparse.Namespace) -> argparse.Namespace:
    g = graph.build_parser().parse_args([])
    g.max_pulses = int(cli.max_pulses)
    g.min_node_pulses = int(cli.graph_min_node_pulses)
    g.knn = int(cli.graph_knn)
    g.link_thresh = float(cli.graph_link_thresh)
    g.hard_p1 = float(cli.graph_hard_p1)
    g.hard_p2 = float(cli.graph_hard_p2)
    g.hard_p4 = float(cli.graph_hard_p4)
    g.hard_p5 = float(cli.graph_hard_p5)
    g.hard_pri = float(cli.graph_hard_pri)
    g.min_component_pulses = int(cli.graph_min_component_pulses)
    g.min_component_batches = int(cli.graph_min_component_batches)
    g.min_component_purity_score = float(cli.graph_min_component_purity_score)
    g.max_component_p1_span = float(cli.graph_max_component_p1_span)
    g.max_component_p2_span = float(cli.graph_max_component_p2_span)
    g.max_component_p4_span = float(cli.graph_max_component_p4_span)
    g.max_component_p5_span_deg = float(cli.graph_max_component_p5_span_deg)
    g.max_component_pri_span_us = float(cli.graph_max_component_pri_span_us)
    g.sort_purity_threshold = float(cli.sort_purity_threshold)
    g.sort_min_target_fraction = float(cli.sort_min_target_fraction)
    g.sort_mix_fail_min_pulses = int(cli.sort_mix_fail_min_pulses)
    g.sort_chunk_seconds = float(cli.sort_chunk_seconds)
    return g


def pre_graph_file(spec: Best.RunSpec) -> Path:
    return spec.output_dir / f"{spec.name}_sort_before_tracklet_graph.txt"


def graph_file(spec: Best.RunSpec) -> Path:
    return spec.output_dir / f"{spec.name}_sort_after_tracklet_graph.txt"


def write_graph_sort(pdw: pd.DataFrame, sigidx, path: Path) -> None:
    graph.write_sort(path, pdw["TOA(s)"].to_numpy(dtype=float), sigidx)


def run_initial_sort(spec: Best.RunSpec, best_args: argparse.Namespace) -> Dict[str, object]:
    backend = best_args.sort_backend
    if backend == "auto":
        backend = "pa_tsr"
    sorter, run_fn = Best.import_sort_backend(backend)
    sort_args = Best.configure_sort_args(sorter.build_parser(), spec, best_args, backend)
    sort_args.output_file = pre_graph_file(spec)
    sort_args.report_json = spec.output_dir / f"{spec.name}_initial_sort_summary.json"
    sort_args.metrics_dir = spec.output_dir / "sort_metrics_initial"
    sort_args.window_report_csv = spec.output_dir / f"{spec.name}_initial_windows.csv"
    sort_args.edge_report_csv = spec.output_dir / f"{spec.name}_initial_edges.csv"
    print(f"[initial_sort] {spec.name}: backend={backend}, output={sort_args.output_file}")
    summary = run_fn(sort_args)
    return {"backend": backend, **dict(summary), "pre_graph_file": str(sort_args.output_file)}


def run_tracklet_graph(
    root: Path,
    spec: Best.RunSpec,
    gargs: argparse.Namespace,
) -> Dict[str, object]:
    pdw = graph.read_pdw(spec.pdw_file, int(gargs.max_pulses))
    seed = graph.read_sigidx(pre_graph_file(spec), int(gargs.max_pulses))
    if len(pdw) != len(seed):
        raise ValueError(f"{spec.name}: PDW rows ({len(pdw)}) and seed rows ({len(seed)}) differ")

    nodes = graph.summarize_tracklets(pdw, seed, int(gargs.min_node_pulses))
    nodes.to_csv(spec.output_dir / f"{spec.name}_tracklet_graph_nodes.csv", index=False, encoding="utf-8-sig")
    edges, scales = graph.build_edges(nodes, gargs)
    edges.to_csv(spec.output_dir / f"{spec.name}_tracklet_graph_edges.csv", index=False, encoding="utf-8-sig")
    graph_sigidx, components = graph.merge_by_graph(seed, nodes, edges, gargs)
    components.to_csv(spec.output_dir / f"{spec.name}_tracklet_graph_components.csv", index=False, encoding="utf-8-sig")
    write_graph_sort(pdw, graph_sigidx, graph_file(spec))
    write_graph_sort(pdw, graph_sigidx, spec.sort_file)

    if spec.truth_file is not None and spec.truth_file.exists():
        metrics = graph.evaluate(
            root,
            pdw,
            spec.truth_file,
            graph_sigidx,
            gargs,
            spec.output_dir / "tracklet_graph_metrics" / f"{spec.name}_tracklet_graph",
        )
    else:
        metrics = {"skipped": True, "reason": "truth file not found"}
    summary = {
        "method": "Tracklet Physics-Consistency Graph Sorting before Best fusion",
        "seed_file": str(pre_graph_file(spec)),
        "output_file": str(graph_file(spec)),
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "num_accepted_edges": int(edges["accepted"].sum()) if len(edges) else 0,
        "num_components": int(len(components)),
        "num_batches_before": int(len(set(int(v) for v in seed if int(v) > 0))),
        "num_batches_after": int(len(set(int(v) for v in graph_sigidx if int(v) > 0))),
        "num_changed_pulses": int((seed != graph_sigidx).sum()),
        "scales": scales,
        "metrics_before_best_fusion": metrics,
    }
    (spec.output_dir / f"{spec.name}_tracklet_graph_summary.json").write_text(
        json.dumps(graph.json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[tracklet_graph] {spec.name}: nodes={summary['num_nodes']}, "
        f"components={summary['num_components']}, batches {summary['num_batches_before']} -> {summary['num_batches_after']}"
    )
    return summary


def run_one(root: Path, spec: Best.RunSpec, best_args: argparse.Namespace, gargs: argparse.Namespace) -> Dict[str, object]:
    if not spec.pdw_file.exists():
        raise FileNotFoundError(f"Missing PDW file: {spec.pdw_file}")
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    if bool(getattr(best_args, "skip_initial_sort", False)):
        if not pre_graph_file(spec).exists():
            raise FileNotFoundError(f"Missing reusable initial sort file: {pre_graph_file(spec)}")
        initial_summary = {"skipped": True, "pre_graph_file": str(pre_graph_file(spec))}
        print(f"[initial_sort] skipped: {pre_graph_file(spec)}")
    else:
        initial_summary = run_initial_sort(spec, best_args)
    graph_summary = run_tracklet_graph(root, spec, gargs)
    print(f"[best_fusion] {spec.name}: input={spec.sort_file}")
    fusion_summary = Best.apply_agile_band_fusion(spec, best_args)

    pdw = graph.read_pdw(spec.pdw_file, int(gargs.max_pulses))
    final_sigidx = graph.read_sigidx(spec.sort_file, int(gargs.max_pulses))
    if spec.truth_file is not None and spec.truth_file.exists():
        final_metrics = graph.evaluate(
            root,
            pdw,
            spec.truth_file,
            final_sigidx,
            gargs,
            spec.output_dir / "final_metrics" / f"{spec.name}_final",
        )
    else:
        final_metrics = {"skipped": True, "reason": "truth file not found"}

    if bool(getattr(best_args, "visualize", False)):
        if render_sorting_visualizations is None:
            raise RuntimeError(f"Visualization is unavailable: {VISUALIZATION_IMPORT_ERROR}")
        seed_sigidx = graph.read_sigidx(pre_graph_file(spec), int(gargs.max_pulses))
        graph_sigidx = graph.read_sigidx(graph_file(spec), int(gargs.max_pulses))
        nodes = pd.read_csv(spec.output_dir / f"{spec.name}_tracklet_graph_nodes.csv")
        edges = pd.read_csv(spec.output_dir / f"{spec.name}_tracklet_graph_edges.csv")
        components = pd.read_csv(spec.output_dir / f"{spec.name}_tracklet_graph_components.csv")
        visual_dir = Path(best_args.visualize_output) / spec.name
        render_sorting_visualizations(
            pdw=pdw,
            seed_sigidx=seed_sigidx,
            graph_sigidx=graph_sigidx,
            final_sigidx=final_sigidx,
            nodes=nodes,
            edges=edges,
            components=components,
            output_dir=visual_dir,
            metrics_before=graph_summary.get("metrics_before_best_fusion", {}),
            metrics_final=final_metrics,
            dpi=int(getattr(best_args, "visualize_dpi", 300)),
            max_points=int(getattr(best_args, "visualize_max_points", 50000)),
        )
        print(f"[visualize] {spec.name}: saved figures to {visual_dir}")
    summary = {
        "sample": spec.name,
        "method": "Best + front-inserted tracklet graph sorting",
        "pdw_file": str(spec.pdw_file),
        "final_sort_file": str(spec.sort_file),
        "initial_sort": initial_summary,
        "tracklet_graph": graph_summary,
        "best_fusion_after_graph": fusion_summary,
        "final_metrics": final_metrics,
    }
    out = spec.output_dir / f"{spec.name}_front_graph_pipeline_summary.json"
    out.write_text(json.dumps(graph.json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[done] {spec.name}: sort={final_metrics.get('sample_sort_acc', float('nan')):.4f}, "
        f"extra={final_metrics.get('sample_extra_batch_rate', float('nan')):.4f}, "
        f"wrong={final_metrics.get('sample_wrong_batch_rate', float('nan')):.4f}, "
        f"track={final_metrics.get('sample_signal_tracking_stability', float('nan')):.4f}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sorting-only Best pipeline with tracklet graph inserted before fusion.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--sample", choices=["sample1", "sample2", "all", "custom"], default="all")
    parser.add_argument("--pdw_file", type=Path, default=None)
    parser.add_argument("--truth_file", type=Path, default=None)
    parser.add_argument("--name", type=str, default="ui_sample")
    parser.add_argument("--output_root", type=Path, default=Path("outputs_best_front_tracklet_graph"))
    parser.add_argument("--sort_backend", choices=["auto", "pa_tgr", "pa_tsr"], default="auto")
    parser.add_argument("--agile_fusion", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--skip_initial_sort", action="store_true")
    parser.add_argument("--max_pulses", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--window_seconds", type=float, default=0.1)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--visualize", action="store_true", help="Save stage-wise 3D figures for PPT.")
    parser.add_argument("--visualize_output", type=Path, default=Path("outputs_visualization"))
    parser.add_argument("--visualize_dpi", type=int, default=300)
    parser.add_argument("--visualize_max_points", type=int, default=12000)


    parser.add_argument("--graph_min_node_pulses", type=int, default=500)
    parser.add_argument("--graph_knn", type=int, default=18)
    parser.add_argument("--graph_link_thresh", type=float, default=0.35)
    parser.add_argument("--graph_hard_p1", type=float, default=0.25)
    parser.add_argument("--graph_hard_p2", type=float, default=0.15)
    parser.add_argument("--graph_hard_p4", type=float, default=0.40)
    parser.add_argument("--graph_hard_p5", type=float, default=0.15)
    parser.add_argument("--graph_hard_pri", type=float, default=0.60)
    parser.add_argument("--graph_min_component_batches", type=int, default=2)
    parser.add_argument("--graph_min_component_pulses", type=int, default=2500)
    parser.add_argument("--graph_min_component_purity_score", type=float, default=0.85)
    parser.add_argument("--graph_max_component_p1_span", type=float, default=20.0)
    parser.add_argument("--graph_max_component_p2_span", type=float, default=0.10)
    parser.add_argument("--graph_max_component_p4_span", type=float, default=4.0)
    parser.add_argument("--graph_max_component_p5_span_deg", type=float, default=0.08)
    parser.add_argument("--graph_max_component_pri_span_us", type=float, default=30.0)


    #
    # parser.add_argument("--graph_min_node_pulses", type=int, default=300)
    # parser.add_argument("--graph_knn", type=int, default=30)
    # parser.add_argument("--graph_link_thresh", type=float, default=0.50)
    # parser.add_argument("--graph_hard_p1", type=float, default=0.40)
    # parser.add_argument("--graph_hard_p2", type=float, default=0.25)
    # parser.add_argument("--graph_hard_p4", type=float, default=0.60)
    # parser.add_argument("--graph_hard_p5", type=float, default=0.25)
    # parser.add_argument("--graph_hard_pri", type=float, default=0.90)
    #
    # parser.add_argument("--graph_min_component_batches", type=int, default=2)
    # parser.add_argument("--graph_min_component_pulses", type=int, default=1200)
    # parser.add_argument("--graph_min_component_purity_score", type=float, default=1.00)
    # parser.add_argument("--graph_max_component_p1_span", type=float, default=40.0)
    # parser.add_argument("--graph_max_component_p2_span", type=float, default=0.20)
    # parser.add_argument("--graph_max_component_p4_span", type=float, default=8.0)
    # parser.add_argument("--graph_max_component_p5_span_deg", type=float, default=0.15)
    # parser.add_argument("--graph_max_component_pri_span_us", type=float, default=60.0)

    parser.add_argument("--sort_purity_threshold", type=float, default=0.90)
    parser.add_argument("--sort_min_target_fraction", type=float, default=0.10)
    parser.add_argument("--sort_mix_fail_min_pulses", type=int, default=150)
    parser.add_argument("--sort_chunk_seconds", type=float, default=0.2)
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    if not cli.output_root.is_absolute():
        cli.output_root = (Path.cwd() / cli.output_root).resolve()
    root = cli.root.resolve()
    if not cli.visualize_output.is_absolute():
        cli.visualize_output = (root / cli.visualize_output).resolve()
    for extra in [root, root / "分选", Path(__file__).resolve().parent]:
        text = str(extra)
        if text not in sys.path:
            sys.path.insert(0, text)
    best_args = make_best_args(cli)
    gargs = graph_args_from_cli(cli)

    cwd = Path.cwd().resolve()
    try:
        import os

        os.chdir(root)
        specs: List[Best.RunSpec] = Best.resolve_run_specs(best_args)
        if cli.sample in {"sample1", "sample2"}:
            specs = [spec for spec in specs if spec.name == cli.sample]
        summaries = [run_one(root, spec, best_args, gargs) for spec in specs]
    finally:
        import os

        os.chdir(cwd)

    cli.output_root.mkdir(parents=True, exist_ok=True)
    (cli.output_root / "front_tracklet_graph_all_summary.json").write_text(
        json.dumps(graph.json_safe(summaries), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()



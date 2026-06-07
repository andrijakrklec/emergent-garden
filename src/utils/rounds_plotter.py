"""@file rounds_plotter.py
@brief Standalone, fully-customizable re-plotter for a SimLogger ``rounds.csv``.

Reads a single ``rounds.csv`` (written by @ref src.utils.sim_logger.SimLogger) and
renders the same round-level metrics that land in ``global_metrics.png`` — but
each metric becomes its **own** figure instead of a 2x3 panel, and every cosmetic
property is overridable: titles, axis labels, axis tick granularity, colors,
y-limits, theme, figure size, DPI, fonts, event markers and output format.

Unlike SimLogger (which plots from its in-memory history during a live run), this
module is post-hoc: point it at any ``rounds.csv`` on disk and re-style the plots
for a thesis figure without re-running the simulation.

Two ways to use it.

CLI
---
    # default 6 metrics, one PNG each, dark theme, alongside the CSV in ./plots
    python -m src.utils.rounds_plotter logs/run_.../rounds.csv

    # rename axes / title, set tick granularity, light theme, save as PDF
    python -m src.utils.rounds_plotter logs/run_.../rounds.csv \
        --metrics avg_local_loss,avg_direction_precision \
        --title  avg_local_loss="Avg local loss" \
        --ylabel avg_local_loss="loss function" \
        --xlabel "round" --xstep 5 --ystep avg_local_loss=0.05 \
        --theme light --format pdf --out figures/

    python -m src.utils.rounds_plotter rounds.csv --all       # every column in the CSV

    # comparison: overlay several runs (folders ok), one line each
    python -m src.utils.rounds_plotter \
        logs/cfl_on logs/cfl_off \
        --labels "CFL on,CFL off" --colors "#5599dd,#e05555" \
        --metrics avg_local_loss,avg_direction_precision --out figures/

    # average several runs into ONE mean line (no comparison)
    python -m src.utils.rounds_plotter \
        logs/cells/cfl_on__emergent_on_run01 logs/cells/cfl_on__emergent_on_run02 \
        --avg --labels "CFL on (mean)" --metrics avg_local_loss --out figures/

    # averaged-group comparison: average each cell's repeated runs into one line
    python -m src.utils.rounds_plotter \
        --group "CFL on=logs/cells/cfl_on__emergent_on_run01,logs/cells/cfl_on__emergent_on_run02" \
        --group "CFL off=logs/cells/cfl_off__emergent_on_run01,logs/cells/cfl_off__emergent_on_run02" \
        --colors "#5599dd,#e05555" --metrics avg_local_loss --out figures/

API
---
    from src.utils.rounds_plotter import RoundsPlotter, PlotStyle, MetricSpec

    plotter = RoundsPlotter(
        "logs/run_.../rounds.csv",
        style=PlotStyle(theme="light", x_step=5, figsize=(8, 5)),
    )
    # tweak one metric's cosmetics, then render just that one
    plotter.registry["avg_local_loss"].title  = "Avg local loss"
    plotter.registry["avg_local_loss"].ylabel = "loss function"
    plotter.registry["avg_local_loss"].y_step = 0.05
    plotter.plot(["avg_local_loss"], out_dir="figures")

    plotter.plot()              # the default SimLogger global-metric set
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Vertical-marker colors for the per-round ``event`` column, matching SimLogger.
# explosion never reaches rounds.csv (it is logged separately), but is kept here
# so a hand-edited CSV carrying it still renders correctly.
EVENT_COLORS = {"split": "#ffaa00", "merge": "#00aaff", "explosion": "#ff3333"}

# Named color/spine/grid palettes. "dark" reproduces SimLogger's look; "light"
# is a print-friendly variant for the thesis PDF.
THEMES: Dict[str, Dict[str, str]] = {
    "dark": {
        "fig_bg":     "#1a1a2e",
        "ax_bg":      "#16213e",
        "fg":         "#ffffff",
        "spine":      "#444466",
        "grid":       "#2e2e4d",
        "legend_bg":  "#1a1a2e",
    },
    "light": {
        "fig_bg":     "#ffffff",
        "ax_bg":      "#ffffff",
        "fg":         "#1a1a1a",
        "spine":      "#999999",
        "grid":       "#dddddd",
        "legend_bg":  "#ffffff",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricSpec:
    """@brief Per-metric appearance: which CSV column, and how to label/scale it.

    @param column Column name in ``rounds.csv``.
    @param title Figure title (override for any language / wording).
    @param ylabel Y-axis label (may contain ``\\n`` for multi-line).
    @param color Line color (any matplotlib color).
    @param ylim Optional (low, high) y-limits; @c None auto-scales.
    @param y_step Optional y tick spacing (major-locator granularity).
    @param integer_y Force integer y ticks (e.g. cluster / particle counts).
    @param xlabel Optional per-metric x-label override (else @ref PlotStyle.xlabel).
    @param filename Optional output filename stem (else the column name).
    """
    column: str
    title: str
    ylabel: str
    color: str = "#5599dd"
    ylim: Optional[Tuple[float, float]] = None
    y_step: Optional[float] = None
    integer_y: bool = False
    xlabel: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class PlotStyle:
    """@brief Figure-wide cosmetic settings shared by every metric plot.

    @param theme "dark" (SimLogger look) or "light" (print-friendly).
    @param figsize (width, height) in inches.
    @param dpi Output resolution.
    @param xlabel Default x-axis label (per-metric @ref MetricSpec.xlabel wins).
    @param x_step Optional x tick spacing (round granularity).
    @param linewidth Line width.
    @param marker Marker style (e.g. "" none, "o", ".").
    @param markersize Marker size.
    @param grid Draw a background grid.
    @param grid_alpha Grid opacity.
    @param title_fontsize Title font size.
    @param label_fontsize Axis-label font size.
    @param label_weight Axis-label font weight ("normal" or "bold").
    @param tick_fontsize Tick-label font size.
    @param legend_fontsize Legend font size.
    @param mark_events Draw split/merge vertical markers from the ``event`` column.
    @param show_legend Draw a legend (metric line + present event markers).
    @param suptitle Optional descriptive line drawn above every title (e.g.
        "CFL: ON | Emergent: OFF"); SimLogger uses this for its config label.
    @param output_format File extension/format: png, pdf, svg, ...
    @param show Open an interactive window instead of (in addition to) saving.
    @param export_csv Also write the plotted data as a sibling ``.csv`` next to
        each figure (round + one column per plotted line).
    """
    theme: str = "dark"
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 130
    xlabel: str = "Round"
    x_step: Optional[float] = None
    linewidth: float = 1.8
    marker: str = ""
    markersize: float = 4.0
    grid: bool = True
    grid_alpha: float = 0.3
    title_fontsize: float = 13
    label_fontsize: float = 11
    label_weight: str = "normal"
    tick_fontsize: float = 9
    legend_fontsize: float = 8
    mark_events: bool = True
    show_legend: bool = True
    suptitle: Optional[str] = None
    output_format: str = "png"
    show: bool = False
    export_csv: bool = False


def default_metric_registry() -> Dict[str, MetricSpec]:
    """@brief Built-in @ref MetricSpec for every column SimLogger writes to rounds.csv.

    The first six entries mirror, one-for-one, the panels of SimLogger's
    ``global_metrics.png``. The remainder are columns SimLogger logs but does not
    plot globally — exposed here so ``--all`` / ``--metrics`` can chart them too.

    @return dict mapping column name -> default MetricSpec.
    """
    return {
        # ── the six panels of global_metrics.png ─────────────────────────────
        "cluster_purity": MetricSpec(
            "cluster_purity", "Cluster Purity",
            "Purity\n(cluster_id vs latent true label)", "#ffaaff", ylim=(0, 1.05)),
        "num_clusters": MetricSpec(
            "num_clusters", "Number of Clusters", "# Clusters", "#5599dd",
            integer_y=True),
        "total_migrations": MetricSpec(
            "total_migrations", "Migrations per Round", "# Particles migrated",
            "#ffaa00"),
        "migration_rate": MetricSpec(
            "migration_rate", "Migration Rate",
            "Migrations / particle (fraction of swarm)", "#ffaa00"),
        "avg_local_loss": MetricSpec(
            "avg_local_loss", "Average Local Loss", "Avg Local Loss", "#e05555",
            ylim=(0, 0.5)),
        "avg_direction_precision": MetricSpec(
            "avg_direction_precision", "Average Direction Precision",
            "Avg Direction Precision\n(model vs true target dir)", "#33cccc",
            ylim=(0, 1.05)),
        "avg_spatial_precision": MetricSpec(
            "avg_spatial_precision", "Average Spatial Precision",
            "Avg Spatial Precision\n(1 - dist(particle, target) / sim_diag)",
            "#dd7733", ylim=(0, 1.05)),
        # ── extra columns logged but not in the global figure ────────────────
        "inertia": MetricSpec(
            "inertia", "KMeans Inertia", "Inertia (within-cluster SSE)", "#9955cc"),
        "num_particles": MetricSpec(
            "num_particles", "Number of Particles", "# Particles", "#aaaaaa",
            integer_y=True),
        "avg_confidence": MetricSpec(
            "avg_confidence", "Average Confidence", "Avg Confidence", "#55bb77",
            ylim=(0, 1.05)),
        "avg_peer_alignment": MetricSpec(
            "avg_peer_alignment", "Average Peer Alignment", "Avg Peer Alignment",
            "#33cccc"),
        "avg_obstacle_pressure": MetricSpec(
            "avg_obstacle_pressure", "Average Obstacle Pressure",
            "Avg Obstacle Pressure", "#dd7733"),
        "avg_drift_velocity": MetricSpec(
            "avg_drift_velocity", "Average Drift Velocity", "Avg Drift Velocity",
            "#e05555"),
        "avg_rounds_stable": MetricSpec(
            "avg_rounds_stable", "Average Rounds Stable", "Avg Rounds Stable",
            "#ddaa33"),
    }


# Default selection = the six panels of SimLogger's global_metrics.png, in order.
DEFAULT_METRIC_COLUMNS: List[str] = [
    "cluster_purity", "num_clusters", "total_migrations",
    "avg_local_loss", "avg_direction_precision", "avg_spatial_precision",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_float(raw: object) -> float:
    """@brief Parse a CSV cell to float; blank / unparsable -> NaN.

    SimLogger writes empty strings for metrics that are N/A in a given run.
    Those become NaN here and are filtered out per-series, exactly as SimLogger
    filters @c None.
    """
    if raw is None:
        return float("nan")
    s = str(raw).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _is_nan(v: float) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _write_data_csv(path: str, header: List[str], rows: List[list]) -> None:
    """@brief Write the data behind a plot to ``path`` (header row + data rows)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _resolve_csv(path: str) -> str:
    """@brief Map a folder to the ``rounds.csv`` inside it; leave file paths as-is.

    Lets callers pass a run *folder* (e.g. ``logs/run_.../``) instead of the full
    ``.../rounds.csv`` path.
    """
    return os.path.join(path, "rounds.csv") if os.path.isdir(path) else path


# Markdown-style **bold** spans -> matplotlib mathtext. Text outside the spans is
# left as ordinary text (its font/diacritics untouched); only the bold span is
# rendered via mathtext, so it stays ASCII-friendly. No "**" -> returned as-is.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


def _bold_markup(text: Optional[str]) -> Optional[str]:
    """@brief Convert ``**bold**`` spans in a label/title to mathtext bold.

    Inside a span, characters that are special to mathtext are escaped and spaces
    are turned into ``\\ `` (mathtext collapses raw spaces). Use for ASCII text;
    accented letters inside a bold span may not render in every matplotlib build.

    @param text Any label/title string (or None).
    @return The string with bold spans converted, or the original if it has none.
    """
    if not text or "**" not in text:
        return text

    def _sub(m: "re.Match") -> str:
        inner = m.group(1)
        for ch in "\\_^${}#%":          # backslash first
            inner = inner.replace(ch, "\\" + ch)
        inner = inner.replace(" ", r"\ ")
        return r"$\mathbf{" + inner + r"}$"

    return _BOLD_RE.sub(_sub, text)


# ──────────────────────────────────────────────────────────────────────────────
# RoundsPlotter
# ──────────────────────────────────────────────────────────────────────────────

class RoundsPlotter:
    """@brief Loads a ``rounds.csv`` and renders one customizable figure per metric."""

    def __init__(
        self,
        csv_path: str,
        style: Optional[PlotStyle] = None,
        registry: Optional[Dict[str, MetricSpec]] = None,
    ) -> None:
        """@brief Read and parse the CSV up front.

        @param csv_path Path to a SimLogger ``rounds.csv`` (or the folder that
            contains it — a directory is resolved to ``<dir>/rounds.csv``).
        @param style Figure-wide @ref PlotStyle (defaults applied if omitted).
        @param registry Column -> @ref MetricSpec map (defaults to the built-in set).
        """
        self.csv_path = _resolve_csv(csv_path)
        csv_path = self.csv_path
        self.style = style or PlotStyle()
        self.registry = registry if registry is not None else default_metric_registry()
        self.rounds, self.data, self.events = self._load(csv_path)

    # ── loading ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load(csv_path: str) -> Tuple[List[int], Dict[str, List[float]], List[Tuple[int, str]]]:
        """@brief Parse rounds.csv into x-rounds, per-column float series, and events.

        @return (rounds, data, events) where @c data maps each numeric column to a
            list aligned with @c rounds (NaN where blank), and @c events is a list
            of (round, type) drawn from the per-row ``event`` column.
        """
        # utf-8-sig tolerates a BOM (Excel / some editors add one), which would
        # otherwise corrupt the first header name ("round" -> "﻿round").
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            value_cols = [f for f in fieldnames if f not in ("round", "event")]
            rounds: List[int] = []
            data: Dict[str, List[float]] = {c: [] for c in value_cols}
            events: List[Tuple[int, str]] = []
            for row in reader:
                try:
                    rnd = int(float(row["round"]))
                except (KeyError, TypeError, ValueError):
                    continue
                rounds.append(rnd)
                for col in value_cols:
                    data[col].append(_to_float(row.get(col, "")))
                ev = (row.get("event") or "").strip().lower()
                if ev in EVENT_COLORS:
                    events.append((rnd, ev))

        # Derived column: migrations normalized by swarm size (fraction of
        # particles that migrated that round) — comparable across runs/cells of
        # different particle counts. NaN where inputs are missing or N == 0.
        if "total_migrations" in data and "num_particles" in data:
            data["migration_rate"] = [
                (m / n) if (not _is_nan(m) and not _is_nan(n) and n) else float("nan")
                for m, n in zip(data["total_migrations"], data["num_particles"])
            ]
        return rounds, data, events

    def _series(self, column: str) -> Tuple[List[int], List[float]]:
        """@brief (rounds, values) for a column with NaN rows dropped.

        @throws KeyError if the column is absent from the CSV.
        """
        if column not in self.data:
            raise KeyError(
                f"Column {column!r} not in {self.csv_path} "
                f"(available: {', '.join(sorted(self.data))})"
            )
        xs, ys = [], []
        for r, v in zip(self.rounds, self.data[column]):
            if _is_nan(v):
                continue
            xs.append(r)
            ys.append(v)
        return xs, ys

    # ── styling internals ───────────────────────────────────────────────────────

    def _plt(self):
        """@brief Import pyplot, forcing the Agg backend unless an interactive show was requested."""
        import matplotlib
        if not self.style.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt

    def _theme(self) -> Dict[str, str]:
        return THEMES.get(self.style.theme, THEMES["dark"])

    def _style_axes(self, fig, ax) -> None:
        s, t = self.style, self._theme()
        fig.patch.set_facecolor(t["fig_bg"])
        ax.set_facecolor(t["ax_bg"])
        ax.tick_params(colors=t["fg"], labelsize=s.tick_fontsize)
        ax.xaxis.label.set_color(t["fg"])
        ax.yaxis.label.set_color(t["fg"])
        ax.title.set_color(t["fg"])
        for spine in ax.spines.values():
            spine.set_edgecolor(t["spine"])
        if s.grid:
            ax.grid(True, color=t["grid"], alpha=s.grid_alpha, linewidth=0.7)
            ax.set_axisbelow(True)

    def _mark_events(self, ax) -> List[str]:
        """@brief Draw split/merge vertical markers; return the event types present."""
        present: List[str] = []
        round_set = set(self.rounds)
        for rnd, ev in self.events:
            if rnd not in round_set:
                continue
            ax.axvline(rnd, color=EVENT_COLORS[ev], linewidth=1.2,
                       linestyle="--", alpha=0.7)
            if ev not in present:
                present.append(ev)
        return present

    @staticmethod
    def _event_patches(present: List[str]):
        from matplotlib.lines import Line2D
        return [
            Line2D([0], [0], color=EVENT_COLORS[ev], lw=1.5, ls="--", label=ev)
            for ev in present
        ]

    # ── public plotting API ──────────────────────────────────────────────────────

    def get_spec(self, column: str) -> MetricSpec:
        """@brief Registry spec for a column, or a sensible default for unknown columns."""
        if column in self.registry:
            return self.registry[column]
        return MetricSpec(column, column, column)

    def plot_metric(self, spec: MetricSpec, out_dir: str) -> Optional[str]:
        """@brief Render and save one metric as its own figure.

        @param spec The @ref MetricSpec describing column + cosmetics.
        @param out_dir Directory to write into (created if missing).
        @return Saved file path, or @c None if the column has no plottable data.
        """
        s = self.style
        t = self._theme()
        try:
            xs, ys = self._series(spec.column)
        except KeyError as exc:
            print(f"  [skip] {spec.column}: {exc}")
            return None
        if not xs:
            print(f"  [skip] {spec.column}: no data (all blank/NaN)")
            return None

        plt = self._plt()
        from matplotlib.ticker import MaxNLocator, MultipleLocator

        fig, ax = plt.subplots(figsize=s.figsize)
        self._style_axes(fig, ax)

        line, = ax.plot(
            xs, ys, color=spec.color, linewidth=s.linewidth,
            marker=s.marker, markersize=s.markersize, label=spec.title,
        )

        present = self._mark_events(ax) if s.mark_events else []

        ax.set_xlabel(_bold_markup(spec.xlabel or s.xlabel), fontsize=s.label_fontsize, fontweight=s.label_weight)
        ax.set_ylabel(_bold_markup(spec.ylabel), fontsize=s.label_fontsize, fontweight=s.label_weight)
        ax.set_title(_bold_markup(spec.title), fontsize=s.title_fontsize, fontweight="bold")

        if spec.ylim is not None:
            ax.set_ylim(*spec.ylim)
        if spec.integer_y:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        elif spec.y_step:
            ax.yaxis.set_major_locator(MultipleLocator(spec.y_step))
        if s.x_step:
            ax.xaxis.set_major_locator(MultipleLocator(s.x_step))

        if s.suptitle:
            fig.suptitle(_bold_markup(s.suptitle), color=t["fg"], fontsize=s.title_fontsize - 1)

        if s.show_legend:
            ax.legend(
                handles=[line] + self._event_patches(present),
                fontsize=s.legend_fontsize, facecolor=t["legend_bg"],
                labelcolor=t["fg"], edgecolor=t["spine"],
            )

        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        stem = spec.filename or spec.column
        path = os.path.join(out_dir, f"{stem}.{s.output_format}")
        fig.savefig(path, dpi=s.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        if s.show:
            plt.show()
        plt.close(fig)
        print(f"  [ok] {os.path.basename(path)}")
        if s.export_csv:
            csv_path = os.path.join(out_dir, f"{stem}.csv")
            _write_data_csv(csv_path, ["round", spec.column], list(zip(xs, ys)))
            print(f"  [ok] {os.path.basename(csv_path)}")
        return path

    def plot(
        self,
        columns: Optional[List[str]] = None,
        out_dir: Optional[str] = None,
    ) -> List[str]:
        """@brief Render the chosen columns, one figure each.

        @param columns Columns to plot; defaults to the six SimLogger global
            metrics (those present in this CSV).
        @param out_dir Output directory; defaults to ``<csv_dir>/plots``.
        @return List of saved file paths.
        """
        if columns is None:
            columns = [c for c in DEFAULT_METRIC_COLUMNS if c in self.data]
        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(os.path.abspath(self.csv_path)), "plots")

        print(f"[RoundsPlotter] {self.csv_path} -> {out_dir}  "
              f"({len(self.rounds)} rounds, theme={self.style.theme})")
        saved: List[str] = []
        for col in columns:
            path = self.plot_metric(self.get_spec(col), out_dir)
            if path:
                saved.append(path)
        print(f"[RoundsPlotter] Done. {len(saved)} figure(s) saved to: {out_dir}")
        return saved


# ──────────────────────────────────────────────────────────────────────────────
# ComparisonPlotter — overlay the same metric from several runs
# ──────────────────────────────────────────────────────────────────────────────

# Per-series line colours when none is given explicitly (one per run).
SERIES_PALETTE = [
    "#e05555", "#5599dd", "#55bb77", "#ddaa33",
    "#9955cc", "#33cccc", "#dd7733", "#aaaaaa",
]


class ComparisonPlotter:
    """@brief Overlay one metric from several runs / averaged groups on one figure.

    One figure is produced per metric; within it, each *group* is a separate
    labeled line. A group is one or more run folders: with several folders their
    per-round metrics are **averaged** (NaN-aware, aligned by round number), so a
    cell of repeated runs collapses to a single mean line. Built for cross-cell
    comparison (e.g. the CFL on/off × emergent on/off ablation, each repeated N
    times). Cosmetics reuse the same @ref MetricSpec (title, y-label, y-limits,
    tick granularity) and @ref PlotStyle as @ref RoundsPlotter; line color is one
    per group. Event markers are not drawn (ambiguous across runs). Output files
    are prefixed ``compare_``.
    """

    def __init__(
        self,
        groups,
        style: Optional[PlotStyle] = None,
        registry: Optional[Dict[str, MetricSpec]] = None,
        palette: Optional[List[str]] = None,
    ) -> None:
        """@brief Load every group's run CSVs.

        @param groups Iterable of group specs. Each group is a dict
            ``{"label":.., "paths":[..], "color":..}`` or a tuple
            ``(label, paths)`` / ``(label, paths, color)`` where @c paths is a
            list of run folders (or ``rounds.csv`` paths). A single path may be
            given instead of a list. A blank label defaults to the common run
            folder name.
        @param style Shared @ref PlotStyle.
        @param registry Shared column -> @ref MetricSpec map.
        @param palette Per-group color cycle (defaults to @ref SERIES_PALETTE).
        """
        self.style = style or PlotStyle()
        self.registry = registry if registry is not None else default_metric_registry()
        self.palette = palette or SERIES_PALETTE
        self.series: List[Dict[str, Any]] = []
        for g in groups:
            if isinstance(g, dict):
                label, paths, color = g.get("label"), g.get("paths"), g.get("color")
            elif isinstance(g, (tuple, list)):
                label = g[0]
                paths = g[1] if len(g) > 1 else None
                color = g[2] if len(g) > 2 else None
            else:                                   # a bare path string
                label, paths, color = None, g, None
            if isinstance(paths, str):
                paths = [paths]
            plotters = [RoundsPlotter(p, self.style, self.registry) for p in (paths or [])]
            self.series.append({
                "label": label or self._auto_label(plotters),
                "color": color,
                "plotters": plotters,
            })

    @staticmethod
    def _auto_label(plotters: List["RoundsPlotter"]) -> str:
        """@brief Default label: the run folder name (single) or its base (group)."""
        if not plotters:
            return "(empty)"
        names = [os.path.basename(os.path.dirname(os.path.abspath(pl.csv_path)))
                 for pl in plotters]
        if len(names) == 1:
            return names[0] or os.path.basename(plotters[0].csv_path)
        # several runs: drop a trailing run-number suffix and show the shared base
        base = re.sub(r"[._-]?run[\s_-]*\d+$", "", names[0], flags=re.IGNORECASE)
        return f"{base or names[0]} (mean of {len(names)})"

    @staticmethod
    def _avg_series(plotters: List["RoundsPlotter"], column: str):
        """@brief Per-round mean (and population std) of a column across runs.

        Aligns by round number; each round is averaged over the runs that have a
        (non-NaN) value there. @return (xs, means, stds).
        """
        from collections import defaultdict
        buckets: Dict[int, List[float]] = defaultdict(list)
        for pl in plotters:
            try:
                xs, ys = pl._series(column)
            except KeyError:
                continue
            for x, y in zip(xs, ys):
                buckets[x].append(y)
        xs = sorted(buckets)
        means, stds = [], []
        for x in xs:
            vals = buckets[x]
            m = sum(vals) / len(vals)
            means.append(m)
            var = sum((v - m) ** 2 for v in vals) / len(vals)
            stds.append(var ** 0.5)
        return xs, means, stds

    def get_spec(self, column: str) -> MetricSpec:
        """@brief Registry spec for a column, or a sensible default for unknown columns."""
        if column in self.registry:
            return self.registry[column]
        return MetricSpec(column, column, column)

    def plot_metric(self, spec: MetricSpec, out_dir: str) -> Optional[str]:
        """@brief Render one overlaid figure (one line per group) for a metric.

        @return Saved file path, or @c None if no group has data for the column.
        """
        s = self.style
        base = self.series[0]["plotters"][0]      # any instance: styling is style-driven
        t = base._theme()
        plt = base._plt()
        from matplotlib.ticker import MaxNLocator, MultipleLocator

        fig, ax = plt.subplots(figsize=s.figsize)
        base._style_axes(fig, ax)

        n_plotted = 0
        series_data = []                       # (label, {round: value}) for CSV export
        for i, ser in enumerate(self.series):
            xs, ys, stds = self._avg_series(ser["plotters"], spec.column)
            if not xs:
                print(f"    [skip series] {ser['label']}: no data for {spec.column!r}")
                continue
            color = ser["color"] or self.palette[i % len(self.palette)]
            ax.plot(xs, ys, color=color, linewidth=s.linewidth,
                    marker=s.marker, markersize=s.markersize, label=ser["label"])
            series_data.append((ser["label"], dict(zip(xs, ys))))
            n_plotted += 1

        if n_plotted == 0:
            plt.close(fig)
            print(f"  [skip] {spec.column}: no data in any source")
            return None

        ax.set_xlabel(_bold_markup(spec.xlabel or s.xlabel), fontsize=s.label_fontsize, fontweight=s.label_weight)
        ax.set_ylabel(_bold_markup(spec.ylabel), fontsize=s.label_fontsize, fontweight=s.label_weight)
        ax.set_title(_bold_markup(spec.title), fontsize=s.title_fontsize, fontweight="bold")
        if spec.ylim is not None:
            ax.set_ylim(*spec.ylim)
        if spec.integer_y:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        elif spec.y_step:
            ax.yaxis.set_major_locator(MultipleLocator(spec.y_step))
        if s.x_step:
            ax.xaxis.set_major_locator(MultipleLocator(s.x_step))
        if s.suptitle:
            fig.suptitle(_bold_markup(s.suptitle), color=t["fg"], fontsize=s.title_fontsize - 1)
        if s.show_legend:
            ax.legend(fontsize=s.legend_fontsize, facecolor=t["legend_bg"],
                      labelcolor=t["fg"], edgecolor=t["spine"])

        fig.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        stem = spec.filename or spec.column
        # one series = a single averaged line (not a comparison) -> avg_ prefix
        prefix = "avg_" if len(self.series) == 1 else "compare_"
        path = os.path.join(out_dir, f"{prefix}{stem}.{s.output_format}")
        fig.savefig(path, dpi=s.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        if s.show:
            plt.show()
        plt.close(fig)
        print(f"  [ok] {os.path.basename(path)}")
        if s.export_csv:
            csv_path = os.path.join(out_dir, f"{prefix}{stem}.csv")
            all_rounds = sorted(set().union(*[set(d) for _, d in series_data]))
            header = ["round"] + [lbl for lbl, _ in series_data]
            rows = [[r] + [d.get(r, "") for _, d in series_data] for r in all_rounds]
            _write_data_csv(csv_path, header, rows)
            print(f"  [ok] {os.path.basename(csv_path)}")
        return path

    def plot(
        self,
        columns: Optional[List[str]] = None,
        out_dir: Optional[str] = None,
    ) -> List[str]:
        """@brief Render the chosen columns as overlaid figures.

        @param columns Columns to plot; defaults to the six SimLogger global
            metrics present in at least one source.
        @param out_dir Output directory; defaults to ``./comparison_plots``.
        @return List of saved file paths.
        """
        if columns is None:
            present = set()
            for ser in self.series:
                for pl in ser["plotters"]:
                    present |= set(pl.data.keys())
            columns = [c for c in DEFAULT_METRIC_COLUMNS if c in present]
        if out_dir is None:
            out_dir = os.path.join(os.getcwd(), "comparison_plots")

        labels = ", ".join(f"{ser['label']}×{len(ser['plotters'])}" for ser in self.series)
        print(f"[ComparisonPlotter] {len(self.series)} group(s) ({labels}) -> {out_dir}  "
              f"theme={self.style.theme}")
        saved: List[str] = []
        for col in columns:
            path = self.plot_metric(self.get_spec(col), out_dir)
            if path:
                saved.append(path)
        print(f"[ComparisonPlotter] Done. {len(saved)} figure(s) saved to: {out_dir}")
        return saved


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_kv(pairs: Optional[List[str]]) -> Dict[str, str]:
    """@brief Parse repeated ``COL=VALUE`` CLI args into a dict (``\\n`` -> newline)."""
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"Expected COL=VALUE, got: {item!r}")
        key, val = item.split("=", 1)
        out[key.strip()] = val.replace("\\n", "\n")
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rounds_plotter",
        description="Re-plot a SimLogger rounds.csv as separate, fully customizable figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv", nargs="*", default=["rounds.csv"],
                   help="One or more run folders (or rounds.csv paths). Two or more "
                        "switches to comparison mode: one line per run.")
    p.add_argument("--out", default=None,
                   help="Output directory (default: <csv_dir>/plots for a single "
                        "run, ./comparison_plots for comparison).")

    cmp = p.add_argument_group("comparison / averaging (2+ runs)")
    cmp.add_argument("--avg", action="store_true",
                     help="Average ALL the positional run folders into a single mean "
                          "line per metric (no comparison). Label/colour come from the "
                          "first --labels/--colors entry.")
    cmp.add_argument("--group", action="append", metavar="LABEL=folder1,folder2,...",
                     help="Define an averaged group: its folders' per-round metrics "
                          "are averaged into one mean line labelled LABEL. Repeat for "
                          "several groups (averaged-cell comparison). Overrides positionals.")
    cmp.add_argument("--labels", default=None,
                     help="Comma-separated legend labels, in run order "
                          "(default: each run's folder name). Ignored when --group is used.")
    cmp.add_argument("--colors", default=None,
                     help="Comma-separated line colours, in run/group order.")

    sel = p.add_argument_group("metric selection")
    sel.add_argument("--metrics", default=None,
                     help="Comma-separated columns to plot "
                          "(default: the 6 SimLogger global metrics).")
    sel.add_argument("--all", action="store_true",
                     help="Plot every numeric column present in the CSV.")

    cos = p.add_argument_group("figure cosmetics")
    cos.add_argument("--theme", choices=sorted(THEMES), default="dark")
    cos.add_argument("--figsize", default=None, metavar="W,H",
                     help="Figure size in inches, e.g. 8,5.")
    cos.add_argument("--dpi", type=int, default=130)
    cos.add_argument("--format", dest="fmt", default="png",
                     help="Output format: png, pdf, svg, ...")
    cos.add_argument("--linewidth", type=float, default=1.8)
    cos.add_argument("--marker", default="", help="Marker style, e.g. o or .")
    cos.add_argument("--markersize", type=float, default=4.0)
    cos.add_argument("--no-grid", action="store_true")
    cos.add_argument("--no-events", action="store_true",
                     help="Do not draw split/merge vertical markers.")
    cos.add_argument("--no-legend", action="store_true")
    cos.add_argument("--suptitle", default=None,
                     help="Descriptive line drawn above every title.")
    cos.add_argument("--show", action="store_true",
                     help="Open interactive windows (still saves files).")
    cos.add_argument("--export-csv", action="store_true",
                     help="Also write the plotted data as a sibling .csv next to each "
                          "figure (round + one column per line; group columns are the mean).")

    ax = p.add_argument_group("axes (global)")
    ax.add_argument("--xlabel", default="Round",
                    help="X-axis label for all plots (**text** renders bold).")
    ax.add_argument("--xstep", type=float, default=None,
                    help="X tick granularity (round spacing).")

    per = p.add_argument_group("axes (per-metric, repeatable COL=VALUE)")
    per.add_argument("--title", action="append", metavar="COL=TEXT",
                     help="Override a metric's title. Wrap part in **double "
                          "asterisks** to render it bold.")
    per.add_argument("--ylabel", action="append", metavar="COL=TEXT",
                     help="Override a metric's y-axis label (\\n allowed; "
                          "**text** renders bold).")
    per.add_argument("--color", action="append", metavar="COL=COLOR",
                     help="Override a metric's line colour.")
    per.add_argument("--ystep", action="append", metavar="COL=VALUE",
                     help="Override a metric's y tick granularity.")
    per.add_argument("--ylim", action="append", metavar="COL=LO,HI",
                     help="Override a metric's y-limits.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """@brief CLI entry point. @return process exit code."""
    args = _build_arg_parser().parse_args(argv)

    csvs = args.csv or ["rounds.csv"]

    def _check(p: str) -> str:
        if not os.path.isfile(_resolve_csv(p)):
            raise SystemExit(f"No rounds.csv found at: {p}")
        return p

    figsize = PlotStyle.figsize
    if args.figsize:
        try:
            w, h = (float(x) for x in args.figsize.split(","))
            figsize = (w, h)
        except ValueError:
            raise SystemExit(f"--figsize wants W,H (e.g. 8,5), got: {args.figsize!r}")

    style = PlotStyle(
        theme=args.theme,
        figsize=figsize,
        dpi=args.dpi,
        xlabel=args.xlabel.replace("\\n", "\n"),
        x_step=args.xstep,
        linewidth=args.linewidth,
        marker=args.marker,
        markersize=args.markersize,
        grid=not args.no_grid,
        mark_events=not args.no_events,
        show_legend=not args.no_legend,
        suptitle=args.suptitle.replace("\\n", "\n") if args.suptitle else None,
        output_format=args.fmt,
        show=args.show,
        export_csv=args.export_csv,
    )

    # Build the shared registry and apply per-metric overrides up front, so both
    # single and comparison modes see the same cosmetics.
    registry = default_metric_registry()
    titles = _parse_kv(args.title)
    ylabels = _parse_kv(args.ylabel)
    colors = _parse_kv(args.color)
    ysteps = _parse_kv(args.ystep)
    ylims = _parse_kv(args.ylim)
    touched = set(titles) | set(ylabels) | set(colors) | set(ysteps) | set(ylims)
    for col in touched:
        spec = registry.setdefault(col, MetricSpec(col, col, col))
        if col in titles:
            spec.title = titles[col]
        if col in ylabels:
            spec.ylabel = ylabels[col]
        if col in colors:
            spec.color = colors[col]
        if col in ysteps:
            spec.y_step = float(ysteps[col])
        if col in ylims:
            try:
                lo, hi = (float(x) for x in ylims[col].split(","))
            except ValueError:
                raise SystemExit(f"--ylim wants COL=LO,HI, got: {col}={ylims[col]!r}")
            spec.ylim = (lo, hi)

    explicit_metrics = ([c.strip() for c in args.metrics.split(",") if c.strip()]
                        if args.metrics else None)
    group_colors = [s.strip() for s in args.colors.split(",")] if args.colors else []

    def _columns_for(cp_or_plotter, all_plotters):
        if explicit_metrics is not None:
            return explicit_metrics
        if args.all:
            present = set()
            for pl in all_plotters:
                present |= set(pl.data.keys())
            return sorted(present)
        return None  # -> the default 6 present

    # ── averaged-group comparison: --group LABEL=folder1,folder2,... ──────────
    if args.group:
        groups = []
        for i, spec in enumerate(args.group):
            if "=" not in spec:
                raise SystemExit(f"--group wants LABEL=folder1,folder2,..., got: {spec!r}")
            label, folders_s = spec.split("=", 1)
            folders = [f.strip() for f in folders_s.split(",") if f.strip()]
            if not folders:
                raise SystemExit(f"--group {label!r} lists no folders")
            for f in folders:
                _check(f)
            clr = group_colors[i] if i < len(group_colors) else None
            groups.append({"label": label.strip(), "paths": folders, "color": clr})
        cp = ComparisonPlotter(groups, style=style, registry=registry)
        all_pl = [pl for ser in cp.series for pl in ser["plotters"]]
        cp.plot(_columns_for(cp, all_pl), out_dir=args.out)
        return 0

    for path in csvs:
        _check(path)

    # ── average all positionals into one mean line (no comparison) ────────────
    if args.avg:
        labels = [s.strip() for s in args.labels.split(",")] if args.labels else []
        group = {"label": labels[0] if labels else None,
                 "paths": csvs,
                 "color": group_colors[0] if group_colors else None}
        cp = ComparisonPlotter([group], style=style, registry=registry)
        all_pl = [pl for ser in cp.series for pl in ser["plotters"]]
        cp.plot(_columns_for(cp, all_pl), out_dir=args.out)
        return 0

    # ── single run: separate per-metric figures ──────────────────────────────
    if len(csvs) == 1:
        plotter = RoundsPlotter(csvs[0], style=style, registry=registry)
        plotter.plot(_columns_for(plotter, [plotter]), out_dir=args.out)
        return 0

    # ── comparison: one line per positional run (each a single-folder group) ──
    labels = [s.strip() for s in args.labels.split(",")] if args.labels else []
    groups = []
    for i, path in enumerate(csvs):
        lbl = labels[i] if i < len(labels) else None
        clr = group_colors[i] if i < len(group_colors) else None
        groups.append({"label": lbl, "paths": [path], "color": clr})
    cp = ComparisonPlotter(groups, style=style, registry=registry)
    all_pl = [pl for ser in cp.series for pl in ser["plotters"]]
    cp.plot(_columns_for(cp, all_pl), out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

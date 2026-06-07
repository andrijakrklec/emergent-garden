"""@file sim_logger.py
@brief Per-round CSV/JSONL logging and matplotlib visualisation for the CFL simulation.

Usage (in game.py):
    from src.utils.sim_logger import SimLogger
    self.logger = SimLogger(log_dir="logs")

    # Inside run_cfl_round call-site:
    self.logger.log_round(
        round_num       = self.cfl_round_counter,
        particles       = self.all_particles,
        kmeans          = self.kmeans,
        cluster_targets = self.cluster_targets,
        transfers       = transfers,
        event           = event,
        num_clusters    = self.num_clusters,
    )

    # On exit / whenever you want the plots:
    self.logger.plot_all()
"""

from __future__ import annotations

import csv
import json
import math
import os
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.constants import SIM_DIM

# Sim diagonal — normalizer for distance-based metrics (spatial_precision,
# local_loss's geo component).
_MAX_DIST = math.hypot(SIM_DIM[0], SIM_DIM[1])

# Must be set BEFORE pyplot is ever imported anywhere in the process.
# Agg is a non-interactive backend that writes to files — safe alongside pygame.
import matplotlib
matplotlib.use("Agg")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _delta_arrow(current: float, previous: float, threshold: float = 0.005) -> str:
    diff = current - previous
    if diff > threshold:
        return "↑"
    if diff < -threshold:
        return "↓"
    return "~"


def _cluster_purity(particles) -> float:
    """@brief Clustering purity of predicted @c cluster_id against ground-truth @c _true_cluster_id.

    Purity = (1/N) * Σ_k max_j |{p : p.cluster_id = k} ∩ {p : p._true_cluster_id = j}|.

    Standard unsupervised-clustering metric: for each predicted cluster, count
    how many of its members share the dominant ground-truth label, sum across
    clusters, divide by N. Headline ablation metric here — CFL aggregation
    drives it toward 1.0 by discovering the latent bias structure, while
    individual learning leaves @c cluster_id pinned at the random init so purity
    hovers near 1/K.

    @param particles List of Particle agents.
    @return float in [0, 1] (0.0 for an empty list).
    """
    if not particles:
        return 0.0
    counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for p in particles:
        counts[p.cluster_id][p._true_cluster_id] += 1
    matched = sum(max(true_counts.values()) for true_counts in counts.values())
    return matched / len(particles)


def _spatial_precision(particle, max_dist: float) -> float:
    """@brief How close a particle is to its target, in [0, 1] (1 = at the target).

    @c spatial_precision = 1 - dist(particle, _target) / sim_diagonal. Same scale
    as @c direction_precision so the two metrics are visually comparable on the
    same axes. Complements @c direction_precision: the model can be aimed
    correctly (high dir_prec) while the body is dragged off-target by emergent
    forces (low spatial_prec) — the two precisions decouple the cognitive and
    physical errors.

    @param particle A Particle with @c x, @c y, @c _target_x, @c _target_y.
    @param max_dist Sim diagonal (normalizer; same one used by local_loss).
    @return float in [0, 1].
    """
    dx = particle._target_x - particle.x
    dy = particle._target_y - particle.y
    dist = math.hypot(dx, dy)
    return float(1.0 - min(dist / max_dist, 1.0))


def _direction_precision(particle) -> float:
    """@brief Cosine alignment of a particle's learned heading to the *true* (unbiased) target direction, mapped to [0, 1].

    The local trainer corrupts each per-agent target direction with a persistent
    @c _local_bias (particle.py). With federation on, IFCA aggregates members'
    models and the random per-agent biases cancel — the learned heading approaches
    the true direction and this metric rises. Without federation, the bias persists
    and the metric stays low. This is the headline ablation metric: CFL vs
    individual learning under non-IID clients.

    1.0 = perfect alignment with the unbiased ideal; 0.5 = orthogonal; 0.0 = anti-aligned.

    @param particle A Particle with @c x, @c y, @c _target_x, @c _target_y, @c model.
    @return float in [0, 1].
    """
    tx = particle._target_x - particle.x
    ty = particle._target_y - particle.y
    n = math.hypot(tx, ty)
    if n == 0.0:
        return 1.0
    cos_sim = particle.model[0] * (tx / n) + particle.model[1] * (ty / n)
    return float((cos_sim + 1.0) * 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# SimLogger
# ──────────────────────────────────────────────────────────────────────────────

class SimLogger:
    """@brief Records per-round statistics to disk and generates matplotlib plots.

    Files written to `log_dir/run_<timestamp>/`:
        rounds.csv          — one row per CFL round, aggregate metrics
        migrations.csv      — one row per (round, src_cluster, dst_cluster) migration event
        cluster_sizes.csv   — one row per (round, cluster_id) with size + model stats
        events.jsonl        — split / merge / explosion events, one JSON object per line
        sim_log.txt         — human-readable mirror of terminal output
    """

    # ── model-dimension names (indices into Particle.model) ──────────────────
    MODEL_DIMS = [
        "dir_x", "dir_y", "confidence", "obstacle_pressure",
        "peer_alignment", "rounds_stable", "local_loss", "drift_velocity",
    ]

    def __init__(
        self,
        log_dir: str = "logs",
        tags: str = "",
        cfl_enabled: bool = True,
        attraction_enabled: bool = True,
        run_name: Optional[str] = None,
    ) -> None:
        """@brief Create a run directory and open all CSV/JSONL/text writers.

        @param log_dir Parent directory for run folders.
        @param tags Optional suffix tag for the auto-generated run directory name (e.g. "cfl_emergent").
        @param cfl_enabled Whether federation is on (recorded in the plot labels).
        @param attraction_enabled Whether emergent forces are on (recorded in the plot labels).
        @param run_name Explicit run-folder name; when given it overrides the
            `run_<timestamp>_<tags>` default (used for repeated/batch runs).
        """
        if run_name:
            self.run_dir = os.path.join(log_dir, run_name)
        else:
            suffix = f"_{tags}" if tags else ""
            self.run_dir = os.path.join(log_dir, f"run_{_ts()}{suffix}")
        _ensure_dir(self.run_dir)
        self.cfl_enabled       = cfl_enabled
        self.attraction_enabled = attraction_enabled
        self._config_label = (
            f"CFL: {'ON' if cfl_enabled else 'OFF'}  |  "
            f"Emergent: {'ON' if attraction_enabled else 'OFF'}"
        )

        # ── open CSV writers ──────────────────────────────────────────────────
        self._rounds_fh        = open(os.path.join(self.run_dir, "rounds.csv"),        "w", newline="")
        self._migrations_fh    = open(os.path.join(self.run_dir, "migrations.csv"),    "w", newline="")
        self._cluster_sizes_fh = open(os.path.join(self.run_dir, "cluster_sizes.csv"), "w", newline="")
        self._events_fh        = open(os.path.join(self.run_dir, "events.jsonl"),      "w")
        self._log_fh           = open(os.path.join(self.run_dir, "sim_log.txt"),       "w", encoding="utf-8")

        self._rounds_writer = csv.DictWriter(
            self._rounds_fh,
            fieldnames=[
                "round", "num_clusters", "num_particles",
                "inertia", "total_migrations", "event",
                "avg_confidence", "avg_local_loss",
                "avg_direction_precision", "avg_spatial_precision", "cluster_purity",
                "avg_peer_alignment", "avg_obstacle_pressure",
                "avg_drift_velocity", "avg_rounds_stable",
            ],
        )
        self._rounds_writer.writeheader()

        self._migrations_writer = csv.DictWriter(
            self._migrations_fh,
            fieldnames=["round", "src_cluster", "dst_cluster", "count"],
        )
        self._migrations_writer.writeheader()

        self._cluster_sizes_writer = csv.DictWriter(
            self._cluster_sizes_fh,
            fieldnames=[
                "round", "cluster_id", "size",
                "avg_confidence", "avg_local_loss",
                "avg_direction_precision", "avg_spatial_precision",
                "avg_peer_alignment", "avg_obstacle_pressure",
                "avg_drift_velocity", "avg_rounds_stable",
                "model_divergence", "spatial_spread",
            ],
        )
        self._cluster_sizes_writer.writeheader()

        # ── in-memory history for plotting ───────────────────────────────────
        # round-level
        self._history: List[Dict[str, Any]] = []
        # per-cluster history: {cluster_id: [{"round":…, "size":…, …}, …]}
        self._cluster_history: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        # migration totals per round
        self._migration_history: List[Tuple[int, int]] = []  # (round, total_migrations)
        # events list for vertical markers
        self._events: List[Dict[str, Any]] = []
        # cumulative migration matrix: (src, dst) -> total count (in-memory, no CSV re-read)
        self._mig_matrix: Dict[Tuple[int, int], int] = defaultdict(int)

        self._log(f"SimLogger initialised — output dir: {self.run_dir}\n")

    # ── public API ────────────────────────────────────────────────────────────

    def log_round(
        self,
        round_num: int,
        particles: list,
        kmeans,
        cluster_targets: list,
        transfers: dict,
        event: Optional[str],
        num_clusters: int,
    ) -> None:
        """@brief Record one round to all CSVs and the in-memory plotting history.

        Call once per round, right after run_cfl_round() returns.

        @param round_num 1-based round index.
        @param particles List of all Particle agents (read for aggregate metrics).
        @param kmeans Fitted KMeans (read for @c .inertia_).
        @param cluster_targets Current per-cluster (x, y) targets.
        @param transfers dict (src, dst) -> migration count for this round.
        @param event None, 'split', 'merge', or 'explosion' for this round.
        @param num_clusters Current number of clusters.
        @return None.
        """

        inertia = float(kmeans.inertia_) if hasattr(kmeans, "inertia_") else 0.0

        # ── aggregate particle metrics ────────────────────────────────────────
        models = np.array([p.model for p in particles])
        avg_conf        = float(np.mean(models[:, 2]))
        avg_loss        = float(np.mean(models[:, 6]))
        avg_peer        = float(np.mean(models[:, 4]))
        avg_obs         = float(np.mean(models[:, 3]))
        avg_drift       = float(np.mean(models[:, 7]))
        avg_stable      = float(np.mean(models[:, 5]))

        # Direction precision: per-agent cosine alignment of learned heading to
        # the *true* (unbiased) target direction, in [0, 1]. Headline ablation
        # metric — rises under CFL (federation averages out non-IID biases),
        # stays low under individual learning.
        dir_precisions = [_direction_precision(p) for p in particles]
        avg_dir_prec   = float(np.mean(dir_precisions)) if dir_precisions else 0.0

        # Spatial precision: 1 - distance/sim_diagonal. Complements direction
        # precision — distinguishes "model points the right way" from "particle
        # actually arrives at the target". Emergent attraction/repulsion forces
        # affect this without touching direction_precision.
        spatial_precisions = [_spatial_precision(p, _MAX_DIST) for p in particles]
        avg_spatial_prec   = float(np.mean(spatial_precisions)) if spatial_precisions else 0.0

        # Cluster purity: how well predicted cluster_id matches the latent
        # _true_cluster_id labels.
        purity = _cluster_purity(particles)

        total_mig = sum(transfers.values()) if transfers else 0

        # ── write rounds.csv ─────────────────────────────────────────────────
        row = {
            "round":               round_num,
            "num_clusters":        num_clusters,
            "num_particles":       len(particles),
            "inertia":             round(inertia, 4),
            "total_migrations":    total_mig,
            "event":               event or "",
            "avg_confidence":      round(avg_conf,  4),
            "avg_local_loss":      round(avg_loss,  4),
            "avg_direction_precision": round(avg_dir_prec, 4),
            "avg_spatial_precision":   round(avg_spatial_prec, 4),
            "cluster_purity":      None if purity is None else round(purity, 4),
            "avg_peer_alignment":  round(avg_peer,  4),
            "avg_obstacle_pressure": round(avg_obs, 4),
            "avg_drift_velocity":  round(avg_drift, 4),
            "avg_rounds_stable":   round(avg_stable, 4),
        }
        self._rounds_writer.writerow(row)
        self._rounds_fh.flush()
        self._history.append(row)
        self._migration_history.append((round_num, total_mig))

        # ── write migrations.csv ──────────────────────────────────────────────
        for (src, dst), count in (transfers or {}).items():
            self._migrations_writer.writerow({
                "round": round_num, "src_cluster": src,
                "dst_cluster": dst, "count": count,
            })
            if src >= 0:
                self._mig_matrix[(src, dst)] += count
        self._migrations_fh.flush()

        # ── write cluster_sizes.csv ───────────────────────────────────────────
        cluster_buckets: Dict[int, list] = defaultdict(list)
        for p in particles:
            cluster_buckets[p.cluster_id].append(p)

        cluster_rows = {}
        for cid in range(num_clusters):
            members = cluster_buckets.get(cid, [])
            if not members:
                continue
            ms = np.array([p.model for p in members])

            # model divergence: mean std across key dimensions (dir, confidence, loss)
            model_div = float(np.mean(np.std(ms[:, [0, 1, 2, 6]], axis=0))) if len(members) > 1 else 0.0

            # spatial spread: mean distance from cluster centroid
            xs = np.array([p.x for p in members])
            ys = np.array([p.y for p in members])
            cx, cy = xs.mean(), ys.mean()
            spatial_spread = float(np.mean(np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)))

            cluster_dir_prec     = float(np.mean([_direction_precision(p) for p in members]))
            cluster_spatial_prec = float(np.mean([_spatial_precision(p, _MAX_DIST) for p in members]))

            cs_row = {
                "round":                round_num,
                "cluster_id":           cid,
                "size":                 len(members),
                "avg_confidence":       round(float(np.mean(ms[:, 2])), 4),
                "avg_local_loss":       round(float(np.mean(ms[:, 6])), 4),
                "avg_direction_precision": round(cluster_dir_prec, 4),
                "avg_spatial_precision":   round(cluster_spatial_prec, 4),
                "avg_peer_alignment":   round(float(np.mean(ms[:, 4])), 4),
                "avg_obstacle_pressure":round(float(np.mean(ms[:, 3])), 4),
                "avg_drift_velocity":   round(float(np.mean(ms[:, 7])), 4),
                "avg_rounds_stable":     round(float(np.mean(ms[:, 5])), 4),
                "model_divergence":      round(model_div,     4),
                "spatial_spread":        round(spatial_spread, 1),
            }
            self._cluster_sizes_writer.writerow(cs_row)
            self._cluster_history[cid].append(cs_row)
            cluster_rows[cid] = cs_row
        self._cluster_sizes_fh.flush()

        # ── write events.jsonl ────────────────────────────────────────────────
        if event:
            ev = {"round": round_num, "type": event, "num_clusters": num_clusters}
            self._events_fh.write(json.dumps(ev) + "\n")
            self._events_fh.flush()
            self._events.append(ev)

        self._write_terminal_log(
            round_num, num_clusters, inertia, avg_conf, avg_loss,
            avg_peer, avg_dir_prec, avg_spatial_prec, purity,
            total_mig, event, transfers, cluster_rows,
        )

    def log_explosion(self, round_num: int) -> None:
        """@brief Record an explosion event so it appears as a marker in the plots.

        @param round_num Round during which the explosion was triggered.
        """
        ev = {"round": round_num, "type": "explosion"}
        self._events_fh.write(json.dumps(ev) + "\n")
        self._events_fh.flush()
        self._events.append(ev)
        self._log(f"\n[ROUND {round_num}]  *** EXPLOSION triggered ***")

    def plot_all(self) -> None:
        """@brief Generate and save all PNG plots from the in-memory history.

        Produces global metrics, per-cluster sizes, per-cluster health, the
        migration heatmap (CFL only) and cohesion/divergence. Safe to call any time.

        @return None — files are written into the run directory.
        """
        if not self._history:
            print("[SimLogger] No data to plot yet.")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[SimLogger] matplotlib not installed -- skipping plots.")
            return

        print(f"[SimLogger] Plotting {len(self._history)} rounds of data...")

        rounds = [r["round"] for r in self._history]
        saved = []

        PALETTE = [
            "#e05555", "#5599dd", "#55bb77", "#ddaa33",
            "#9955cc", "#33cccc", "#dd7733", "#aaaaaa",
        ]

        def _mark_events(ax):
            for ev in self._events:
                r = ev["round"]
                if r not in rounds:
                    continue
                colour = {"split": "#ffaa00", "merge": "#00aaff", "explosion": "#ff3333"}.get(ev["type"], "#888888")
                ax.axvline(r, color=colour, linewidth=1.2, linestyle="--", alpha=0.7)

        def _event_legend_patches():
            from matplotlib.lines import Line2D
            return [
                Line2D([0], [0], color="#ffaa00", lw=1.5, ls="--", label="split"),
                Line2D([0], [0], color="#00aaff", lw=1.5, ls="--", label="merge"),
                Line2D([0], [0], color="#ff3333", lw=1.5, ls="--", label="explosion"),
            ]

        def _style_ax(ax):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

        def _disable_ax(ax, reason):
            """Grey out a panel that is N/A for this configuration."""
            ax.set_facecolor("#111118")
            ax.tick_params(colors="#555566")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333344")
            ax.text(0.5, 0.5, reason, transform=ax.transAxes,
                    ha="center", va="center", color="#666677", fontsize=10, style="italic")

        all_cids = sorted(self._cluster_history.keys())

        # FIGURE 1 -- Global metrics 2x3 grid (same layout for all configs)
        # Row 0: CFL-specific (inertia / #clusters / migrations)
        # Row 1: Always-on   (confidence / loss / peer-alignment)
        try:
            fig1, axes = plt.subplots(2, 3, figsize=(16, 9))
            fig1.suptitle(
                f"Global Metrics per Round\n{self._config_label}",
                fontsize=13, fontweight="bold", color="white",
            )
            fig1.patch.set_facecolor("#1a1a2e")
            for ax in axes.flat:
                _style_ax(ax)

            def _line(ax, key, label, color, ylim=None):
                # Skip rounds where this metric is absent (a key missing from
                # some history rows) so a partial series still plots.
                pairs = [(r, h.get(key)) for r, h in zip(rounds, self._history)
                         if h.get(key) is not None]
                if not pairs:
                    _disable_ax(ax, f"N/A\n({label})")
                    return
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, color=color, linewidth=1.8, label=label)
                _mark_events(ax)
                ax.set_xlabel("Round")
                ax.set_title(label)
                if ylim:
                    ax.set_ylim(*ylim)
                ax.legend(handles=[ax.lines[0]] + _event_legend_patches(),
                          fontsize=7, facecolor="#1a1a2e", labelcolor="white")

            # Row 0: CFL-only diagnostics. Purity, # Clusters and Migrations
            # all depend on IFCA running — without federation they're either a
            # flat chance line or undefined.
            if self.cfl_enabled:
                from matplotlib.ticker import MaxNLocator
                _line(axes[0, 0], "cluster_purity",   "Cluster Purity\n(cluster_id vs latent true label)", "#ffaaff", ylim=(0, 1.05))
                _line(axes[0, 1], "num_clusters",     "# Clusters",                                        "#5599dd")
                axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
                _line(axes[0, 2], "total_migrations", "Migrations per Round",                              "#ffaa00")
            else:
                _disable_ax(axes[0, 0], "N/A — CFL disabled\n(Cluster Purity ≈ 1/K random)")
                _disable_ax(axes[0, 1], "N/A — CFL disabled\n(# Clusters)")
                _disable_ax(axes[0, 2], "N/A — CFL disabled\n(Migrations)")

            # Row 1: always-on convergence/precision metrics (work in any
            # ablation, fixed scales for cross-ablation comparability).
            _line(axes[1, 0], "avg_local_loss",          "Avg Local Loss",                                                "#e05555", ylim=(0, 0.5))
            _line(axes[1, 1], "avg_direction_precision", "Avg Direction Precision\n(model vs true target dir)",          "#33cccc", ylim=(0, 1.05))
            _line(axes[1, 2], "avg_spatial_precision",   "Avg Spatial Precision\n(1 - dist(particle, target) / sim_diag)", "#dd7733", ylim=(0, 1.05))

            plt.tight_layout()
            p1 = os.path.join(self.run_dir, "global_metrics.png")
            fig1.savefig(p1, dpi=130, bbox_inches="tight", facecolor=fig1.get_facecolor())
            plt.close(fig1)
            saved.append(p1)
            print(f"  [ok] global_metrics.png")
        except Exception:
            print("  [FAIL] global_metrics.png")
            traceback.print_exc()

        # FIGURE 2 -- Per-cluster size over time
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            fig2.patch.set_facecolor("#1a1a2e")
            _style_ax(ax2)
            fig2.suptitle(f"Cluster Size Over Rounds\n{self._config_label}",
                          color="white", fontsize=12, fontweight="bold")

            for cid in all_cids:
                hist = self._cluster_history[cid]
                rs = [h["round"] for h in hist]
                sz = [h["size"]  for h in hist]
                ax2.plot(rs, sz, color=PALETTE[cid % len(PALETTE)],
                         linewidth=1.8, label=f"Cluster {cid}", marker="o", markersize=3)

            _mark_events(ax2)
            ax2.set_xlabel("Round")
            ax2.set_ylabel("# Particles")
            ax2.set_ylim(bottom=0)
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles + _event_legend_patches(),
                       labels  + ["split", "merge", "explosion"],
                       fontsize=8, facecolor="#1a1a2e", labelcolor="white", loc="upper right")
            plt.tight_layout()
            p2 = os.path.join(self.run_dir, "cluster_sizes.png")
            fig2.savefig(p2, dpi=130, bbox_inches="tight", facecolor=fig2.get_facecolor())
            plt.close(fig2)
            saved.append(p2)
            print(f"  [ok] cluster_sizes.png")
        except Exception:
            print("  [FAIL] cluster_sizes.png")
            traceback.print_exc()

        # FIGURE 3 -- Per-cluster confidence, loss, direction precision, spatial precision
        # (fixed axes on the [0,1] metrics for cross-ablation comparability)
        try:
            fig3, (ax3a, ax3b, ax3c, ax3d) = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
            fig3.patch.set_facecolor("#1a1a2e")
            fig3.suptitle(f"Per-Cluster Model Health\n{self._config_label}",
                          color="white", fontsize=12, fontweight="bold")
            for ax in (ax3a, ax3b, ax3c, ax3d):
                _style_ax(ax)

            for cid in all_cids:
                hist  = self._cluster_history[cid]
                rs    = [h["round"]          for h in hist]
                conf  = [h["avg_confidence"] for h in hist]
                loss  = [h["avg_local_loss"] for h in hist]
                dprec = [h.get("avg_direction_precision", 0.0) for h in hist]
                sprec = [h.get("avg_spatial_precision",   0.0) for h in hist]
                color = PALETTE[cid % len(PALETTE)]
                ax3a.plot(rs, conf,  color=color, linewidth=1.6, label=f"Cluster {cid}", marker=".", markersize=3)
                ax3b.plot(rs, loss,  color=color, linewidth=1.6, label=f"Cluster {cid}", marker=".", markersize=3)
                ax3c.plot(rs, dprec, color=color, linewidth=1.6, label=f"Cluster {cid}", marker=".", markersize=3)
                ax3d.plot(rs, sprec, color=color, linewidth=1.6, label=f"Cluster {cid}", marker=".", markersize=3)

            _mark_events(ax3a)
            _mark_events(ax3b)
            _mark_events(ax3c)
            _mark_events(ax3d)
            ax3a.set_ylabel("Avg Confidence", color="white")
            ax3a.set_ylim(0, 1.05)
            ax3b.set_ylabel("Avg Local Loss", color="white")
            ax3b.set_ylim(0, 0.5)
            ax3c.set_ylabel("Avg Direction Precision\n(model vs true target dir)", color="white")
            ax3c.set_ylim(0, 1.05)
            ax3d.set_ylabel("Avg Spatial Precision\n(particle vs target position)", color="white")
            ax3d.set_ylim(0, 1.05)
            ax3d.set_xlabel("Round", color="white")

            for ax in (ax3a, ax3b, ax3c, ax3d):
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles + _event_legend_patches(),
                          labels  + ["split", "merge", "explosion"],
                          fontsize=7, facecolor="#1a1a2e", labelcolor="white", loc="lower right")

            plt.tight_layout()
            p3 = os.path.join(self.run_dir, "cluster_health.png")
            fig3.savefig(p3, dpi=130, bbox_inches="tight", facecolor=fig3.get_facecolor())
            plt.close(fig3)
            saved.append(p3)
            print(f"  [ok] cluster_health.png")
        except Exception:
            print("  [FAIL] cluster_health.png")
            traceback.print_exc()

        # FIGURE 4 -- Migration heatmap (CFL only)
        try:
            if not self.cfl_enabled:
                print("  [skip] migration_heatmap.png (CFL disabled)")
            elif self._mig_matrix:
                all_ids = sorted({k for pair in self._mig_matrix for k in pair})
                n = len(all_ids)
                idx_map = {cid: i for i, cid in enumerate(all_ids)}
                mat = np.zeros((n, n), dtype=int)
                for (src, dst), cnt in self._mig_matrix.items():
                    mat[idx_map[src], idx_map[dst]] += cnt

                fig4, ax4 = plt.subplots(figsize=(max(5, n + 2), max(4, n + 1)))
                fig4.patch.set_facecolor("#1a1a2e")
                _style_ax(ax4)
                im = ax4.imshow(mat, cmap="YlOrRd", aspect="auto")
                ax4.set_xticks(range(n))
                ax4.set_yticks(range(n))
                ax4.set_xticklabels([f"C{c}" for c in all_ids], color="white")
                ax4.set_yticklabels([f"C{c}" for c in all_ids], color="white")
                ax4.set_xlabel("Destination Cluster", color="white")
                ax4.set_ylabel("Source Cluster", color="white")
                ax4.set_title(f"Total Migrations (all rounds)\n{self._config_label}", color="white", fontsize=11)
                cbar = fig4.colorbar(im, ax=ax4)
                cbar.ax.tick_params(colors="white")
                cbar.set_label("# Particles", color="white")
                max_val = mat.max()
                for i in range(n):
                    for j in range(n):
                        if mat[i, j]:
                            ax4.text(j, i, str(mat[i, j]), ha="center", va="center",
                                     fontsize=9, color="black" if mat[i, j] > max_val * 0.5 else "white")
                plt.tight_layout()
                p4 = os.path.join(self.run_dir, "migration_heatmap.png")
                fig4.savefig(p4, dpi=130, bbox_inches="tight", facecolor=fig4.get_facecolor())
                plt.close(fig4)
                saved.append(p4)
                print(f"  [ok] migration_heatmap.png")
            else:
                print("  [skip] migration_heatmap.png (no inter-cluster migrations yet)")
        except Exception:
            print("  [FAIL] migration_heatmap.png")
            traceback.print_exc()

        # FIGURE 5 -- Cohesion & Divergence (fixed spatial axis for comparability)
        try:
            has_div    = any("model_divergence" in h for cid in all_cids for h in self._cluster_history[cid])
            has_spread = any("spatial_spread"   in h for cid in all_cids for h in self._cluster_history[cid])

            if has_div or has_spread:
                fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
                fig5.patch.set_facecolor("#1a1a2e")
                fig5.suptitle(f"Per-Cluster Cohesion & Model Divergence\n{self._config_label}",
                              color="white", fontsize=12, fontweight="bold")
                for ax in (ax5a, ax5b):
                    _style_ax(ax)

                for cid in all_cids:
                    hist  = self._cluster_history[cid]
                    color = PALETTE[cid % len(PALETTE)]

                    if has_div:
                        rs  = [h["round"]           for h in hist if "model_divergence" in h]
                        div = [h["model_divergence"] for h in hist if "model_divergence" in h]
                        if rs:
                            ax5a.plot(rs, div, color=color, linewidth=1.6,
                                      label=f"Cluster {cid}", marker=".", markersize=3)

                    if has_spread:
                        rs2 = [h["round"]          for h in hist if "spatial_spread" in h]
                        spr = [h["spatial_spread"] for h in hist if "spatial_spread" in h]
                        if rs2:
                            ax5b.plot(rs2, spr, color=color, linewidth=1.6,
                                      label=f"Cluster {cid}", marker=".", markersize=3)

                _mark_events(ax5a)
                _mark_events(ax5b)
                ax5a.set_ylabel("Model Divergence (avg std)", color="white")
                ax5a.set_ylim(bottom=0)
                ax5b.set_ylabel("Spatial Spread (px, mean dist from centroid)", color="white")
                ax5b.set_ylim(0, 500)   # fixed: SIM is 1000×800, spread > 500 px is an outlier
                ax5b.set_xlabel("Round", color="white")

                for ax in (ax5a, ax5b):
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles + _event_legend_patches(),
                              labels  + ["split", "merge", "explosion"],
                              fontsize=7, facecolor="#1a1a2e", labelcolor="white", loc="upper right")

                plt.tight_layout()
                p5 = os.path.join(self.run_dir, "cohesion_divergence.png")
                fig5.savefig(p5, dpi=130, bbox_inches="tight", facecolor=fig5.get_facecolor())
                plt.close(fig5)
                saved.append(p5)
                print("  [ok] cohesion_divergence.png")
        except Exception:
            print("  [FAIL] cohesion_divergence.png")
            traceback.print_exc()

        print(f"[SimLogger] Done. {len(saved)} plot(s) saved to: {self.run_dir}")

    # ── teardown ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """@brief Flush all file handles, render the final plots, then close everything."""
        for fh in (self._rounds_fh, self._migrations_fh,
                   self._cluster_sizes_fh, self._events_fh, self._log_fh):
            try:
                fh.flush()
            except Exception:
                pass

        print("\n[SimLogger] Generating plots...")
        try:
            self.plot_all()
        except Exception:
            print("[SimLogger] ERROR in plot_all():")
            traceback.print_exc()

        for fh in (self._rounds_fh, self._migrations_fh,
                   self._cluster_sizes_fh, self._events_fh, self._log_fh):
            try:
                fh.close()
            except Exception:
                pass

        print(f"[SimLogger] All data saved to: {self.run_dir}")

    # ── internal ──────────────────────────────────────────────────────────────

    def _write_terminal_log(
        self,
        round_num: int,
        num_clusters: int,
        inertia: float,
        avg_conf: float,
        avg_loss: float,
        avg_peer: float,
        avg_dir_prec: float,
        avg_spatial_prec: float,
        purity: Optional[float],
        total_mig: int,
        event: Optional[str],
        transfers: dict,
        cluster_rows: Dict[int, Dict],
    ) -> None:
        prev = self._history[-2] if len(self._history) >= 2 else None

        conf_arrow  = _delta_arrow(avg_conf,         prev["avg_confidence"])           if prev else " "
        loss_arrow  = _delta_arrow(avg_loss,         prev["avg_local_loss"])           if prev else " "
        peer_arrow  = _delta_arrow(avg_peer,         prev["avg_peer_alignment"])       if prev else " "
        dprec_arrow = _delta_arrow(avg_dir_prec,     prev["avg_direction_precision"]) if prev else " "
        sprec_arrow = _delta_arrow(avg_spatial_prec, prev["avg_spatial_precision"])   if prev else " "
        if prev is not None and prev.get("cluster_purity") is not None and purity is not None:
            purity_arrow = _delta_arrow(purity, prev["cluster_purity"])
        else:
            purity_arrow = " "
        purity_str = f"  purity={purity:.3f}{purity_arrow}" if purity is not None else ""

        lines = [
            f"\n[ROUND {round_num}]  clusters={num_clusters}  inertia={inertia:.2f}  migrations={total_mig}",
            f"  global │ conf={avg_conf:.3f}{conf_arrow}  loss={avg_loss:.3f}{loss_arrow}  "
            f"peer={avg_peer:.3f}{peer_arrow}  dir_prec={avg_dir_prec:.3f}{dprec_arrow}  "
            f"sp_prec={avg_spatial_prec:.3f}{sprec_arrow}{purity_str}",
        ]

        if cluster_rows:
            lines.append("  ┌─────────┬──────┬────────┬────────┬─────────┬─────────┬────────┬────────┬────────┐")
            lines.append("  │ cluster │ size │  conf  │  loss  │ dir_prec │ sp_prec │  peer  │ diverg │ spread │")
            lines.append("  ├─────────┼──────┼────────┼────────┼─────────┼─────────┼────────┼────────┼────────┤")
            for cid, r in sorted(cluster_rows.items()):
                lines.append(
                    f"  │ {cid:^7d} │ {r['size']:^4d} │ "
                    f"{r['avg_confidence']:^6.3f} │ {r['avg_local_loss']:^6.3f} │ "
                    f"{r['avg_direction_precision']:^7.3f} │ "
                    f"{r['avg_spatial_precision']:^7.3f} │ "
                    f"{r['avg_peer_alignment']:^6.3f} │ {r['model_divergence']:^6.3f} │ "
                    f"{r['spatial_spread']:^6.0f} │"
                )
            lines.append("  └─────────┴──────┴────────┴────────┴─────────┴─────────┴────────┴────────┴────────┘")

        if event:
            lines.append(f"  *** EVENT: {event.upper()} ***")
        if transfers:
            for (src, dst), cnt in sorted(transfers.items(), key=lambda x: -x[1]):
                src_name = "Unassigned" if src == -1 else f"Cluster {src}"
                lines.append(f"    {cnt:3d} agents  {src_name} -> Cluster {dst}")

        self._log("\n".join(lines))

    def _log(self, text: str) -> None:
        self._log_fh.write(text + "\n")
        self._log_fh.flush()

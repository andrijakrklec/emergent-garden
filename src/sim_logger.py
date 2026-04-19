"""
sim_logger.py — Logging and visualisation for the CFL simulation.

Usage (in game.py):
    from src.sim_logger import SimLogger
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
import os
import math
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


# ──────────────────────────────────────────────────────────────────────────────
# SimLogger
# ──────────────────────────────────────────────────────────────────────────────

class SimLogger:
    """
    Records per-round statistics to disk and generates matplotlib plots.

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

    def __init__(self, log_dir: str = "logs") -> None:
        self.run_dir = os.path.join(log_dir, f"run_{_ts()}")
        _ensure_dir(self.run_dir)

        # ── open CSV writers ──────────────────────────────────────────────────
        self._rounds_fh        = open(os.path.join(self.run_dir, "rounds.csv"),        "w", newline="")
        self._migrations_fh    = open(os.path.join(self.run_dir, "migrations.csv"),    "w", newline="")
        self._cluster_sizes_fh = open(os.path.join(self.run_dir, "cluster_sizes.csv"), "w", newline="")
        self._events_fh        = open(os.path.join(self.run_dir, "events.jsonl"),      "w")
        self._log_fh           = open(os.path.join(self.run_dir, "sim_log.txt"),       "w")

        self._rounds_writer = csv.DictWriter(
            self._rounds_fh,
            fieldnames=[
                "round", "num_clusters", "num_particles",
                "inertia", "total_migrations", "event",
                "avg_confidence", "avg_local_loss",
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
                "avg_peer_alignment", "avg_obstacle_pressure",
                "avg_drift_velocity",
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
        """Call once per CFL round, right after run_cfl_round() returns."""

        inertia = float(kmeans.inertia_) if hasattr(kmeans, "inertia_") else 0.0

        # ── aggregate particle metrics ────────────────────────────────────────
        models = np.array([p.model for p in particles])
        avg_conf        = float(np.mean(models[:, 2]))
        avg_loss        = float(np.mean(models[:, 6]))
        avg_peer        = float(np.mean(models[:, 4]))
        avg_obs         = float(np.mean(models[:, 3]))
        avg_drift       = float(np.mean(models[:, 7]))
        avg_stable      = float(np.mean(models[:, 5]))

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
        self._migrations_fh.flush()

        # ── write cluster_sizes.csv ───────────────────────────────────────────
        cluster_buckets: Dict[int, list] = defaultdict(list)
        for p in particles:
            cluster_buckets[p.cluster_id].append(p)

        for cid in range(num_clusters):
            members = cluster_buckets.get(cid, [])
            if not members:
                continue
            ms = np.array([p.model for p in members])
            cs_row = {
                "round":                round_num,
                "cluster_id":           cid,
                "size":                 len(members),
                "avg_confidence":       round(float(np.mean(ms[:, 2])), 4),
                "avg_local_loss":       round(float(np.mean(ms[:, 6])), 4),
                "avg_peer_alignment":   round(float(np.mean(ms[:, 4])), 4),
                "avg_obstacle_pressure":round(float(np.mean(ms[:, 3])), 4),
                "avg_drift_velocity":   round(float(np.mean(ms[:, 7])), 4),
            }
            self._cluster_sizes_writer.writerow(cs_row)
            self._cluster_history[cid].append(cs_row)
        self._cluster_sizes_fh.flush()

        # ── write events.jsonl ────────────────────────────────────────────────
        if event:
            ev = {"round": round_num, "type": event, "num_clusters": num_clusters}
            self._events_fh.write(json.dumps(ev) + "\n")
            self._events_fh.flush()
            self._events.append(ev)

        # ── human-readable log ────────────────────────────────────────────────
        sizes = {cid: len(m) for cid, m in cluster_buckets.items()}
        lines = [
            f"\n[ROUND {round_num}]  clusters={num_clusters}  inertia={inertia:.2f}",
            f"  avg confidence={avg_conf:.3f}  avg loss={avg_loss:.3f}  "
            f"peer_align={avg_peer:.3f}  migrations={total_mig}",
            f"  cluster sizes: {dict(sorted(sizes.items()))}",
        ]
        if event:
            lines.append(f"  *** EVENT: {event.upper()} ***")
        if transfers:
            for (src, dst), cnt in sorted(transfers.items(), key=lambda x: -x[1]):
                src_name = "Unassigned" if src == -1 else f"Cluster {src}"
                lines.append(f"    {cnt:3d} agents  {src_name} -> Cluster {dst}")
        self._log("\n".join(lines))

    def log_explosion(self, round_num: int) -> None:
        """Call from trigger_explosion() so the event is timestamped."""
        ev = {"round": round_num, "type": "explosion"}
        self._events_fh.write(json.dumps(ev) + "\n")
        self._events_fh.flush()
        self._events.append(ev)
        self._log(f"\n[ROUND {round_num}]  *** EXPLOSION triggered ***")

    def plot_all(self) -> None:
        """Generate and save all plots. Safe to call at any time."""
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

        # FIGURE 1 -- Global metrics 2x3 grid
        try:
            fig1, axes = plt.subplots(2, 3, figsize=(16, 9))
            fig1.suptitle("CFL Simulation -- Global Metrics per Round", fontsize=14, fontweight="bold", color="white")
            fig1.patch.set_facecolor("#1a1a2e")
            for ax in axes.flat:
                _style_ax(ax)

            def _line(ax, key, label, color, ylabel=None):
                vals = [r[key] for r in self._history]
                ax.plot(rounds, vals, color=color, linewidth=1.8, label=label)
                _mark_events(ax)
                ax.set_xlabel("Round")
                ax.set_ylabel(ylabel or label)
                ax.set_title(label)
                ax.legend(handles=[ax.lines[0]] + _event_legend_patches(),
                          fontsize=7, facecolor="#1a1a2e", labelcolor="white")

            _line(axes[0, 0], "inertia",           "KMeans Inertia",       "#e05555", "inertia")
            _line(axes[0, 1], "num_clusters",       "# Clusters",           "#5599dd", "clusters")
            _line(axes[0, 2], "total_migrations",   "Migrations per Round", "#ffaa00", "migrations")
            _line(axes[1, 0], "avg_confidence",     "Avg Confidence",       "#55bb77", "0-1")
            _line(axes[1, 1], "avg_local_loss",     "Avg Local Loss",       "#dd7733", "0-1")
            _line(axes[1, 2], "avg_peer_alignment", "Avg Peer Alignment",   "#9955cc", "0-1")

            plt.tight_layout()
            p1 = os.path.join(self.run_dir, "global_metrics.png")
            fig1.savefig(p1, dpi=130, bbox_inches="tight", facecolor=fig1.get_facecolor())
            plt.close(fig1)
            saved.append(p1)
            print(f"  [ok] global_metrics.png")
        except Exception:
            print("  [FAIL] global_metrics.png")
            traceback.print_exc()

        all_cids = sorted(self._cluster_history.keys())

        # FIGURE 2 -- Per-cluster size over time
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            fig2.patch.set_facecolor("#1a1a2e")
            _style_ax(ax2)
            ax2.set_title("Cluster Size Over Rounds", color="white", fontsize=12)

            for cid in all_cids:
                hist = self._cluster_history[cid]
                rs = [h["round"] for h in hist]
                sz = [h["size"]  for h in hist]
                ax2.plot(rs, sz, color=PALETTE[cid % len(PALETTE)],
                         linewidth=1.8, label=f"Cluster {cid}", marker="o", markersize=3)

            _mark_events(ax2)
            ax2.set_xlabel("Round")
            ax2.set_ylabel("# Particles")
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

        # FIGURE 3 -- Per-cluster confidence & loss
        try:
            fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            fig3.patch.set_facecolor("#1a1a2e")
            fig3.suptitle("Per-Cluster Model Health", color="white", fontsize=12, fontweight="bold")
            for ax in (ax3a, ax3b):
                _style_ax(ax)

            for cid in all_cids:
                hist  = self._cluster_history[cid]
                rs    = [h["round"]          for h in hist]
                conf  = [h["avg_confidence"] for h in hist]
                loss  = [h["avg_local_loss"] for h in hist]
                color = PALETTE[cid % len(PALETTE)]
                ax3a.plot(rs, conf, color=color, linewidth=1.6, label=f"Cluster {cid}", marker=".", markersize=3)
                ax3b.plot(rs, loss, color=color, linewidth=1.6, label=f"Cluster {cid}", marker=".", markersize=3)

            _mark_events(ax3a)
            _mark_events(ax3b)
            ax3a.set_ylabel("Avg Confidence", color="white")
            ax3a.set_ylim(0, 1.05)
            ax3b.set_ylabel("Avg Local Loss", color="white")
            ax3b.set_ylim(0, 1.05)
            ax3b.set_xlabel("Round", color="white")

            for ax in (ax3a, ax3b):
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

        # FIGURE 4 -- Migration heatmap
        try:
            mig_matrix: Dict[Tuple[int, int], int] = defaultdict(int)
            mig_csv = os.path.join(self.run_dir, "migrations.csv")
            with open(mig_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    src = int(row["src_cluster"])
                    dst = int(row["dst_cluster"])
                    cnt = int(row["count"])
                    if src >= 0:
                        mig_matrix[(src, dst)] += cnt

            if mig_matrix:
                all_ids = sorted({k for pair in mig_matrix for k in pair})
                n = len(all_ids)
                idx_map = {cid: i for i, cid in enumerate(all_ids)}
                mat = np.zeros((n, n), dtype=int)
                for (src, dst), cnt in mig_matrix.items():
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
                ax4.set_title("Total Migrations (all rounds)", color="white", fontsize=11)
                cbar = fig4.colorbar(im, ax=ax4)
                cbar.ax.tick_params(colors="white")
                cbar.set_label("# Particles", color="white")
                for i in range(n):
                    for j in range(n):
                        if mat[i, j]:
                            ax4.text(j, i, str(mat[i, j]), ha="center", va="center",
                                     fontsize=9, color="black" if mat[i, j] > mat.max() * 0.5 else "white")
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

        print(f"[SimLogger] Done. {len(saved)} plot(s) saved to: {self.run_dir}")

    # ── teardown ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush all file handles, generate plots, then close."""
        # Flush CSVs first so migrations.csv is complete before plot_all reads it
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

    def _log(self, text: str) -> None:
        self._log_fh.write(text + "\n")
        self._log_fh.flush()

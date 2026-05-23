"""@file game.py
@brief Simulation orchestration and the pygame GUI.

Defines two cooperating halves that run on separate threads:
  - SimulationThread — a daemon thread that owns @e all simulation state and
    steps the physics + federation at a fixed rate, publishing immutable snapshots.
  - Game — the main-thread pygame front-end that renders snapshots and forwards
    user input back to the simulation through a command queue.

Also holds the layered configuration loader (_load_merged_config(),
_load_force_config()) that overlays a per-experiment config on top of the master
config.json.
"""

import pygame
import numpy as np
import random
import math
import time
import threading
import queue
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans

from src.constants import (
    SCREEN_DIM, SIM_WIDTH, SIM_HEIGHT, GUI_WIDTH, GUI_BACKGROUND_COLOR,
    PARTICLE_DEFAULT_SPAWN_NUM, PARTICLE_DEFAULT_SPAWN_FRAME,
    BACK_BLACK, FRAME_RATE, PARTICLE_COLOR_WHITE, WALL_BOUNDARY,
    PARTICLE_COLOR_RED, PARTICLE_COLOR_YELLOW, PARTICLE_COLOR_GREEN, PARTICLE_COLOR_BLUE,
)
from src.particle import (
    instantiateGroup, local_train, run_cfl_round,
    apply_physics_rules, update_peer_alignment,
    MIN_CLUSTERS, MAX_CLUSTERS, BLEND_MATURITY_ROUNDS,
)
from src.sim_logger import SimLogger


# Physics tick rate
PHYSICS_HZ = 60
# Append a trail point every N physics steps (keeps trail density frame-rate-independent)
TRAIL_SAMPLE_EVERY = PHYSICS_HZ // FRAME_RATE  # = 10

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


def _load_merged_config(config_path: str) -> dict:
    """@brief Load master config.json, then overlay a specific config file on top.

    Only keys present in the specific file overwrite master values, so a config in
    configs/ only needs to list the settings that differ from config.json. The
    @c pair_rules table is deep-merged so a specific file can change individual
    pairs without repeating the whole table. Keys starting with '_' are dropped.

    @param config_path Path to the per-experiment config (may equal the master path).
    @return dict the merged configuration.
    """
    def _read(path: str) -> dict:
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            return {k: v for k, v in raw.items() if not k.startswith("_")}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    master = _read(DEFAULT_CONFIG_PATH)
    if os.path.abspath(config_path) == os.path.abspath(DEFAULT_CONFIG_PATH):
        return master

    override = _read(config_path)
    if not override:
        return master

    # Deep-merge pair_rules; shallow-merge everything else
    base_rules = master.get("pair_rules", {})
    ovr_rules  = override.get("pair_rules", {})
    merged = {**master, **override}
    if base_rules or ovr_rules:
        merged["pair_rules"] = {**base_rules, **ovr_rules}
    return merged


def _load_force_config(config_path: str) -> Tuple[float, float, Dict, float]:
    """@brief Load g_attract, g_repel and the optional pair_rules from merged config.

    Config force values are in a readable ~-1..+1 scale; @c force_scale multiplies
    them into internal units before use. Falls back to hardcoded defaults if the
    config is missing or malformed.

    @param config_path Path to the per-experiment config.
    @return tuple (g_attract, g_repel, rules, force_scale) where @c rules maps (i, j) -> float.
    """
    FALLBACK_SCALE    = 25.0
    FALLBACK_ATTRACT  = -10.0 / FALLBACK_SCALE   # = -0.4
    FALLBACK_REPEL    =  25.0 / FALLBACK_SCALE   # =  1.0

    cfg = _load_merged_config(config_path)
    if not cfg:
        print("[config] config.json not found — using hardcoded defaults.")
        return -10.0, 25.0, {}, FALLBACK_SCALE

    scale     = float(cfg.get("force_scale", FALLBACK_SCALE))
    g_attract = float(cfg.get("g_attract",   FALLBACK_ATTRACT)) * scale
    g_repel   = float(cfg.get("g_repel",     FALLBACK_REPEL))   * scale

    rules: Dict[Tuple, float] = {}
    for key, val in cfg.get("pair_rules", {}).items():
        try:
            i, j = (int(x) for x in key.split("-"))
            internal_val = float(val) * scale
            rules[(min(i, j), max(i, j))] = internal_val #max(-50.0, min(50.0, internal_val))
        except (ValueError, TypeError):
            print(f"[config] Skipping invalid pair_rules entry: {key!r}: {val!r}")

    print(f"[config] Loaded from config.json — force_scale={scale}, "
          f"g_attract={g_attract:.2f}, g_repel={g_repel:.2f}, "
          f"{len(rules)} pair rule(s)")
    return g_attract, g_repel, rules, scale


CLUSTER_PALETTE = [
    PARTICLE_COLOR_RED,
    PARTICLE_COLOR_YELLOW,
    PARTICLE_COLOR_GREEN,
    PARTICLE_COLOR_BLUE,
    PARTICLE_COLOR_WHITE,
    (0, 255, 255),   # cyan — 6th slot for MAX_CLUSTERS=6
]


@dataclass
class SimSnapshot:
    """@brief Immutable, read-only view of simulation state handed to the GUI thread.

    Built once per physics step under a lock so the renderer never touches live
    mutable state. Holds rendering geometry (particles, targets, trails, obstacles),
    cluster metadata, the current force rules, the live toggles, and rolling metric
    histories for the dashboard.
    """
    particles: list          # [(x, y, radius, color), ...]
    cluster_targets: list    # [(x, y), ...]
    cluster_colors: dict
    obstacles: list          # [(ox, oy, orad), ...]
    target_trails: list      # [[(x, y), ...], ...]
    cfl_round_counter: int
    num_clusters: int
    cluster_stats: dict      # {cluster_id: count}
    g_attract: float
    g_repel: float
    force_scale: float       # config multiplier (internal value = config value * force_scale)
    rules: dict              # {(i,j): float}
    particle_inner_targets: list   # [(tx, ty), ...] parallel to particles — each particle's own target
    cfl_enabled: bool
    attraction_enabled: bool
    avg_loss: float
    avg_confidence: float
    loss_history: list       # per-round avg_loss values (last N rounds)
    conf_history: list       # per-round avg_confidence values (last N rounds)


class SimulationThread(threading.Thread):
    """@brief Owns all simulation state; runs physics + federation in a daemon thread.

    Steps the cognitive and physical domains at @c PHYSICS_HZ, fires an IFCA
    federation round every @c cluster_update_interval steps, drains GUI commands
    from a queue, and publishes an immutable SimSnapshot each step. All state
    mutation happens here so the GUI thread stays read-only.
    """

    def __init__(self, cmd_queue: queue.Queue, config_path: str = DEFAULT_CONFIG_PATH):
        """@brief Construct the simulation thread and initialise all state from config.

        @param cmd_queue Thread-safe queue the GUI uses to send commands (toggles, edits, detonate).
        @param config_path Path to the configuration file to load.
        """
        super().__init__(daemon=True, name="SimThread")
        self.cmd_queue = cmd_queue
        self.config_path = config_path
        self._stop_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._snapshot: Optional[SimSnapshot] = None
        self._init_sim()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_obstacles(spec):
        """@brief Resolve the @c obstacles config field into a list of (x, y, r) tuples.

        Accepted forms: None/missing -> 4 random; int N -> N random; list of
        [x, y, r] or {"x":..,"y":..,"r":..} -> exactly those. Malformed entries are
        skipped with a warning.

        @param spec The raw @c obstacles value from the config.
        @return list of (x, y, radius) obstacle tuples.
        """
        def _random_set(n):
            return [
                (random.randint(150, SIM_WIDTH - 150),
                 random.randint(150, SIM_HEIGHT - 150),
                 random.randint(30, 70))
                for _ in range(n)
            ]

        if spec is None:
            print("[config] obstacles: random (4)")
            return _random_set(4)

        if isinstance(spec, int):
            n = max(0, spec)
            print(f"[config] obstacles: random ({n})")
            return _random_set(n)

        if not isinstance(spec, list):
            print(f"[config] obstacles: invalid type {type(spec).__name__} — using random (4)")
            return _random_set(4)

        out = []
        for idx, item in enumerate(spec):
            try:
                if isinstance(item, dict):
                    x, y, r = int(item["x"]), int(item["y"]), int(item["r"])
                elif isinstance(item, (list, tuple)) and len(item) == 3:
                    x, y, r = int(item[0]), int(item[1]), int(item[2])
                else:
                    raise ValueError("expected [x,y,r] or {x,y,r}")
                if r <= 0:
                    raise ValueError("r must be positive")
                out.append((x, y, r))
            except (KeyError, ValueError, TypeError) as e:
                print(f"[config] obstacles[{idx}] skipped: {item!r} ({e})")

        print(f"[config] obstacles: explicit ({len(out)} from config.json)")
        return out

    def _init_sim(self):
        """@brief Build the initial world from config: clusters, agents, obstacles, forces, logger.

        Spawns one populated group per cluster when @c num_clusters is configured
        (skipping the initial CFL round), or five randomly-assigned groups settled by
        one CFL round otherwise. Also seeds targets, trails, colours and the SimLogger.
        """
        _cfg = _load_merged_config(self.config_path)
        _nc = _cfg.get("num_clusters")
        if isinstance(_nc, int) and MIN_CLUSTERS <= _nc <= MAX_CLUSTERS:
            self.num_clusters = _nc
            _nc_configured = True
            print(f"[config] num_clusters={self.num_clusters} (from config)")
        else:
            self.num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
            _nc_configured = False
            if _nc is not None:
                print(f"[config] num_clusters={_nc!r} out of range [{MIN_CLUSTERS},{MAX_CLUSTERS}] — using random ({self.num_clusters})")
        D = WALL_BOUNDARY * 4
        self.cluster_targets = [
            (random.randint(D, SIM_WIDTH - D), random.randint(D, SIM_HEIGHT - D))
            for _ in range(self.num_clusters)
        ]

        self.cooldown_counter = 0
        self.cluster_ages = {i: BLEND_MATURITY_ROUNDS for i in range(self.num_clusters)}

        trail_maxlen = FRAME_RATE * 3  # ~3 seconds of history at FRAME_RATE sample rate
        self.target_trails: List[deque] = [deque(maxlen=trail_maxlen) for _ in range(self.num_clusters)]
        self._trail_counter = 0

        self.cluster_colors = {i: CLUSTER_PALETTE[i] for i in range(self.num_clusters)}
        self.cluster_colors[-1] = (80, 80, 80)

        self.all_particles = []
        if _nc_configured:
            # Spawn one group per cluster so every cluster starts populated.
            # Skip the initial CFL round — the configured count is authoritative.
            for i in range(self.num_clusters):
                group = instantiateGroup(
                    num=PARTICLE_DEFAULT_SPAWN_NUM,
                    c=PARTICLE_COLOR_WHITE,
                    frame=PARTICLE_DEFAULT_SPAWN_FRAME,
                    target_idx=i,
                )
                for p in group:
                    p.cluster_id = i
                self.all_particles.extend(group)
        else:
            # Random cluster count: spawn 5 groups with random assignment and
            # let the initial CFL round settle the actual cluster structure.
            for _ in range(5):
                t_idx = random.randint(0, self.num_clusters - 1)
                self.all_particles.extend(instantiateGroup(
                    num=PARTICLE_DEFAULT_SPAWN_NUM,
                    c=PARTICLE_COLOR_WHITE,
                    frame=PARTICLE_DEFAULT_SPAWN_FRAME,
                    target_idx=t_idx,
                ))

        self.kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0)
        self.cluster_update_interval = 200  # physics steps between CFL rounds
        self.cluster_update_timer = 0
        self.cfl_round_counter = 0
        self.cfl_enabled        = bool(_cfg.get("cfl_enabled",        True))
        self.attraction_enabled = bool(_cfg.get("attraction_enabled", True))
        _mr = _cfg.get("max_rounds")
        self.max_rounds = int(_mr) if isinstance(_mr, (int, float)) and _mr > 0 else None
        self._avg_loss_history: deque = deque(maxlen=60)
        self._avg_conf_history: deque = deque(maxlen=60)

        self.obstacles = self._init_obstacles(_cfg.get("obstacles"))

        self.g_attract, self.g_repel, self.rules, self.force_scale = _load_force_config(self.config_path)

        if not _nc_configured:
            _, self.kmeans, self.cluster_targets, self.cluster_colors, \
                self.cluster_ages, self.num_clusters, _, self.cooldown_counter = run_cfl_round(
                self.all_particles, self.kmeans, self.cluster_targets,
                self.cluster_colors, self.cluster_ages, self.cooldown_counter,
            )
            max_idx = len(self.cluster_targets) - 1
            for p in self.all_particles:
                if p.cluster_id >= 0:
                    p.target_idx = min(p.cluster_id, max_idx)

        self._sync_rules()

        print(f"\n{'=' * 60}")
        print(f"INIT: Starting Simulation with {len(self.all_particles)} particles "
              f"in {self.num_clusters} clusters.")
        print(f"{'=' * 60}\n")

        _tags = []
        if "cfl_enabled" in _cfg and _cfg["cfl_enabled"]:
            _tags.append("cfl")
        if "attraction_enabled" in _cfg and _cfg["attraction_enabled"]:
            _tags.append("emergent")
        self.logger = SimLogger(
            log_dir="logs",
            tags="_".join(_tags),
            cfl_enabled=self.cfl_enabled,
            attraction_enabled=self.attraction_enabled,
        )

    # ------------------------------------------------------------------
    # Rule helpers
    # ------------------------------------------------------------------

    def _sync_rules(self):
        """@brief Add missing default rules for current cluster pairs and drop stale ones."""
        current_keys = set()
        for i in range(self.num_clusters):
            for j in range(i, self.num_clusters):
                key = (i, j)
                current_keys.add(key)
                if key not in self.rules:
                    self.rules[key] = self.g_attract if i == j else self.g_repel
        for k in [k for k in self.rules if k not in current_keys]:
            del self.rules[k]

    def set_rule(self, i, j, val):
        """@brief Set the internal-scale force for cluster pair (i, j).

        @p val is stored in internal units (config value * force_scale). The GUI
        validates the config-scale input before scaling, so no clamp is applied
        here — clamping would corrupt reset-to-default, whose defaults are scaled too.

        @param i First cluster id. @param j Second cluster id. @param val Internal-scale force.
        """
        self.rules[(min(i, j), max(i, j))] = float(val)

    def reset_rule(self, i, j):
        """@brief Reset the (i, j) force to the default (g_attract if i==j else g_repel)."""
        self.set_rule(i, j, self.g_attract if i == j else self.g_repel)

    # ------------------------------------------------------------------
    # Command handling (called from sim thread only)
    # ------------------------------------------------------------------

    def _handle_cmd(self, cmd):
        """@brief Dispatch one GUI command (detonate, set/reset rule, toggle CFL/attraction).

        @param cmd dict with a @c 'type' key plus type-specific fields.
        """
        t = cmd['type']
        if t == 'detonate':
            self._trigger_explosion()
        elif t == 'set_rule':
            self.set_rule(cmd['i'], cmd['j'], cmd['val'])
        elif t == 'reset_rule':
            self.reset_rule(cmd['i'], cmd['j'])
        elif t == 'toggle_cfl':
            self.cfl_enabled = not self.cfl_enabled
            print(f"\n[CFL] Federation rounds {'ENABLED' if self.cfl_enabled else 'DISABLED'}")
        elif t == 'toggle_attraction':
            self.attraction_enabled = not self.attraction_enabled
            print(f"\n[ATTRACTION] Inter-particle forces {'ENABLED' if self.attraction_enabled else 'DISABLED'}")

    def _trigger_explosion(self):
        """@brief Scatter every agent and randomise its model — a re-convergence stress test."""
        for p in self.all_particles:
            p.vx = random.uniform(-80, 80)
            p.vy = random.uniform(-80, 80)
            p.x += random.uniform(-400, 400)
            p.y += random.uniform(-200, 200)
            p.model = np.array([
                random.uniform(-1, 1), random.uniform(-1, 1),
                0.1, 0.8, 0.0, 0.0, 1.0, 0.0,
            ])
            p.model[0:2] /= np.linalg.norm(p.model[0:2])
        self.logger.log_explosion(self.cfl_round_counter)
        print("\n!!! KA-BOOM !!! - Swarm disrupted.")

    # ------------------------------------------------------------------
    # Physics step
    # ------------------------------------------------------------------

    def _step(self, dt: float):
        """@brief Advance the simulation by one physics tick.

        Drifts each agent's inner target, runs local_train() for every agent,
        fires an IFCA run_cfl_round() when the round timer elapses (handling
        split/merge bookkeeping, metrics and logging), then applies
        update_peer_alignment() and apply_physics_rules() and samples trails.

        @param dt Integration timestep in seconds.
        """
        _DRIFT_SPEED  = 0.10
        _DRIFT_MARGIN = WALL_BOUNDARY * 4

        # Drift each particle's own inner target independently
        for p in self.all_particles:
            p._target_angle += random.uniform(-0.01, 0.01)
            nx = p._target_x + _DRIFT_SPEED * math.cos(p._target_angle)
            ny = p._target_y + _DRIFT_SPEED * math.sin(p._target_angle)
            if nx < _DRIFT_MARGIN or nx > SIM_WIDTH - _DRIFT_MARGIN:
                p._target_angle = math.pi - p._target_angle
                nx = max(_DRIFT_MARGIN, min(SIM_WIDTH - _DRIFT_MARGIN, nx))
            if ny < _DRIFT_MARGIN or ny > SIM_HEIGHT - _DRIFT_MARGIN:
                p._target_angle = -p._target_angle
                ny = max(_DRIFT_MARGIN, min(SIM_HEIGHT - _DRIFT_MARGIN, ny))
            p._target_x = nx
            p._target_y = ny

        # Recompute cluster targets as the average of each cluster's inner targets
        sums: Dict[int, List[float]] = {}
        counts: Dict[int, int] = {}
        for p in self.all_particles:
            cid = p.cluster_id if p.cluster_id >= 0 else p.target_idx
            if cid not in sums:
                sums[cid] = [0.0, 0.0]
                counts[cid] = 0
            sums[cid][0] += p._target_x
            sums[cid][1] += p._target_y
            counts[cid] += 1
        for i in range(len(self.cluster_targets)):
            if i in sums:
                n = counts[i]
                self.cluster_targets[i] = (sums[i][0] / n, sums[i][1] / n)

        # Local training — each particle chases its own inner target
        for p in self.all_particles:
            local_train(p, (p._target_x, p._target_y), self.obstacles, learning_rate=0.02)

        # Simulation round — fires every cluster_update_interval physics steps.
        self.cluster_update_timer += 1
        if self.cluster_update_timer >= self.cluster_update_interval:
            self.cluster_update_timer = 0
            self.cfl_round_counter += 1

            if self.max_rounds is not None and self.cfl_round_counter >= self.max_rounds:
                print(f"\n[SIM] max_rounds={self.max_rounds} reached — stopping simulation.")
                self._stop_event.set()
                pygame.event.post(pygame.event.Event(pygame.QUIT))
                return

            # Track metrics every round regardless of whether CFL is enabled,
            # so the sparkline shows the contrast when the mode is toggled.
            if self.all_particles:
                _al = sum(p.model[6] for p in self.all_particles) / len(self.all_particles)
                _ac = sum(p.model[2] for p in self.all_particles) / len(self.all_particles)
                self._avg_loss_history.append(_al)
                self._avg_conf_history.append(_ac)

            if self.cfl_enabled:
                transfers, self.kmeans, self.cluster_targets, self.cluster_colors, \
                    self.cluster_ages, self.num_clusters, event, self.cooldown_counter = run_cfl_round(
                    self.all_particles, self.kmeans, self.cluster_targets,
                    self.cluster_colors, self.cluster_ages, self.cooldown_counter,
                )

                max_idx = len(self.cluster_targets) - 1
                for p in self.all_particles:
                    if p.cluster_id >= 0:
                        p.target_idx = min(p.cluster_id, max_idx)

                event_type = event[0] if isinstance(event, tuple) else event
                trail_maxlen = self.target_trails[0].maxlen if self.target_trails else FRAME_RATE * 3

                if event_type == 'split':
                    self.target_trails.append(deque(maxlen=trail_maxlen))
                    self._sync_rules()
                    print(f"   > SPLIT — now {self.num_clusters} clusters")
                elif event_type == 'merge':
                    drop_idx = event[1]
                    if drop_idx < len(self.target_trails):
                        del self.target_trails[drop_idx]
                    elif self.target_trails:
                        self.target_trails.pop()
                    self._sync_rules()
                    print(f"   > MERGE — now {self.num_clusters} clusters")

                inertia = self.kmeans.inertia_ if hasattr(self.kmeans, 'inertia_') else 0.0
                counts = {}
                for p in self.all_particles:
                    counts[p.cluster_id] = counts.get(p.cluster_id, 0) + 1

                print(f"\n[ROUND {self.cfl_round_counter}] CFL Complete")
                print(f"   > Inertia:       {inertia:.2f}")
                print(f"   > Cluster Sizes: {dict(sorted(counts.items()))}")
                if transfers:
                    print(f"   > Migrations:")
                    for (old_id, new_id), cnt in sorted(transfers.items(), key=lambda x: -x[1]):
                        src = "Unassigned" if old_id == -1 else f"Cluster {old_id}"
                        print(f"       - {cnt:3d} agents: {src} -> Cluster {new_id}")
                else:
                    print("   > Migrations: (Stable)")
                print("-" * 50)

                self.logger.log_round(
                    round_num=self.cfl_round_counter,
                    particles=self.all_particles,
                    kmeans=self.kmeans,
                    cluster_targets=self.cluster_targets,
                    transfers=transfers,
                    event=event_type,
                    num_clusters=self.num_clusters,
                )
            else:
                print(f"\n[ROUND {self.cfl_round_counter}] (CFL disabled — individual learning only)")
                self.logger.log_round(
                    round_num=self.cfl_round_counter,
                    particles=self.all_particles,
                    kmeans=self.kmeans,
                    cluster_targets=self.cluster_targets,
                    transfers={},
                    event=None,
                    num_clusters=self.num_clusters,
                )

            if self.cfl_round_counter % 20 == 0:
                self.logger.plot_all()

        update_peer_alignment(self.all_particles)
        apply_physics_rules(
            self.all_particles, self.obstacles, self.g_attract, self.g_repel,
            dt, self.rules, attraction_enabled=self.attraction_enabled,
        )

        # Trail sampling at reduced rate
        self._trail_counter += 1
        if self._trail_counter >= TRAIL_SAMPLE_EVERY:
            self._trail_counter = 0
            while len(self.target_trails) < len(self.cluster_targets):
                maxlen = self.target_trails[0].maxlen if self.target_trails else FRAME_RATE * 3
                self.target_trails.append(deque(maxlen=maxlen))
            while len(self.target_trails) > len(self.cluster_targets):
                self.target_trails.pop()
            for i, (tx, ty) in enumerate(self.cluster_targets):
                self.target_trails[i].append((int(tx), int(ty)))

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _build_snapshot(self) -> SimSnapshot:
        """@brief Assemble an immutable SimSnapshot of current state for the GUI.

        @return SimSnapshot a deep-enough copy that the renderer can read lock-free.
        """
        particle_data = [
            (p.x, p.y, p.r, self.cluster_colors.get(p.cluster_id, (255, 255, 255)))
            for p in self.all_particles
        ]
        particle_target_indices = [(p._target_x, p._target_y) for p in self.all_particles]
        stats: Dict[int, int] = {}
        for p in self.all_particles:
            stats[p.cluster_id] = stats.get(p.cluster_id, 0) + 1

        if self.all_particles:
            avg_loss = sum(p.model[6] for p in self.all_particles) / len(self.all_particles)
            avg_conf = sum(p.model[2] for p in self.all_particles) / len(self.all_particles)
        else:
            avg_loss = avg_conf = 0.0

        return SimSnapshot(
            particles=particle_data,
            particle_inner_targets=particle_target_indices,
            cluster_targets=list(self.cluster_targets),
            cluster_colors=dict(self.cluster_colors),
            obstacles=list(self.obstacles),
            target_trails=[list(t) for t in self.target_trails],
            cfl_round_counter=self.cfl_round_counter,
            num_clusters=self.num_clusters,
            cluster_stats=stats,
            g_attract=self.g_attract,
            g_repel=self.g_repel,
            force_scale=self.force_scale,
            rules=dict(self.rules),
            cfl_enabled=self.cfl_enabled,
            attraction_enabled=self.attraction_enabled,
            avg_loss=avg_loss,
            avg_confidence=avg_conf,
            loss_history=list(self._avg_loss_history),
            conf_history=list(self._avg_conf_history),
        )

    def get_snapshot(self) -> Optional[SimSnapshot]:
        """@brief Return the latest published snapshot (thread-safe), or None before the first step."""
        with self._snapshot_lock:
            return self._snapshot

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self):
        """@brief Thread entry point: drain commands, step, publish, and pace to PHYSICS_HZ until stopped."""
        dt = 1.0 / FRAME_RATE  # physics uses same dt the old code used per-step
        period = 1.0 / PHYSICS_HZ

        try:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()

                # Drain commands (all of them before the next physics step)
                while True:
                    try:
                        self._handle_cmd(self.cmd_queue.get_nowait())
                    except queue.Empty:
                        break

                self._step(dt)

                snap = self._build_snapshot()
                with self._snapshot_lock:
                    self._snapshot = snap

                elapsed = time.perf_counter() - t0
                remaining = period - elapsed
                if remaining > 0.0005:
                    time.sleep(remaining)
        finally:
            self.logger.close()

    def stop(self):
        """@brief Signal the thread to stop after the current step (triggers logger teardown)."""
        self._stop_event.set()


# ==============================================================================
# GUI (main thread)
# ==============================================================================

class Game:
    """@brief Main-thread pygame front-end: renders snapshots and forwards input.

    Holds no simulation state of its own — it reads immutable SimSnapshot
    objects from the SimulationThread and posts user actions (rule edits,
    toggles, detonate) back through the command queue.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """@brief Initialise pygame, build the window/widgets, and start the simulation thread.

        @param config_path Path to the configuration file passed through to the simulation.
        """
        pygame.init()
        pygame.key.set_repeat(350, 50)
        self.game_running = True
        self.clock = pygame.time.Clock()

        self.font_header = pygame.font.SysFont("Arial", 22, bold=True)
        self.font        = pygame.font.SysFont("Arial", 16)
        self.font_small  = pygame.font.SysFont("Consolas", 12)

        self.screen = pygame.display.set_mode(size=SCREEN_DIM)
        pygame.display.set_caption("CFL Simulation")

        # Bomb button
        btn_w = 110
        btn_x = SIM_WIDTH + (GUI_WIDTH - btn_w) // 2
        btn_y = SIM_HEIGHT - 50
        self.bomb_rect  = pygame.Rect(btn_x, btn_y, btn_w, 30)
        self.bomb_color = (200, 50, 50)
        self.bomb_text  = self.font.render("DETONATE", True, (255, 255, 255))

        # Matrix editing state (GUI-thread only)
        self.rule_cell_rects: Dict[Tuple, pygame.Rect] = {}
        self.editing_cell   = None
        self.edit_buffer    = ""
        self.last_click_cell = None
        self.last_click_time = 0

        # Toggle button rects — updated each draw call
        self.cfl_toggle_rect        = pygame.Rect(0, 0, 1, 1)
        self.attraction_toggle_rect = pygame.Rect(0, 0, 1, 1)
        self.show_target_lines      = False

        # Start simulation thread
        self.cmd_queue = queue.Queue()
        self.sim = SimulationThread(self.cmd_queue, config_path=config_path)
        self.sim.start()

        # Wait for first snapshot so we have valid state before drawing
        while self.sim.get_snapshot() is None:
            time.sleep(0.005)

    # ------------------------------------------------------------------
    # Matrix editing helpers
    # ------------------------------------------------------------------

    def _commit_edit(self, snap: SimSnapshot):
        """@brief Commit the in-progress rule-cell edit to the simulation, if any.

        @param snap Current snapshot (for context); the parsed value is sent via the command queue.
        """
        if self.editing_cell is None:
            return
        try:
            # edit_buffer holds a config-scale value (e.g. 0.65); convert to the
            # internal scale the simulator uses before sending it across.
            cfg_val = max(-5.0, min(5.0, float(self.edit_buffer)))
            scale = (snap.force_scale if snap else 1.0) or 1.0
            i, j = self.editing_cell
            self.cmd_queue.put({'type': 'set_rule', 'i': i, 'j': j, 'val': cfg_val * scale})
        except (ValueError, TypeError):
            pass
        self.editing_cell = None
        self.edit_buffer = ""

    def _current_rule_val(self, canonical, snap: SimSnapshot) -> float:
        """@brief Look up the current force for a canonical (i, j) pair, falling back to defaults.

        @param canonical (min(i,j), max(i,j)) cluster-pair key.
        @param snap Current snapshot holding the rules table and defaults.
        @return float the effective force for that pair.
        """
        i, j = canonical
        return snap.rules.get(canonical, snap.g_attract if i == j else snap.g_repel)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_gui(self, snap: SimSnapshot):
        """@brief Render the right-hand dashboard: clusters, toggles, metrics, rules matrix, detonate.

        @param snap Snapshot to render from.
        """
        sidebar_rect = pygame.Rect(SIM_WIDTH, 0, GUI_WIDTH, SIM_HEIGHT)
        pygame.draw.rect(self.screen, GUI_BACKGROUND_COLOR, sidebar_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (SIM_WIDTH, 0), (SIM_WIDTH, SIM_HEIGHT), 2)

        start_x = SIM_WIDTH + 24
        y = 24
        line_h = 34

        title = "CFL Dashboard" if snap.cfl_enabled else "Simulation Metrics"
        self.screen.blit(self.font_header.render(title, True, (255, 255, 255)), (start_x, y))
        y += 46

        if snap.cfl_enabled:
            self.screen.blit(self.font.render(f"Round: {snap.cfl_round_counter}", True, (200, 200, 200)), (start_x, y))
            y += int(line_h * 1.4)

        self.screen.blit(self.font.render("Clusters:", True, (255, 255, 255)), (start_x, y))
        y += 22
        cluster_row_h = 18
        for i in range(snap.num_clusters):
            color = snap.cluster_colors.get(i, (255, 255, 255))
            pygame.draw.rect(self.screen, color, (start_x, y + 2, 12, 12))
            count = snap.cluster_stats.get(i, 0)
            self.screen.blit(
                self.font_small.render(f"Cluster {i}: {count} agents", True, (180, 180, 180)),
                (start_x + 20, y + 1),
            )
            y += cluster_row_h

        y += 12
        pygame.draw.line(self.screen, (80, 80, 80), (start_x, y), (SIM_WIDTH + GUI_WIDTH - 20, y), 1)
        y += 8

        # --- Toggles: CFL (federation) + Attraction (emergent physics) ---
        def _toggle_button(rect, label, is_on):
            bg = (30, 110, 30) if is_on else (110, 30, 30)
            br = (50, 160, 50) if is_on else (160, 50, 50)
            pygame.draw.rect(self.screen, bg, rect, border_radius=5)
            pygame.draw.rect(self.screen, br, rect, 2, border_radius=5)
            surf = self.font_small.render(label, True, (240, 240, 240))
            self.screen.blit(surf, surf.get_rect(center=rect.center))

        btn_w = GUI_WIDTH - 40
        self.cfl_toggle_rect = pygame.Rect(start_x, y, btn_w, 26)
        _toggle_button(
            self.cfl_toggle_rect,
            f"Federation (CFL): {'ON' if snap.cfl_enabled else 'OFF'}",
            snap.cfl_enabled,
        )
        y += 40

        self.attraction_toggle_rect = pygame.Rect(start_x, y, btn_w, 26)
        _toggle_button(
            self.attraction_toggle_rect,
            f"Attraction (emergent): {'ON' if snap.attraction_enabled else 'OFF'}",
            snap.attraction_enabled,
        )
        y += 40

        # --- Live metric bars (loss + confidence) ---
        bar_total_w = GUI_WIDTH - 100
        bar_h = 14
        for label, val, bar_color in [
            ("Loss", snap.avg_loss,       (200,  70,  70)),
            ("Conf", snap.avg_confidence, ( 70, 190,  70)),
        ]:
            self.screen.blit(self.font_small.render(label, True, (160, 160, 160)), (start_x, y + 3))
            bx = start_x + 30
            pygame.draw.rect(self.screen, (45, 45, 45), (bx, y + 2, bar_total_w, bar_h), border_radius=3)
            filled_w = int(bar_total_w * max(0.0, min(1.0, val)))
            if filled_w > 0:
                pygame.draw.rect(self.screen, bar_color, (bx, y + 2, filled_w, bar_h), border_radius=3)
            val_str = self.font_small.render(f"{val:.3f}", True, (200, 200, 200))
            self.screen.blit(val_str, (bx + bar_total_w + 4, y + 2))
            y += 30

        # --- Sparkline (last N rounds of avg_loss and avg_confidence) ---
        if len(snap.loss_history) >= 2:
            sk_h = 60
            sk_w = GUI_WIDTH - 48
            sk_x = start_x
            # Labels above the graph
            self.screen.blit(self.font_small.render("— loss", True, (200, 80, 80)), (sk_x, y))
            self.screen.blit(self.font_small.render("— conf", True, (80, 200, 80)), (sk_x + 44, y))
            y += 14
            sk_y = y
            pygame.draw.rect(self.screen, (18, 18, 28), (sk_x, sk_y, sk_w, sk_h), border_radius=3)
            pygame.draw.rect(self.screen, (60, 60, 80), (sk_x, sk_y, sk_w, sk_h), 1, border_radius=3)

            def _spark(vals, color):
                n = len(vals)
                if n < 2:
                    return
                pts = [
                    (sk_x + int(i * (sk_w - 1) / (n - 1)),
                     sk_y + sk_h - 1 - int(max(0.0, min(1.0, v)) * (sk_h - 2)))
                    for i, v in enumerate(vals)
                ]
                pygame.draw.aalines(self.screen, color, False, pts)

            _spark(snap.loss_history, (200, 80, 80))
            _spark(snap.conf_history, (80, 200, 80))
            y += sk_h + 6

        y += 14
        pygame.draw.line(self.screen, (80, 80, 80), (start_x, y), (SIM_WIDTH + GUI_WIDTH - 20, y), 1)
        y += 8

        # --- Rules matrix (hidden when attraction is disabled) ---
        if snap.attraction_enabled:
            self.screen.blit(self.font_small.render("ATTRACTION RULES  (−=attract  +=repel  diag=intra)", True, (160, 160, 160)), (start_x, y))
            y += 16
            self.screen.blit(self.font_small.render("click to edit  ·  double-click to reset", True, (100, 100, 100)), (start_x, y))
            y += 28

            N = snap.num_clusters
            # Matrix shows config-scale values (config value = internal / force_scale),
            # so what you edit matches config.json. Colour range adapts to the data.
            scale = snap.force_scale or 1.0
            _rng = [abs(snap.g_attract) / scale, abs(snap.g_repel) / scale] + \
                   [abs(v) / scale for v in snap.rules.values()]
            color_range = max([1.0] + _rng)
            header_col_w = 20
            cell_w = min(52, (GUI_WIDTH - 48 - header_col_w) // max(1, N))
            cell_h = 26
            mx = start_x
            new_rects = {}

            for j in range(N):
                col_color = snap.cluster_colors.get(j, (200, 200, 200))
                cx = mx + header_col_w + j * cell_w + cell_w // 2 - 6
                pygame.draw.rect(self.screen, col_color, (cx, y + 2, 12, 12))
            y += 18

            for i in range(N):
                pygame.draw.rect(self.screen, snap.cluster_colors.get(i, (200, 200, 200)), (mx, y + 5, 12, 12))
                for j in range(N):
                    canonical = (min(i, j), max(i, j))
                    is_editing = self.editing_cell == canonical
                    disp = (
                        (float(self.edit_buffer) if self.edit_buffer not in ('', '-', '.') else 0.0)
                        if is_editing
                        else self._current_rule_val(canonical, snap) / scale
                    )

                    cx = mx + header_col_w + j * cell_w
                    cell_rect = pygame.Rect(cx + 1, y + 1, cell_w - 2, cell_h - 2)
                    new_rects[(i, j)] = cell_rect

                    t = max(-1.0, min(1.0, disp / color_range))
                    if t < 0:
                        r, g_c, b = int(30 + (1 + t) * 30), int(30 + (1 + t) * 30), int(30 + (-t) * 200)
                    else:
                        r, g_c, b = int(30 + t * 200), int(30 + (1 - t) * 30), int(30 + (1 - t) * 30)

                    cell_bg    = (50, 50, 50)      if is_editing else (r, g_c, b)
                    border_col = (220, 220, 100)   if is_editing else (80, 80, 80)
                    pygame.draw.rect(self.screen, cell_bg, cell_rect)
                    pygame.draw.rect(self.screen, border_col, cell_rect, 1)

                    display_str = (self.edit_buffer + "|") if is_editing else f"{disp:.2f}"
                    val_surf = self.font_small.render(display_str, True, (230, 230, 230))
                    self.screen.blit(val_surf, val_surf.get_rect(center=cell_rect.center))

                y += cell_h

            self.rule_cell_rects = new_rects
        else:
            # Drop click targets and abandon any in-progress edit so a hidden
            # cell can't receive keystrokes or be re-clicked.
            self.rule_cell_rects = {}
            self.editing_cell = None
            self.edit_buffer = ""

        pygame.draw.rect(self.screen, self.bomb_color, self.bomb_rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 100, 100), self.bomb_rect, 2, border_radius=8)
        self.screen.blit(self.bomb_text, self.bomb_text.get_rect(center=self.bomb_rect.center))

    def _draw_sim(self, snap: SimSnapshot):
        """@brief Render the left-hand simulation area: obstacles, target trails, agents.

        @param snap Snapshot to render from.
        """
        self.screen.fill(BACK_BLACK)
        pygame.draw.rect(self.screen, (20, 20, 20), (0, 0, SIM_WIDTH, SIM_HEIGHT))

        for ox, oy, orad in snap.obstacles:
            pygame.draw.circle(self.screen, (50, 50, 50), (ox, oy), orad)
            pygame.draw.circle(self.screen, (150, 50, 50), (ox, oy), orad, 2)

        for idx, trail in enumerate(snap.target_trails):
            if len(trail) >= 2:
                pygame.draw.aalines(self.screen, snap.cluster_colors.get(idx, (255, 255, 255)), False, trail)

        for idx, (tx, ty) in enumerate(snap.cluster_targets):
            color = snap.cluster_colors.get(idx, (255, 255, 255))
            itx, ity = int(tx), int(ty)
            pygame.draw.circle(self.screen, color, (itx, ity), 10, 1)
            pygame.draw.line(self.screen, color, (itx - 15, ity), (itx + 15, ity), 1)
            pygame.draw.line(self.screen, color, (itx, ity - 15), (itx, ity + 15), 1)

        if self.show_target_lines:
            for (px, py, pr, pc), (tx, ty) in zip(snap.particles, snap.particle_inner_targets):
                faded = (pc[0] // 3, pc[1] // 3, pc[2] // 3)
                pygame.draw.line(self.screen, faded, (int(px), int(py)), (int(tx), int(ty)), 1)

        for px, py, pr, pc in snap.particles:
            pygame.draw.circle(self.screen, pc, (int(px), int(py)), pr)

        label = f"[T] Target lines: {'ON' if self.show_target_lines else 'OFF'}"
        hint = self.font_small.render(label, True, (80, 80, 80))
        self.screen.blit(hint, (8, SIM_HEIGHT - hint.get_height() - 6))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """@brief Main GUI loop: poll input, render the latest snapshot, and cap to FRAME_RATE."""
        while self.game_running:
            if self.sim._stop_event.is_set():
                self.game_running = False
                break

            snap = self.sim.get_snapshot()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.bomb_rect.collidepoint(event.pos):
                        self._commit_edit(snap)
                        self.cmd_queue.put({'type': 'detonate'})
                    elif self.cfl_toggle_rect.collidepoint(event.pos):
                        self._commit_edit(snap)
                        self.cmd_queue.put({'type': 'toggle_cfl'})
                    elif self.attraction_toggle_rect.collidepoint(event.pos):
                        self._commit_edit(snap)
                        self.cmd_queue.put({'type': 'toggle_attraction'})
                    else:
                        now = pygame.time.get_ticks()
                        hit = None
                        for (i, j), rect in self.rule_cell_rects.items():
                            if rect.collidepoint(event.pos):
                                hit = (min(i, j), max(i, j))
                                break

                        if hit is not None:
                            if self.last_click_cell == hit and now - self.last_click_time < 400:
                                i, j = hit
                                self.cmd_queue.put({'type': 'reset_rule', 'i': i, 'j': j})
                                self.editing_cell = None
                                self.edit_buffer = ""
                                self.last_click_cell = None
                            else:
                                self._commit_edit(snap)
                                self.editing_cell = hit
                                self.edit_buffer = f"{self._current_rule_val(hit, snap) / (snap.force_scale or 1.0):.2f}"
                                self.last_click_cell = hit
                                self.last_click_time = now
                        else:
                            self._commit_edit(snap)
                            self.last_click_cell = None

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t and self.editing_cell is None:
                        self.show_target_lines = not self.show_target_lines
                    elif self.editing_cell is not None:
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self._commit_edit(snap)
                        elif event.key == pygame.K_ESCAPE:
                            self.editing_cell = None
                            self.edit_buffer = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.edit_buffer = self.edit_buffer[:-1]
                        elif event.unicode in "0123456789":
                            self.edit_buffer += event.unicode
                        elif event.unicode == "-" and self.edit_buffer == "":
                            self.edit_buffer = "-"
                        elif event.unicode == "." and "." not in self.edit_buffer:
                            self.edit_buffer += "."

            if snap is not None:
                self._draw_sim(snap)
                self.draw_gui(snap)
                pygame.display.flip()

            self.clock.tick(FRAME_RATE)

    def quit(self):
        """@brief Stop the simulation thread, close pygame, and wait for logger teardown/plots."""
        self.sim.stop()
        pygame.quit()
        self.sim.join(timeout=60)  # wait for logger.close() / plot_all(), but don't hang forever

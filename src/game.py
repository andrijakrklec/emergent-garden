"""@file game.py
@brief Simulation engine: state, the physics/federation step loop, and snapshots.

This module is the simulation half of the app (runs on a daemon thread):
  - SimulationThread — owns @e all simulation state and steps the physics +
    federation at a fixed rate, publishing immutable SimSnapshot objects.
  - SimSnapshot — the immutable per-step view handed to the GUI.
  - apply_physics_rules() — the physics tick over the agents (glue around the
    pure src.physics force library).

The main-thread pygame front-end (the Game class) lives in gui.py, which imports
SimulationThread / SimSnapshot from here — the dependency is one-directional.

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
    SIM_DIM, SIM_WIDTH, SIM_HEIGHT,
    PARTICLE_DEFAULT_SPAWN_NUM, PARTICLE_DEFAULT_SPAWN_FRAME,
    FRAME_RATE, PARTICLE_COLOR_WHITE, WALL_BOUNDARY, CLUSTER_PALETTE,
    PARTICLE_FORCE_LOWER_RANGE, PARTICLE_FORCE_UPPER_RANGE,
    PARTICLE_POWER_OF_DISTANCE, PARTICLE_LOSE_ENERGY, PARTICLE_MAX_SPEED,
)
from src.particle import (
    Particle, local_train, update_peer_alignment,
)
from src import physics
from src.cfl import instantiate_group, run_cfl_round
from src import cfl_params
from src.utils.sim_logger import SimLogger


# Physics tick rate
PHYSICS_HZ = 60
# Append a trail point every N physics steps (keeps trail density frame-rate-independent)
TRAIL_SAMPLE_EVERY = PHYSICS_HZ // FRAME_RATE  # = 10

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


def _load_merged_config(config_path: str, preset_override: Optional[str] = None) -> dict:
    """@brief Resolve the layered configuration: master + emergent preset + per-run override.

    Three layers are merged, each overriding the previous:
      1. the master config.json (globals plus the @c emergent_preset selector);
      2. the emergent preset it names (`configs/emergent/<name>.json`), if any;
      3. the per-run --config file (used by run_all.py), if different from master.
    The @c pair_rules table is deep-merged at each step; comment keys (those
    starting with '_') are dropped.

    @param config_path Path to the per-run config (may equal the master path).
    @param preset_override If given, overrides the master's @c emergent_preset selector
        (used by the --preset CLI flag to batch over presets).
    @return dict the fully merged configuration.
    """
    def _read(path: str) -> dict:
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            return {k: v for k, v in raw.items() if not k.startswith("_")}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _deep_merge(base: dict, override: dict) -> dict:
        """Shallow-merge two configs, but deep-merge their pair_rules tables."""
        merged = {**base, **override}
        base_rules = base.get("pair_rules", {})
        ovr_rules  = override.get("pair_rules", {})
        if base_rules or ovr_rules:
            merged["pair_rules"] = {**base_rules, **ovr_rules}
        return merged

    cfg = _read(DEFAULT_CONFIG_PATH)

    # Overlay the named emergent preset so switching `emergent_preset` in
    # config.json swaps the whole attract/repel personality in one place.
    preset_name = preset_override if preset_override is not None else cfg.get("emergent_preset")
    if preset_name:
        preset_path = os.path.join(
            os.path.dirname(DEFAULT_CONFIG_PATH), "configs", "emergent", f"{preset_name}.json"
        )
        preset = _read(preset_path)
        if preset:
            cfg = _deep_merge(cfg, preset)
        else:
            print(f"[config] emergent_preset {preset_name!r} not found at {preset_path} — ignoring.")

    # Overlay the per-run --config file (run_all.py ablation), if any.
    if os.path.abspath(config_path) == os.path.abspath(DEFAULT_CONFIG_PATH):
        return cfg
    override = _read(config_path)
    if not override:
        return cfg
    return _deep_merge(cfg, override)


def _load_force_config(config_path: str, preset_override: Optional[str] = None) -> Tuple[float, float, Dict, float]:
    """@brief Load g_attract, g_repel and the optional pair_rules from merged config.

    Config force values are in a readable ~-1..+1 scale; @c force_scale multiplies
    them into internal units before use. Falls back to hardcoded defaults if the
    config is missing or malformed.

    @param config_path Path to the per-experiment config.
    @param preset_override If given, overrides the master's @c emergent_preset selector
        (passed through to _load_merged_config; used by the --preset CLI flag).
    @return tuple (g_attract, g_repel, rules, force_scale) where @c rules maps (i, j) -> float.
    """
    FALLBACK_SCALE    = 25.0
    FALLBACK_ATTRACT  = -10.0 / FALLBACK_SCALE   # = -0.4
    FALLBACK_REPEL    =  25.0 / FALLBACK_SCALE   # =  1.0

    cfg = _load_merged_config(config_path, preset_override)
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


def apply_physics_rules(
        particles: List[Particle],
        obstacles: List[Tuple[int, int, int]],
        g_attract: float,
        g_repel: float,
        dt: float,
        rules: dict = None,
        attraction_enabled: bool = True):
    """@brief Integrate one physics tick over the agents (the physics 'system').

    The simulation-side glue around the pure @ref physics force library: it
    loops over Particle objects, resolves each pair's force via
    @ref physics.pair_force (emergency repulsion + the standard gravity formula),
    adds soft-obstacle and model-directed behavioral pushes, then integrates
    velocities/positions and resolves hard obstacle and wall collisions.

    @param particles List of all Particle agents; positions/velocities mutated in place.
    @param obstacles List of (ox, oy, radius) obstacle circles.
    @param g_attract Default intra-cluster coefficient (negative attracts).
    @param g_repel Default inter-cluster coefficient (positive repels).
    @param dt Integration timestep in seconds.
    @param rules Optional dict (min(ci,cj), max(ci,cj)) -> coefficient for per-pair forces.
    @param attraction_enabled If False, intra/inter-cluster forces are disabled and only the
        emergency anti-stacking repulsion remains — used to ablate the emergent physics.
    @return None.
    """
    n = len(particles)
    forces = [np.zeros(2) for _ in particles]

    # --- Pairwise inter-agent forces (emergency repulsion + standard gravity) ---
    upper_sq = PARTICLE_FORCE_UPPER_RANGE ** 2
    for i in range(n):
        a = particles[i]
        for j in range(i + 1, n):
            b = particles[j]

            dx = a.x - b.x
            dy = a.y - b.y
            d_sq = dx ** 2 + dy ** 2

            # Cull far pairs before any sqrt / force math.
            if d_sq > upper_sq:
                continue
            if d_sq == 0:
                d_sq = 0.001
            d = d_sq ** 0.5

            coeff = physics.interaction_coefficient(
                a.cluster_id, b.cluster_id, g_attract, g_repel, rules)
            f_scalar = physics.pair_force(
                d, coeff, n, PARTICLE_FORCE_LOWER_RANGE,
                power=PARTICLE_POWER_OF_DISTANCE,
                attraction_enabled=attraction_enabled,
            )

            fx = f_scalar * dx
            fy = f_scalar * dy
            forces[i] += np.array([fx, fy])
            forces[j] -= np.array([fx, fy])

    # --- Soft obstacle zone: pre-contact repulsion so agents redirect early ---
    soft_zone = cfl_params.get_obstacle_soft_zone()
    soft_strength = cfl_params.get_obstacle_soft_strength()
    for i, p in enumerate(particles):
        for ox, oy, orad in obstacles:
            dx, dy = p.x - ox, p.y - oy
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            push = physics.soft_obstacle_push(dist, orad + p.r, soft_zone, soft_strength)
            forces[i][0] += (dx / dist) * push
            forces[i][1] += (dy / dist) * push

    # --- Behavioral force: model-directed push scaled by confidence ---
    # model[3] (obstacle_pressure) amplifies the force so sustained pressure
    # translates directly into a stronger escape push.
    behavioral_force = cfl_params.get_behavioral_force()
    for i, p in enumerate(particles):
        if p.cluster_id != -1 and p.model[2] > 0.15:
            pressure_boost = 1.0 + float(p.model[3]) * 1.25
            strength = behavioral_force * float(p.model[2]) * pressure_boost
            forces[i][0] += p.model[0] * strength
            forces[i][1] += p.model[1] * strength

    # --- Integrate velocities/positions, resolve obstacle + wall collisions ---
    for i, p in enumerate(particles):
        fx, fy = forces[i]

        p.vx = (p.vx + fx * dt) * PARTICLE_LOSE_ENERGY
        p.vy = (p.vy + fy * dt) * PARTICLE_LOSE_ENERGY
        p.vx, p.vy = physics.clamp_speed(p.vx, p.vy, PARTICLE_MAX_SPEED)

        p.x += p.vx * dt
        p.y += p.vy * dt

        # Hard physical collisions with obstacles: push out + bounce.
        for ox, oy, orad in obstacles:
            dx, dy = p.x - ox, p.y - oy
            dist = math.hypot(dx, dy)
            if dist < orad + p.r and dist > 0:
                overlap = (orad + p.r) - dist
                nx, ny = dx / dist, dy / dist
                p.x += nx * overlap
                p.y += ny * overlap
                p.vx, p.vy = physics.reflect_velocity(p.vx, p.vy, nx, ny, restitution=0.8)

        # Wall collisions (simulation bounds, not the window).
        V = 0.9
        D = WALL_BOUNDARY
        if p.x < D:
            p.x = D
            p.vx *= -V
        if p.x > SIM_DIM[0] - D:
            p.x = SIM_DIM[0] - D
            p.vx *= -V
        if p.y < D:
            p.y = D
            p.vy *= -V
        if p.y > SIM_DIM[1] - D:
            p.y = SIM_DIM[1] - D
            p.vy *= -V


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

    def __init__(self, cmd_queue: queue.Queue, config_path: str = DEFAULT_CONFIG_PATH,
                 log_dir: str = "logs", run_name: Optional[str] = None, preset: Optional[str] = None):
        """@brief Construct the simulation thread and initialize all state from config.

        @param cmd_queue Thread-safe queue the GUI uses to send commands (toggles, edits, detonate).
        @param config_path Path to the configuration file to load.
        @param log_dir Base directory for the run's log folder.
        @param run_name Explicit run-folder name (overrides the timestamped default); for batch runs.
        @param preset Overrides the emergent_preset from config.json (for batch runs over presets).
        """
        super().__init__(daemon=True, name="SimThread")
        self.cmd_queue = cmd_queue
        self.config_path = config_path
        self.log_dir = log_dir
        self.run_name = run_name
        self.preset = preset
        self._stop_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._snapshot: Optional[SimSnapshot] = None
        self._init_sim()

    # ------------------------------------------------------------------
    # Initialization
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
        one CFL round otherwise. Also seeds targets, trails, colors and the SimLogger.
        """
        _cfg = _load_merged_config(self.config_path, self.preset)
        _nc = _cfg.get("num_clusters")
        _min_clusters = cfl_params.get_min_clusters()
        _max_clusters = cfl_params.get_max_clusters()

        if isinstance(_nc, int) and _min_clusters <= _nc <= _max_clusters:
            self.num_clusters = _nc
            print(f"[config] num_clusters={self.num_clusters} (from config)")
        else:
            self.num_clusters = random.randint(_min_clusters, _max_clusters)
            if _nc is not None:
                print(f"[config] num_clusters={_nc!r} out of range [{_min_clusters},{_max_clusters}] — using random ({self.num_clusters})")

        D = WALL_BOUNDARY * 4

        # Per-true-cluster spatial anchors: evenly spaced on a circle inside the
        # sim. All members of true cluster i share anchor i as their _target_x/_y
        # so cluster members have a SHARED goal — that shared signal is what CFL
        # aggregation can recover, while individual learning cannot.
        _cx, _cy = SIM_WIDTH / 2.0, SIM_HEIGHT / 2.0
        _radius = min(SIM_WIDTH, SIM_HEIGHT) / 2.0 - D
        self._true_anchors: List[Tuple[float, float]] = [
            (_cx + _radius * math.cos(2 * math.pi * i / self.num_clusters),
             _cy + _radius * math.sin(2 * math.pi * i / self.num_clusters))
            for i in range(self.num_clusters)
        ]
        self._true_anchor_angles = [random.uniform(0, 2 * math.pi) for _ in range(self.num_clusters)]

        # Pin initial cluster_targets to the true anchors so IFCA has a
        # non-degenerate spatial loss landscape from round 1 (otherwise the
        # random init scrambles them to ~the sim centroid).
        self.cluster_targets = [(int(ax), int(ay)) for ax, ay in self._true_anchors]

        # Shared anchors pin K — disable IFCA split/merge from the start.
        self.cooldown_counter = 10**6
        self.cluster_ages = {i: cfl_params.get_blend_maturity_rounds() for i in range(self.num_clusters)}

        trail_maxlen = FRAME_RATE * 3  # ~3 seconds of history at FRAME_RATE sample rate
        self.target_trails: List[deque] = [deque(maxlen=trail_maxlen) for _ in range(self.num_clusters)]
        self._trail_counter = 0

        self.cluster_colors = {i: CLUSTER_PALETTE[i] for i in range(self.num_clusters)}
        self.cluster_colors[-1] = (80, 80, 80)

        # Spawn one group per true cluster. Each group shares a latent bias
        # direction (evenly spaced around 2π) AND a shared spatial anchor —
        # together those constitute the ground-truth latent structure CFL is
        # meant to recover. Groups spawn in the global center box, but each has
        # _target_x/_y pinned to its own perimeter anchor — so particles start
        # far from their target, geometric loss is large at round 1, and the
        # loss curve shows a real per-round decrease as the swarm navigates to
        # its anchor (spawning AT the anchor would flatten the curve at the
        # bias-limited floor). Initial cluster_id is randomized so init purity
        # ≈ 1/K — CFL must do the discovery work, CFL-off stays at the baseline.
        _cluster_bias_dirs = [
            (math.cos(2 * math.pi * i / self.num_clusters),
             math.sin(2 * math.pi * i / self.num_clusters))
            for i in range(self.num_clusters)
        ]

        self.all_particles = []
        for i in range(self.num_clusters):
            ax, ay = self._true_anchors[i]
            group = instantiate_group(
                num=PARTICLE_DEFAULT_SPAWN_NUM,
                c=PARTICLE_COLOR_WHITE,
                frame=PARTICLE_DEFAULT_SPAWN_FRAME,
                target_idx=i,
                true_cluster_id=i,
                cluster_bias_dir=_cluster_bias_dirs[i],
            )
            for p in group:
                p._target_x = float(ax)
                p._target_y = float(ay)
                p._target_angle = self._true_anchor_angles[i]
                p.cluster_id = random.randint(0, self.num_clusters - 1)
            self.all_particles.extend(group)

        self.kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0)

        # Physics steps between CFL rounds. Shorter → model isn't fully converged
        # between rounds, so the per-round loss trajectory shows real convergence
        # across rounds (not just a flat line at the bias-limited floor).
        # Configurable via config.json's "cluster_update_interval".
        _cui = _cfg.get("cluster_update_interval")
        self.cluster_update_interval = int(_cui) if isinstance(_cui, (int, float)) and _cui > 0 else 50

        # Rounds during which IFCA split/merge is suppressed (only relevant in
        # shared-anchor mode where K is pinned to K_true). Lets IFCA find the
        # latent structure first, then unpins so dynamic-K behavior can emerge.
        # Configurable via config.json's "split_merge_start_round". 0 = never
        # pin; very large = always pin.
        _smsr = _cfg.get("split_merge_start_round")
        self.split_merge_start_round = int(_smsr) if isinstance(_smsr, (int, float)) and _smsr >= 0 else 20
        self.cluster_update_timer = 0
        self.cfl_round_counter = 0
        self.cfl_enabled        = bool(_cfg.get("cfl_enabled",        True))
        self.attraction_enabled = bool(_cfg.get("attraction_enabled", True))
        _mr = _cfg.get("max_rounds")
        self.max_rounds = int(_mr) if isinstance(_mr, (int, float)) and _mr > 0 else None
        self._avg_loss_history: deque = deque(maxlen=60)
        self._avg_conf_history: deque = deque(maxlen=60)

        self.obstacles = self._init_obstacles(_cfg.get("obstacles"))

        self.g_attract, self.g_repel, self.rules, self.force_scale = _load_force_config(self.config_path, self.preset)

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
            log_dir=self.log_dir,
            tags="_".join(_tags),
            cfl_enabled=self.cfl_enabled,
            attraction_enabled=self.attraction_enabled,
            run_name=self.run_name,
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
        """@brief Scatter every agent and randomize its model — a re-convergence stress test."""
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

        # Drift K anchors (one per true cluster), then refresh each particle's
        # _target from its true cluster's anchor. This keeps all members of a
        # true cluster locked to a SHARED goal — federation can cancel the
        # remaining per-particle bias noise; individual learning cannot.
        for i in range(len(self._true_anchors)):
            ang = self._true_anchor_angles[i] + random.uniform(-0.01, 0.01)
            ax, ay = self._true_anchors[i]
            nx = ax + _DRIFT_SPEED * math.cos(ang)
            ny = ay + _DRIFT_SPEED * math.sin(ang)
            if nx < _DRIFT_MARGIN or nx > SIM_WIDTH - _DRIFT_MARGIN:
                ang = math.pi - ang
                nx = max(_DRIFT_MARGIN, min(SIM_WIDTH - _DRIFT_MARGIN, nx))
            if ny < _DRIFT_MARGIN or ny > SIM_HEIGHT - _DRIFT_MARGIN:
                ang = -ang
                ny = max(_DRIFT_MARGIN, min(SIM_HEIGHT - _DRIFT_MARGIN, ny))
            self._true_anchors[i] = (nx, ny)
            self._true_anchor_angles[i] = ang

        for p in self.all_particles:
            if 0 <= p._true_cluster_id < len(self._true_anchors):
                p._target_x, p._target_y = self._true_anchors[p._true_cluster_id]

        # Pin cluster_targets[k] = anchor_k for k < K_true so the IFCA loss
        # landscape stays aligned with the ground truth. Any extra entries
        # introduced by IFCA splits keep whatever value run_cfl_round set.
        for i in range(min(len(self.cluster_targets), len(self._true_anchors))):
            self.cluster_targets[i] = self._true_anchors[i]

        # Compute per-particle emergent pull (net attract/repel from neighbors)
        # so local_train can fold it into the learned heading. Same force topology
        # as apply_physics_rules — respects the rules table, falls back to
        # g_attract/g_repel by intra/inter-cluster. Skipped when attraction is
        # off so the emergent_pull defaults to (0,0) and behavior is identical
        # to the pre-emergent-model code path.
        n_parts = len(self.all_particles)
        emergent_pulls = [(0.0, 0.0)] * n_parts
        if self.attraction_enabled and n_parts > 1:
            from src.constants import PARTICLE_FORCE_UPPER_RANGE
            _upper_sq = PARTICLE_FORCE_UPPER_RANGE ** 2
            for i, p in enumerate(self.all_particles):
                if p.cluster_id == -1:
                    continue
                ex, ey = 0.0, 0.0
                for j, other in enumerate(self.all_particles):
                    if i == j or other.cluster_id == -1:
                        continue
                    dx = p.x - other.x
                    dy = p.y - other.y
                    d_sq = dx * dx + dy * dy
                    if d_sq > _upper_sq or d_sq < 1.0:
                        continue
                    ci, cj = p.cluster_id, other.cluster_id
                    if self.rules:
                        g = self.rules.get((min(ci, cj), max(ci, cj)),
                                           self.g_attract if ci == cj else self.g_repel)
                    else:
                        g = self.g_attract if ci == cj else self.g_repel
                    d = math.sqrt(d_sq)
                    F = g / (d * n_parts)
                    ex += F * dx
                    ey += F * dy
                emergent_pulls[i] = (ex, ey)

        # Local training — each particle chases its own inner target
        for i, p in enumerate(self.all_particles):
            local_train(p, (p._target_x, p._target_y), self.obstacles,
                        learning_rate=0.02, emergent_pull=emergent_pulls[i])

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

                # Shared-anchor mode pins K=K_true for the first
                # `split_merge_start_round` rounds so IFCA can find the latent
                # structure cleanly. After that, the pin is released and IFCA's
                # split/merge becomes active so K can drift adaptively. Without
                # the explicit reset, the leftover 999_999 from the pin would
                # take a million rounds to decrement back to 0.
                if self._true_anchors:
                    if self.cfl_round_counter < self.split_merge_start_round:
                        self.cooldown_counter = 10**6
                    elif self.cooldown_counter > 100:
                        self.cooldown_counter = 0

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

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


# Physics tick rate — same as old SIM_STEPS_PER_FRAME * FRAME_RATE
PHYSICS_HZ = 250
# Append a trail point every N physics steps (keeps trail density frame-rate-independent)
TRAIL_SAMPLE_EVERY = PHYSICS_HZ // FRAME_RATE  # = 10

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")

def _load_force_config() -> Tuple[float, float, Dict]:
    """Load g_attract, g_repel, and optional pair_rules from config.json.

    Config values are in "original simulation scale" (roughly -1 to +1).
    force_scale multiplies all of them before use so the physics match the
    original emergent-garden simulation's feel.  Defaults give the same
    internal values as the hardcoded fallback (-10, 25).
    Falls back to hardcoded defaults if the file is missing or malformed.
    """
    FALLBACK_SCALE    = 25.0
    FALLBACK_ATTRACT  = -10.0 / FALLBACK_SCALE   # = -0.4
    FALLBACK_REPEL    =  25.0 / FALLBACK_SCALE   # =  1.0

    try:
        with open(CONFIG_PATH, "r") as f:
            raw = json.load(f)
        # Strip comment keys (keys starting with "_") so users can annotate the file
        cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    except FileNotFoundError:
        print("[config] config.json not found — using hardcoded defaults.")
        return -10.0, 25.0, {}
    except json.JSONDecodeError as e:
        print(f"[config] config.json parse error: {e} — using hardcoded defaults.")
        return -10.0, 25.0, {}

    scale     = float(cfg.get("force_scale", FALLBACK_SCALE))
    g_attract = float(cfg.get("g_attract",   FALLBACK_ATTRACT)) * scale
    g_repel   = float(cfg.get("g_repel",     FALLBACK_REPEL))   * scale

    rules: Dict[Tuple, float] = {}
    for key, val in cfg.get("pair_rules", {}).items():
        try:
            i, j = (int(x) for x in key.split("-"))
            internal_val = float(val) * scale
            rules[(min(i, j), max(i, j))] = max(-50.0, min(50.0, internal_val))
        except (ValueError, TypeError):
            print(f"[config] Skipping invalid pair_rules entry: {key!r}: {val!r}")

    print(f"[config] Loaded from config.json — force_scale={scale}, "
          f"g_attract={g_attract:.2f}, g_repel={g_repel:.2f}, "
          f"{len(rules)} pair rule(s)")
    return g_attract, g_repel, rules


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
    """Lightweight read-only view of sim state, safe to hand to the GUI thread."""
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
    rules: dict              # {(i,j): float}


class SimulationThread(threading.Thread):
    """Owns all simulation state. Runs physics at PHYSICS_HZ in a daemon thread."""

    def __init__(self, cmd_queue: queue.Queue):
        super().__init__(daemon=True, name="SimThread")
        self.cmd_queue = cmd_queue
        self._stop_event = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._snapshot: Optional[SimSnapshot] = None
        self._init_sim()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_sim(self):
        try:
            with open(CONFIG_PATH) as f:
                _cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
            _nc = _cfg.get("num_clusters")
            if isinstance(_nc, int) and MIN_CLUSTERS <= _nc <= MAX_CLUSTERS:
                self.num_clusters = _nc
                print(f"[config] num_clusters={self.num_clusters} (from config.json)")
            else:
                self.num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
                if _nc is not None:
                    print(f"[config] num_clusters={_nc!r} out of range [{MIN_CLUSTERS},{MAX_CLUSTERS}] — using random ({self.num_clusters})")
        except Exception:
            self.num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
        D = WALL_BOUNDARY * 4
        self.cluster_targets = [
            (random.randint(D, SIM_WIDTH - D), random.randint(D, SIM_HEIGHT - D))
            for _ in range(self.num_clusters)
        ]

        self.cooldown_counter = 0
        self.cluster_ages = {i: BLEND_MATURITY_ROUNDS for i in range(self.num_clusters)}
        self.target_angles = [random.uniform(0, 2 * math.pi) for _ in range(self.num_clusters)]

        trail_maxlen = FRAME_RATE * 3  # ~3 seconds of history at FRAME_RATE sample rate
        self.target_trails: List[deque] = [deque(maxlen=trail_maxlen) for _ in range(self.num_clusters)]
        self._trail_counter = 0

        self.cluster_colors = {i: CLUSTER_PALETTE[i] for i in range(self.num_clusters)}
        self.cluster_colors[-1] = (80, 80, 80)

        self.all_particles = []
        for _ in range(5):
            t_idx = random.randint(0, self.num_clusters - 1)
            self.all_particles.extend(instantiateGroup(
                num=PARTICLE_DEFAULT_SPAWN_NUM,
                c=PARTICLE_COLOR_WHITE,
                frame=PARTICLE_DEFAULT_SPAWN_FRAME,
                target_idx=t_idx,
            ))

        self.kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0)
        self.cluster_update_interval = 100  # physics steps between CFL rounds
        self.cluster_update_timer = 0
        self.cfl_round_counter = 0

        self.obstacles = []
        for _ in range(4):
            ox = random.randint(150, SIM_WIDTH - 150)
            oy = random.randint(150, SIM_HEIGHT - 150)
            orad = random.randint(30, 70)
            self.obstacles.append((ox, oy, orad))

        self.g_attract, self.g_repel, self.rules = _load_force_config()

        print(f"\n{'=' * 60}")
        print(f"INIT: Starting Simulation with {len(self.all_particles)} particles.")
        print(f"{'=' * 60}\n")

        _, self.kmeans, self.cluster_targets, self.cluster_colors, \
            self.cluster_ages, self.num_clusters, _, self.cooldown_counter = run_cfl_round(
            self.all_particles, self.kmeans, self.cluster_targets,
            self.cluster_colors, self.cluster_ages, self.cooldown_counter,
        )
        self._sync_rules()

        max_idx = len(self.cluster_targets) - 1
        for p in self.all_particles:
            if p.cluster_id >= 0:
                p.target_idx = min(p.cluster_id, max_idx)

        self.logger = SimLogger(log_dir="logs")

    # ------------------------------------------------------------------
    # Rule helpers
    # ------------------------------------------------------------------

    def _sync_rules(self):
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
        self.rules[(min(i, j), max(i, j))] = max(-50.0, min(50.0, val))

    def reset_rule(self, i, j):
        self.set_rule(i, j, self.g_attract if i == j else self.g_repel)

    # ------------------------------------------------------------------
    # Command handling (called from sim thread only)
    # ------------------------------------------------------------------

    def _handle_cmd(self, cmd):
        t = cmd['type']
        if t == 'detonate':
            self._trigger_explosion()
        elif t == 'set_rule':
            self.set_rule(cmd['i'], cmd['j'], cmd['val'])
        elif t == 'reset_rule':
            self.reset_rule(cmd['i'], cmd['j'])

    def _trigger_explosion(self):
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
        _DRIFT_SPEED  = 0.35
        _DRIFT_MARGIN = WALL_BOUNDARY * 4

        # Sync angle list length with targets
        while len(self.target_angles) < len(self.cluster_targets):
            self.target_angles.append(random.uniform(0, 2 * math.pi))
        while len(self.target_angles) > len(self.cluster_targets):
            self.target_angles.pop()

        # Drift targets
        for i, (tx, ty) in enumerate(self.cluster_targets):
            self.target_angles[i] += random.uniform(-0.01, 0.01)
            nx = tx + _DRIFT_SPEED * math.cos(self.target_angles[i])
            ny = ty + _DRIFT_SPEED * math.sin(self.target_angles[i])
            if nx < _DRIFT_MARGIN or nx > SIM_WIDTH - _DRIFT_MARGIN:
                self.target_angles[i] = math.pi - self.target_angles[i]
                nx = max(_DRIFT_MARGIN, min(SIM_WIDTH - _DRIFT_MARGIN, nx))
            if ny < _DRIFT_MARGIN or ny > SIM_HEIGHT - _DRIFT_MARGIN:
                self.target_angles[i] = -self.target_angles[i]
                ny = max(_DRIFT_MARGIN, min(SIM_HEIGHT - _DRIFT_MARGIN, ny))
            self.cluster_targets[i] = (nx, ny)

        # Local training
        for p in self.all_particles:
            local_train(p, self.cluster_targets[p.target_idx], self.obstacles, learning_rate=0.05)

        # CFL round — fires every cluster_update_interval physics steps,
        # back-to-back as fast as computation allows (mirrors the notebook's for-loop).
        self.cluster_update_timer += 1
        if self.cluster_update_timer >= self.cluster_update_interval:
            self.cluster_update_timer = 0
            self.cfl_round_counter += 1

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
                self.target_angles.append(random.uniform(0, 2 * math.pi))
                self.target_trails.append(deque(maxlen=trail_maxlen))
                self._sync_rules()
                print(f"   > SPLIT — now {self.num_clusters} clusters")
            elif event_type == 'merge':
                drop_idx = event[1]
                if drop_idx < len(self.target_angles):
                    del self.target_angles[drop_idx]
                elif self.target_angles:
                    self.target_angles.pop()
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
            if self.cfl_round_counter % 20 == 0:
                self.logger.plot_all()

        update_peer_alignment(self.all_particles)
        apply_physics_rules(self.all_particles, self.obstacles, self.g_attract, self.g_repel, dt, self.rules)

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
        particle_data = [
            (p.x, p.y, p.r, self.cluster_colors.get(p.cluster_id, (255, 255, 255)))
            for p in self.all_particles
        ]
        stats: Dict[int, int] = {}
        for p in self.all_particles:
            stats[p.cluster_id] = stats.get(p.cluster_id, 0) + 1

        return SimSnapshot(
            particles=particle_data,
            cluster_targets=list(self.cluster_targets),
            cluster_colors=dict(self.cluster_colors),
            obstacles=list(self.obstacles),
            target_trails=[list(t) for t in self.target_trails],
            cfl_round_counter=self.cfl_round_counter,
            num_clusters=self.num_clusters,
            cluster_stats=stats,
            g_attract=self.g_attract,
            g_repel=self.g_repel,
            rules=dict(self.rules),
        )

    def get_snapshot(self) -> Optional[SimSnapshot]:
        with self._snapshot_lock:
            return self._snapshot

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self):
        dt = 1.0 / FRAME_RATE  # physics uses same dt the old code used per-step
        period = 1.0 / PHYSICS_HZ

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

    def stop(self):
        self._stop_event.set()
        self.logger.close()


# ==============================================================================
# GUI (main thread)
# ==============================================================================

class Game:
    """Handles pygame events and rendering. All simulation state lives in SimulationThread."""

    def __init__(self):
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
        btn_x = SIM_WIDTH + 20
        btn_y = SIM_HEIGHT - 80
        self.bomb_rect  = pygame.Rect(btn_x, btn_y, GUI_WIDTH - 40, 50)
        self.bomb_color = (200, 50, 50)
        self.bomb_text  = self.font_header.render("DETONATE", True, (255, 255, 255))

        # Matrix editing state (GUI-thread only)
        self.rule_cell_rects: Dict[Tuple, pygame.Rect] = {}
        self.editing_cell   = None
        self.edit_buffer    = ""
        self.last_click_cell = None
        self.last_click_time = 0

        # Start simulation thread
        self.cmd_queue = queue.Queue()
        self.sim = SimulationThread(self.cmd_queue)
        self.sim.start()

        # Wait for first snapshot so we have valid state before drawing
        while self.sim.get_snapshot() is None:
            time.sleep(0.005)

    # ------------------------------------------------------------------
    # Matrix editing helpers
    # ------------------------------------------------------------------

    def _commit_edit(self, snap: SimSnapshot):
        if self.editing_cell is None:
            return
        try:
            val = float(self.edit_buffer)
            i, j = self.editing_cell
            self.cmd_queue.put({'type': 'set_rule', 'i': i, 'j': j, 'val': val})
        except (ValueError, TypeError):
            pass
        self.editing_cell = None
        self.edit_buffer = ""

    def _current_rule_val(self, canonical, snap: SimSnapshot) -> float:
        i, j = canonical
        return snap.rules.get(canonical, snap.g_attract if i == j else snap.g_repel)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_gui(self, snap: SimSnapshot):
        sidebar_rect = pygame.Rect(SIM_WIDTH, 0, GUI_WIDTH, SIM_HEIGHT)
        pygame.draw.rect(self.screen, GUI_BACKGROUND_COLOR, sidebar_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (SIM_WIDTH, 0), (SIM_WIDTH, SIM_HEIGHT), 2)

        start_x = SIM_WIDTH + 20
        y = 20
        line_h = 30

        self.screen.blit(self.font_header.render("CFL Dashboard", True, (255, 255, 255)), (start_x, y))
        y += 40

        total = sum(snap.cluster_stats.values())
        self.screen.blit(self.font.render(f"Round: {snap.cfl_round_counter}", True, (200, 200, 200)), (start_x, y))
        y += line_h
        self.screen.blit(self.font.render(f"Particles: {total}", True, (200, 200, 200)), (start_x, y))
        y += int(line_h * 1.5)

        self.screen.blit(self.font.render("Active Clusters:", True, (255, 255, 255)), (start_x, y))
        y += line_h
        for i in range(snap.num_clusters):
            color = snap.cluster_colors.get(i, (255, 255, 255))
            pygame.draw.rect(self.screen, color, (start_x, y + 5, 15, 15))
            count = snap.cluster_stats.get(i, 0)
            self.screen.blit(
                self.font.render(f"Cluster {i}: {count} agents", True, (180, 180, 180)),
                (start_x + 25, y),
            )
            y += line_h

        y += 20
        pygame.draw.line(self.screen, (80, 80, 80), (start_x, y), (SIM_WIDTH + GUI_WIDTH - 20, y), 1)
        y += 10

        # --- Rules matrix ---
        self.screen.blit(self.font_small.render("ATTRACTION RULES", True, (200, 200, 200)), (start_x, y))
        y += 14
        for line in (
            "Force between cluster pairs",
            "negative = attract, positive = repel",
            "diagonal = intra cluster",
            "click = edit  |  dbl-click = reset",
        ):
            color = (110, 110, 110) if "click" in line else (140, 140, 140)
            self.screen.blit(self.font_small.render(line, True, color), (start_x, y))
            y += 13
        y += 4

        N = snap.num_clusters
        header_col_w = 18
        cell_w = min(45, (GUI_WIDTH - 40 - header_col_w) // max(1, N))
        cell_h = 22
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
                val = (
                    (float(self.edit_buffer) if self.edit_buffer not in ('', '-', '.') else 0.0)
                    if is_editing
                    else self._current_rule_val(canonical, snap)
                )

                cx = mx + header_col_w + j * cell_w
                cell_rect = pygame.Rect(cx + 1, y + 1, cell_w - 2, cell_h - 2)
                new_rects[(i, j)] = cell_rect

                t = max(-1.0, min(1.0, val / 50.0))
                if t < 0:
                    r, g_c, b = int(30 + (1 + t) * 30), int(30 + (1 + t) * 30), int(30 + (-t) * 200)
                else:
                    r, g_c, b = int(30 + t * 200), int(30 + (1 - t) * 30), int(30 + (1 - t) * 30)

                cell_bg    = (50, 50, 50)      if is_editing else (r, g_c, b)
                border_col = (220, 220, 100)   if is_editing else (80, 80, 80)
                pygame.draw.rect(self.screen, cell_bg, cell_rect)
                pygame.draw.rect(self.screen, border_col, cell_rect, 1)

                display_str = (self.edit_buffer + "|") if is_editing else f"{val:.0f}"
                val_surf = self.font_small.render(display_str, True, (230, 230, 230))
                self.screen.blit(val_surf, val_surf.get_rect(center=cell_rect.center))

            y += cell_h

        self.rule_cell_rects = new_rects
        y += 8
        self.screen.blit(self.font_small.render("Check terminal for details...", True, (100, 100, 100)), (start_x, y))

        pygame.draw.rect(self.screen, self.bomb_color, self.bomb_rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 100, 100), self.bomb_rect, 2, border_radius=8)
        self.screen.blit(self.bomb_text, self.bomb_text.get_rect(center=self.bomb_rect.center))

    def _draw_sim(self, snap: SimSnapshot):
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

        for px, py, pr, pc in snap.particles:
            pygame.draw.circle(self.screen, pc, (int(px), int(py)), pr)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while self.game_running:
            snap = self.sim.get_snapshot()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.bomb_rect.collidepoint(event.pos):
                        self._commit_edit(snap)
                        self.cmd_queue.put({'type': 'detonate'})
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
                                self.edit_buffer = f"{self._current_rule_val(hit, snap):.0f}"
                                self.last_click_cell = hit
                                self.last_click_time = now
                        else:
                            self._commit_edit(snap)
                            self.last_click_cell = None

                elif event.type == pygame.KEYDOWN and self.editing_cell is not None:
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
        self.sim.stop()
        pygame.quit()

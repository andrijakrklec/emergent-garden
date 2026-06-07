"""@file gui.py
@brief Main-thread pygame front-end (the GUI): renders SimSnapshots, forwards input.

The display half of the simulation, split out of game.py. Defines the Game class:
it owns the pygame window/widgets, reads immutable SimSnapshot objects published
by the SimulationThread, renders the simulation area + dashboard, and posts user
actions (rule edits, toggles, detonate) back through the command queue.

Holds no simulation state of its own. Imports the engine (SimulationThread,
SimSnapshot) from game.py — the dependency is one-directional (game.py does not
import this module).
"""
import queue
import time
from typing import Dict, Tuple, Optional

import pygame

from src.constants import (
    SCREEN_DIM, SIM_WIDTH, SIM_HEIGHT, GUI_WIDTH, GUI_BACKGROUND_COLOR,
    BACK_BLACK, FRAME_RATE,
)
from src.game import SimSnapshot, SimulationThread, DEFAULT_CONFIG_PATH


class Game:
    """@brief Main-thread pygame front-end: renders snapshots and forwards input.

    Holds no simulation state of its own — it reads immutable SimSnapshot
    objects from the SimulationThread and posts user actions (rule edits,
    toggles, detonate) back through the command queue.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH,
                 log_dir: str = "logs", run_name: Optional[str] = None, preset: Optional[str] = None):
        """@brief Initialise pygame, build the window/widgets, and start the simulation thread.

        @param config_path Path to the configuration file passed through to the simulation.
        @param log_dir Base directory for the run's log folder.
        @param run_name Explicit run-folder name (overrides the timestamped default); for batch runs.
        @param preset Overrides the emergent_preset from config.json (for batch runs over presets).
        """
        self._log_dir = log_dir
        self._run_name = run_name
        self._preset = preset
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
        self.sim = SimulationThread(self.cmd_queue, config_path=config_path,
                                    log_dir=self._log_dir, run_name=self._run_name, preset=self._preset)
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

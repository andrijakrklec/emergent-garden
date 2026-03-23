import pygame
import numpy as np
import random
import math
from sklearn.cluster import KMeans

from src.constants import (
    SCREEN_DIM, SIM_DIM, SIM_WIDTH, SIM_HEIGHT, GUI_WIDTH, GUI_BACKGROUND_COLOR,
    PARTICLE_DEFAULT_SPAWN_NUM, PARTICLE_COLOR_RED, PARTICLE_DEFAULT_SPAWN_FRAME,
    BACK_BLACK, PARTICLE_COLOR_YELLOW, PARTICLE_COLOR_GREEN,
    PARTICLE_COLOR_BLUE, FRAME_RATE, PARTICLE_COLOR_WHITE, WALL_BOUNDARY
)
from src.particle import Particle, instantiateGroup, local_train, run_cfl_round, apply_physics_rules, update_peer_alignment

MIN_CLUSTERS = 2
MAX_CLUSTERS = 5


class Game:
    def __init__(self):
        pygame.init()
        self.game_running = True
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_header = pygame.font.SysFont("Arial", 22, bold=True)
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Consolas", 12)

        self.screen = pygame.display.set_mode(size=SCREEN_DIM)
        pygame.display.set_caption("CFL Simulation")

        self.dt = 0.1
        self.cfl_round_counter = 0

        # --- SETUP CLUSTERS & TARGETS ---
        self.num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
        self.cluster_anchors = []
        self.cluster_targets = []

        # Ensure targets are within SIMULATION bounds, not the whole screen
        D = WALL_BOUNDARY * 4
        for _ in range(self.num_clusters):
            x = random.randint(D, SIM_WIDTH - D)
            y = random.randint(D, SIM_HEIGHT - D)
            self.cluster_anchors.append((x, y))
            self.cluster_targets.append((x, y))

        # Colors
        CLUSTER_PALETTE = [
            (220, 80, 80),  # red
            (80, 160, 220),  # blue
            (80, 200, 120),  # green
            (220, 180, 60),  # amber
            (180, 80, 220),  # purple
        ]
        self.cluster_colors = {i: CLUSTER_PALETTE[i] for i in range(self.num_clusters)}
        self.cluster_colors[-1] = (80, 80, 80)  # unassigned

        # --- SPAWN PARTICLES ---
        self.all_particles = []
        for _ in range(5):
            t_idx = random.randint(0, self.num_clusters - 1)
            self.all_particles.extend(instantiateGroup(
                num=PARTICLE_DEFAULT_SPAWN_NUM,
                c=PARTICLE_COLOR_WHITE,
                frame=PARTICLE_DEFAULT_SPAWN_FRAME,
                target_idx=t_idx
            ))

        # --- CFL SETUP ---
        self.kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0)
        self.cluster_update_timer = 0
        self.cluster_update_interval = 180*2

        # --- SETUP OBSTACLES ---
        self.obstacles = []
        num_obstacles = 4
        for _ in range(num_obstacles):
            # (x, y, radius)
            ox = random.randint(150, SIM_WIDTH - 150)
            oy = random.randint(150, SIM_HEIGHT - 150)
            orad = random.randint(30, 70)
            self.obstacles.append((ox, oy, orad))

        # DEFINE BOMB BUTTON (Bottom of Sidebar)
        btn_x = SIM_WIDTH + 20
        btn_y = SIM_HEIGHT - 80
        btn_w = GUI_WIDTH - 40
        btn_h = 50
        self.bomb_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self.bomb_color = (200, 50, 50)  # Red
        self.bomb_text = self.font_header.render("DETONATE", True, (255, 255, 255))

        # Initial Round
        print(f"\n{'=' * 60}")
        print(f"INIT: Starting Simulation with {len(self.all_particles)} particles.")
        print(f"{'=' * 60}\n")
        run_cfl_round(self.all_particles, self.kmeans)

    def draw_gui(self):
        """Draws the GUI in the sidebar area (Right side)"""

        # 1. Background for Sidebar
        sidebar_rect = pygame.Rect(SIM_WIDTH, 0, GUI_WIDTH, SIM_HEIGHT)
        pygame.draw.rect(self.screen, GUI_BACKGROUND_COLOR, sidebar_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (SIM_WIDTH, 0), (SIM_WIDTH, SIM_HEIGHT), 2)

        # 2. Stats
        stats = {i: 0 for i in range(self.num_clusters)}
        for p in self.all_particles:
            if p.cluster_id in stats:
                stats[p.cluster_id] += 1

        start_x = SIM_WIDTH + 20
        y = 20
        line_h = 30

        # Title
        title = self.font_header.render("CFL Dashboard", True, (255, 255, 255))
        self.screen.blit(title, (start_x, y))
        y += 40

        # General Info
        self.screen.blit(self.font.render(f"Round: {self.cfl_round_counter}", True, (200, 200, 200)), (start_x, y))
        y += line_h
        self.screen.blit(self.font.render(f"Particles: {len(self.all_particles)}", True, (200, 200, 200)), (start_x, y))
        y += line_h * 1.5

        # Clusters
        self.screen.blit(self.font.render("Active Clusters:", True, (255, 255, 255)), (start_x, y))
        y += line_h

        for i in range(self.num_clusters):
            color = self.cluster_colors.get(i, (255, 255, 255))

            # Color box
            pygame.draw.rect(self.screen, color, (start_x, y + 5, 15, 15))

            # Text
            txt = self.font.render(f"Cluster {i}: {stats[i]} agents", True, (180, 180, 180))
            self.screen.blit(txt, (start_x + 25, y))
            y += line_h

        y += 20
        pygame.draw.line(self.screen, (80, 80, 80), (start_x, y), (SIM_WIDTH + GUI_WIDTH - 20, y), 1)
        y += 10
        self.screen.blit(self.font_small.render("Check terminal for details...", True, (150, 150, 150)), (start_x, y))

        # DRAW BOMB BUTTON
        pygame.draw.rect(self.screen, self.bomb_color, self.bomb_rect, border_radius=8)
        pygame.draw.rect(self.screen, (255, 100, 100), self.bomb_rect, 2, border_radius=8)  # Border

        # Center the text
        text_rect = self.bomb_text.get_rect(center=self.bomb_rect.center)
        self.screen.blit(self.bomb_text, text_rect)

    def run(self):
        while self.game_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.game_running = False

                # CHECK MOUSE CLICK
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.bomb_rect.collidepoint(event.pos):
                        print("\n!!! KA-BOOM !!! - Swarm disrupted.")
                        self.trigger_explosion()

            # --- LOCAL TRAINING ---
            for p in self.all_particles:
                real_target_pos = self.cluster_targets[p.target_idx]
                local_train(p, real_target_pos, self.obstacles, learning_rate=0.05)

            # --- GLOBAL CFL ROUND & LOGGING ---
            self.cluster_update_timer += 1
            if self.cluster_update_timer >= self.cluster_update_interval:
                self.cluster_update_timer = 0
                self.cfl_round_counter += 1

                # 1. Run the Clustering
                transfers = run_cfl_round(self.all_particles, self.kmeans)

                # 2. LOGGING TO TERMINAL
                inertia = self.kmeans.inertia_
                iterations = self.kmeans.n_iter_

                # Count sizes for log
                counts = {}
                for p in self.all_particles:
                    counts[p.cluster_id] = counts.get(p.cluster_id, 0) + 1

                print(f"\n[ROUND {self.cfl_round_counter}] CFL Complete")
                print(f"   > Inertia:       {inertia:.2f}")
                print(f"   > Cluster Sizes: {dict(sorted(counts.items()))}")

                print(f"   > Migrations (Transfers):")
                if not transfers:
                    print("       (Stable - No particles switched clusters)")
                else:
                    # Sort by number of particles moving (highest first)
                    sorted_transfers = sorted(transfers.items(), key=lambda item: item[1], reverse=True)

                    for (old_id, new_id), count in sorted_transfers:
                        # Handle the very first round where old_id is -1
                        src_name = "Unassigned" if old_id == -1 else f"Cluster {old_id}"
                        print(f"       - {count:3d} agents moved: {src_name} -> Cluster {new_id}")

                print("-" * 50)

            update_peer_alignment(self.all_particles)  # new — computes model[4]

            # --- PHYSICS ---
            # using attract == repel
            apply_physics_rules(self.all_particles, self.obstacles, -150.0, 150.0, self.dt)

            # --- DRAWING ---
            self.screen.fill(BACK_BLACK)  # Fills whole screen black

            # Draw Boundaries of Simulation
            pygame.draw.rect(self.screen, (20, 20, 20), (0, 0, SIM_WIDTH, SIM_HEIGHT))

            # Draw Obstacles
            for ox, oy, orad in self.obstacles:
                pygame.draw.circle(self.screen, (50, 50, 50), (ox, oy), orad)
                pygame.draw.circle(self.screen, (150, 50, 50), (ox, oy), orad, 2)  # Red border

            # Draw Targets (Crosshairs so you can see where they are going)
            for idx, (tx, ty) in enumerate(self.cluster_targets):
                color = self.cluster_colors.get(idx, (255, 255, 255))
                pygame.draw.circle(self.screen, color, (tx, ty), 10, 1)
                pygame.draw.line(self.screen, color, (tx - 15, ty), (tx + 15, ty), 1)
                pygame.draw.line(self.screen, color, (tx, ty - 15), (tx, ty + 15), 1)

            # Draw Particles
            for particle in self.all_particles:
                # Tint based on cluster
                viz_color = self.cluster_colors.get(particle.cluster_id, (0xff, 0xff, 0xff))
                particle.c = viz_color
                particle.draw(self.screen)

            # Draw Sidebar
            self.draw_gui()

            pygame.display.flip()
            self.dt = self.clock.tick(FRAME_RATE) / 1000.0

    def trigger_explosion(self):
        """
        Creates a massive disruption:
        1. Physical: Scatters particles with high velocity.
        2. Cognitive: Resets their learned models (Memory Loss).
        """
        center_x, center_y = SIM_WIDTH // 2, SIM_HEIGHT // 2

        for p in self.all_particles:
            # 1. PHYSICAL CHAOS (Random High Velocity)
            p.vx = random.uniform(-80, 80)
            p.vy = random.uniform(-80, 80)

            # Teleport them slightly to break clumps instantly
            p.x += random.uniform(-400, 400)
            p.y += random.uniform(-200, 200)

            p.model = np.array([
                random.uniform(-1, 1), random.uniform(-1, 1),
                0.1,  # very low confidence after explosion
                0.8,  # high obstacle pressure (chaos)
                0.0, 0.0, 1.0, 0.0
            ])
            p.model[0:2] /= np.linalg.norm(p.model[0:2])

    def quit(self):
        pygame.quit()
import pygame
import random

from src.constants import (
    SCREEN_DIM, SIM_DIM, SIM_WIDTH, SIM_HEIGHT, GUI_WIDTH, GUI_BACKGROUND_COLOR,
    PARTICLE_DEFAULT_SPAWN_NUM, PARTICLE_DEFAULT_SPAWN_FRAME,
    BACK_BLACK, FRAME_RATE, WALL_BOUNDARY
)
from src.particle import instantiateGroup, apply_physics_rules

MIN_GROUPS = 5
MAX_GROUPS = 5


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
        pygame.display.set_caption("Particle Simulation")

        self.dt = 0.1

        # --- SETUP GROUPS ---
        self.num_groups = random.randint(MIN_GROUPS, MAX_GROUPS)

        # Colors
        self.group_colors = {}
        for i in range(self.num_groups):
            self.group_colors[i] = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

        # --- SPAWN PARTICLES ---
        self.all_particles = []

        # Spawn particles for each group
        for i in range(self.num_groups):
            # Assign color based on group ID
            color = self.group_colors[i]

            self.all_particles.extend(instantiateGroup(
                num=PARTICLE_DEFAULT_SPAWN_NUM * 2,  # Increased spawn count slightly since no training needed
                c=color,
                frame=PARTICLE_DEFAULT_SPAWN_FRAME,
                cluster_id=i
            ))

        # DEFINE BOMB BUTTON (Bottom of Sidebar)
        btn_x = SIM_WIDTH + 20
        btn_y = SIM_HEIGHT - 80
        btn_w = GUI_WIDTH - 40
        btn_h = 50
        self.bomb_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        self.bomb_color = (200, 50, 50)  # Red
        self.bomb_text = self.font_header.render("DETONATE", True, (255, 255, 255))

        print(f"\n{'=' * 60}")
        print(f"INIT: Starting Simulation with {len(self.all_particles)} particles.")
        print(f"{'=' * 60}\n")

    def draw_gui(self):
        """Draws the GUI in the sidebar area (Right side)"""

        # 1. Background for Sidebar
        sidebar_rect = pygame.Rect(SIM_WIDTH, 0, GUI_WIDTH, SIM_HEIGHT)
        pygame.draw.rect(self.screen, GUI_BACKGROUND_COLOR, sidebar_rect)
        pygame.draw.line(self.screen, (100, 100, 100), (SIM_WIDTH, 0), (SIM_WIDTH, SIM_HEIGHT), 2)

        # 2. Stats
        stats = {i: 0 for i in range(self.num_groups)}
        for p in self.all_particles:
            if p.cluster_id in stats:
                stats[p.cluster_id] += 1

        start_x = SIM_WIDTH + 20
        y = 20
        line_h = 30

        # Title
        title = self.font_header.render("Dashboard", True, (255, 255, 255))
        self.screen.blit(title, (start_x, y))
        y += 40

        # General Info
        self.screen.blit(self.font.render(f"Particles: {len(self.all_particles)}", True, (200, 200, 200)), (start_x, y))
        y += line_h * 1.5

        # Groups
        self.screen.blit(self.font.render("Particle Groups:", True, (255, 255, 255)), (start_x, y))
        y += line_h

        for i in range(self.num_groups):
            color = self.group_colors.get(i, (255, 255, 255))

            # Color box
            pygame.draw.rect(self.screen, color, (start_x, y + 5, 15, 15))

            # Text
            txt = self.font.render(f"Group {i}: {stats[i]} agents", True, (180, 180, 180))
            self.screen.blit(txt, (start_x + 25, y))
            y += line_h

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

            # --- PHYSICS ---
            # Use weaker attraction (-50) and strong motive logic
            # Params: Particles, AttractionStrength, RepulsionStrength, dt
            apply_physics_rules(self.all_particles, -250.0, 100.0, self.dt)

            # --- DRAWING ---
            self.screen.fill(BACK_BLACK)  # Fills whole screen black

            # Draw Boundaries of Simulation
            pygame.draw.rect(self.screen, (20, 20, 20), (0, 0, SIM_WIDTH, SIM_HEIGHT))

            # Draw Particles
            for particle in self.all_particles:
                particle.draw(self.screen)

            # Draw Sidebar
            self.draw_gui()

            pygame.display.flip()
            self.dt = self.clock.tick(FRAME_RATE) / 1000.0

    def trigger_explosion(self):
        """
        Creates a massive physical disruption.
        """
        for p in self.all_particles:
            # 1. PHYSICAL CHAOS (Random High Velocity)
            p.vx = random.uniform(-80, 80)
            p.vy = random.uniform(-80, 80)

            # Teleport them slightly to break clumps instantly
            p.x += random.uniform(-200, 200)
            p.y += random.uniform(-200, 200)

            # Keep inside bounds
            p.x = max(WALL_BOUNDARY, min(SIM_WIDTH - WALL_BOUNDARY, p.x))
            p.y = max(WALL_BOUNDARY, min(SIM_HEIGHT - WALL_BOUNDARY, p.y))

    def quit(self):
        pygame.quit()
import pygame
import numpy as np
from sklearn.cluster import KMeans
import random

from src.constants import (
    PARTICLE_DEFAULT_SPAWN_NUM, PARTICLE_COLOR_RED, PARTICLE_DEFAULT_SPAWN_FRAME,
    SCREEN_DIM, BACK_BLACK, PARTICLE_COLOR_YELLOW, PARTICLE_COLOR_GREEN,
    PARTICLE_COLOR_BLUE, FRAME_RATE, PARTICLE_COLOR_WHITE, WALL_BOUNDARY
)
from src.particle import Particle, instantiateGroup, local_train, run_cfl_round, apply_physics_rules

MIN_CLUSTERS = 2
MAX_CLUSTERS = 5


class Game:
    """This class represents the game instances
    """

    def __init__(self):
        """Generates a new Game object
        """
        pygame.init()
        self.game_running = True
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18, bold=True)
        self.status_font = pygame.font.SysFont("Arial", 16, italic=True)  # For status messages
        self.show_status_timer = 0

        self.screen = pygame.display.set_mode(size=SCREEN_DIM)
        pygame.display.set_caption("Clustered Federated Learning Simulacija")

        self.dt = 0.1

        # --- Postavke nasumičnih klastera ---
        self.num_clusters = random.randint(MIN_CLUSTERS, MAX_CLUSTERS)
        print(f"--- Pokrećem simulaciju s {self.num_clusters} nasumičnih klastera ---")

        # 1. Generiraj nasumične "skrivene ciljeve"
        self.cluster_targets = []
        D = WALL_BOUNDARY * 2 # Odbijanje od ruba
        for _ in range(self.num_clusters):
            x = random.randint(D, SCREEN_DIM[0] - D)
            y = random.randint(D, SCREEN_DIM[1] - D)
            self.cluster_targets.append((x, y))

        # 2. Generiraj nasumične boje za vizualizaciju klastera
        self.cluster_colors = {}
        for i in range(self.num_clusters):
            # Generiraj nasumičnu svijetlu boju
            r = random.randint(100, 255)
            g = random.randint(100, 255)
            b = random.randint(100, 255)
            self.cluster_colors[i] = (r, g, b)
        self.cluster_colors[-1] = (0x50, 0x50, 0x50) # Boja za "šum" / neklasificirane

        # --- Inicijalizacija čestica (Klijenata) ---
        self.all_particles = []

        # Svaka grupa čestica dobiva JEDAN od nasumičnih ciljeva
        self.all_particles.extend(instantiateGroup(num=PARTICLE_DEFAULT_SPAWN_NUM, c=PARTICLE_COLOR_RED, frame=PARTICLE_DEFAULT_SPAWN_FRAME, target_pos=random.choice(self.cluster_targets)))
        self.all_particles.extend(instantiateGroup(num=PARTICLE_DEFAULT_SPAWN_NUM, c=PARTICLE_COLOR_YELLOW, frame=PARTICLE_DEFAULT_SPAWN_FRAME, target_pos=random.choice(self.cluster_targets)))
        self.all_particles.extend(instantiateGroup(num=PARTICLE_DEFAULT_SPAWN_NUM, c=PARTICLE_COLOR_GREEN, frame=PARTICLE_DEFAULT_SPAWN_FRAME, target_pos=random.choice(self.cluster_targets)))
        self.all_particles.extend(instantiateGroup(num=PARTICLE_DEFAULT_SPAWN_NUM, c=PARTICLE_COLOR_BLUE, frame=PARTICLE_DEFAULT_SPAWN_FRAME, target_pos=random.choice(self.cluster_targets)))
        self.all_particles.extend(instantiateGroup(num=PARTICLE_DEFAULT_SPAWN_NUM, c=PARTICLE_COLOR_WHITE, frame=PARTICLE_DEFAULT_SPAWN_FRAME, target_pos=random.choice(self.cluster_targets)))

        # --- CFL Postavke ---
        # KMeans sada koristi nasumičan broj klastera
        self.kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=0)

        self.cluster_update_timer = 0
        self.cluster_update_interval = 180 # Pokreni CFL svakih ~3 sekunde

        print(f"--- Pokrećem INICIJALNU CFL rundu na {len(self.all_particles)} čestica ---")
        run_cfl_round(self.all_particles, self.kmeans)

    def draw_gui(self):
        """Renders the cluster information overlay with status indicators"""
        stats = {i: 0 for i in range(self.num_clusters)}
        for p in self.all_particles:
            if p.cluster_id in stats:
                stats[p.cluster_id] += 1

        # Configuration for Spacing
        start_x, start_y = 20, 20
        line_height = 30

        # Calculate panel height: Header + Total + Clusters + Status Message space
        panel_height = (start_y * 2) + (line_height * (self.num_clusters + 3))
        panel_rect = pygame.Rect(10, 10, 260, panel_height)

        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        pygame.draw.rect(self.screen, (200, 200, 200), panel_rect, 2)

        # 1. Title
        title = self.font.render("CFL Simulation Status", True, (255, 255, 255))
        self.screen.blit(title, (start_x, start_y))

        # [cite_start]2. Particle Count [cite: 1]
        total_txt = self.font.render(f"Total Particles: {len(self.all_particles)}", True, (180, 180, 180))
        self.screen.blit(total_txt, (start_x, start_y + line_height))

        # 3. Cluster Legend
        offset_y = start_y + (line_height * 2)
        for i in range(self.num_clusters):
            color = self.cluster_colors.get(i, (255, 255, 255))
            current_y = offset_y + (i * line_height)
            pygame.draw.rect(self.screen, color, (start_x, current_y + 5, 15, 15))
            cluster_txt = self.font.render(f"Cluster {i}: {stats[i]} particles", True, (255, 255, 255))
            self.screen.blit(cluster_txt, (start_x + 25, current_y))

        # 4. CFL Round Status Indicator
        if self.show_status_timer > 0:
            status_y = offset_y + (self.num_clusters * line_height) + 10
            # Create a blinking effect or solid green text
            status_color = (0, 255, 0) if (pygame.time.get_ticks() // 250) % 2 == 0 else (0, 200, 0)
            status_txt = self.status_font.render("✔ CFL ROUND FINISHED", True, status_color)
            self.screen.blit(status_txt, (start_x, status_y))
            self.show_status_timer -= 1  # Countdown frames

    def run(self):
        """Runs the game
        """
        while self.game_running:

            # Obrada događaja
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_running = False

            # 1.1. Lokalni trening
            for p in self.all_particles:
                local_train(p, learning_rate=0.05) # Spora stopa učenja

            # 1.2. Globalna CFL runda (grupiranje i agregacija)
            self.cluster_update_timer += 1
            if self.cluster_update_timer >= self.cluster_update_interval:
                self.cluster_update_timer = 0
                print(f"--- Pokrećem CFL rundu grupiranja na {len(self.all_particles)} čestica ---")
                run_cfl_round(self.all_particles, self.kmeans)
                self.show_status_timer = 90

            # Privlačenje je NEGATIVNO, Odbijanje je POZITIVNO
            G_ATTRACT = -500.0
            G_REPEL = 150.0

            apply_physics_rules(self.all_particles, G_ATTRACT, G_REPEL, self.dt)

            # --- CRTANJE ---
            self.screen.fill(BACK_BLACK)

            for particle in self.all_particles:
                # Koristi dinamički generirane boje klastera
                viz_color = self.cluster_colors.get(particle.cluster_id, (0xff, 0xff, 0xff))

                original_color = particle.c
                particle.c = viz_color
                particle.draw(self.screen)
                particle.c = original_color

            self.draw_gui()
            pygame.display.flip()
            self.dt = self.clock.tick(FRAME_RATE) / 1000.0

            # Ograniči FPS i izračunaj delta-time (dt)
            self.dt = self.clock.tick(FRAME_RATE) / 1000.0


    def quit(self):
        """Quits the game
        """
        pygame.quit()
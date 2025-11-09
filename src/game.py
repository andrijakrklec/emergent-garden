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

            pygame.display.flip()

            # Ograniči FPS i izračunaj delta-time (dt)
            self.dt = self.clock.tick(FRAME_RATE) / 1000.0


    def quit(self):
        """Quits the game
        """
        pygame.quit()
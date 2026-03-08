"""
AUTHOR: Vishal Paudel
(Modified: CFL removed, Standard Particle Physics)
"""
from typing import Tuple, List
from ctypes import c_ubyte
import pygame
import random
import numpy as np

from src.constants import (
    SIM_DIM,
    PARTICLE_DEFAULT_RADIUS, WALL_BOUNDARY,
    PARTICLE_FORCE_LOWER_RANGE, PARTICLE_FORCE_UPPER_RANGE,
    PARTICLE_POWER_OF_DISTANCE, PARTICLE_LOSE_ENERGY, PARTICLE_MAX_SPEED
)

class Particle:
    def __init__(
        self,
        x: Tuple[int, int],
        v: Tuple[float, float],
        c: Tuple[c_ubyte, c_ubyte, c_ubyte],
        cluster_id: int,
        r: float = PARTICLE_DEFAULT_RADIUS
        ):
        """Initializes a particle

        Args:
            x   (Tuple[int, int]):      The postion     vector  of the particle
            v   (Tuple[float, float]):  The velocity    vector  of the particle
            c   (Tuple[int, int, int]): The color       RGB     of the particle
            cluster_id (int):           The group ID this particle belongs to
            r   (int):                  The radius      pixel   of the particle
        """
        # The position attributes
        self.x = x[0]
        self.y = x[1]

        # The velocity attributes
        self.vx = v[0]
        self.vy = v[1]

        # The look and feel attributes
        self.c = c
        self.r = r

        # Fixed Group ID (Determines physics interactions)
        self.cluster_id = cluster_id

    def draw(self, screen: pygame.Surface):
        """Draws the particle(a circle)"""
        pygame.draw.circle(screen, self.c, (self.x, self.y), self.r)


def apply_physics_rules(particles: List[Particle], g_attract: float, g_repel: float, dt: float):
    """
    Standard Particle Physics:
    - Same cluster_id: Attract
    - Different cluster_id: Repel
    """
    # Initialize forces
    forces = [np.zeros(2) for _ in particles]

    # Calculate interactions
    for i in range(len(particles)):
        a = particles[i]
        for j in range(i + 1, len(particles)):
            b = particles[j]

            dx = a.x - b.x
            dy = a.y - b.y
            d_sq = dx ** 2 + dy ** 2

            # 1. VISUAL OPTIMIZATION: If they are too far, skip math entirely
            if d_sq > PARTICLE_FORCE_UPPER_RANGE ** 2:
                continue

            # Avoid division by zero
            if d_sq == 0:
                d_sq = 0.001

            d = d_sq ** 0.5

            # 2. EMERGENCY REPULSION
            # If particles are touching (closer than radius), push them apart HARD
            if d < PARTICLE_FORCE_LOWER_RANGE * 2:
                F_scalar = 2000.0  # Massive repulsion to unclump them

            # 3. NORMAL PHYSICS
            else:
                g = 0.0

                if a.cluster_id == b.cluster_id:
                    g = g_attract  # Attraction
                else:
                    g = g_repel  # Repulsion

                # Standard Gravity Formula
                denom = (d ** PARTICLE_POWER_OF_DISTANCE) * len(particles)
                if denom == 0: denom = 0.001
                F_scalar = g * (1 / denom)

            # Apply forces
            fx = F_scalar * dx
            fy = F_scalar * dy

            forces[i] += np.array([fx, fy])
            forces[j] -= np.array([fx, fy])

    # Apply forces to velocities and positions
    for i, p in enumerate(particles):
        fx, fy = forces[i]

        p.vx = (p.vx + fx * dt) * PARTICLE_LOSE_ENERGY
        p.vy = (p.vy + fy * dt) * PARTICLE_LOSE_ENERGY

        speed = (p.vx ** 2 + p.vy ** 2) ** 0.5

        # If going faster than max, scale it down
        if speed > PARTICLE_MAX_SPEED:
            scale = PARTICLE_MAX_SPEED / speed
            p.vx *= scale
            p.vy *= scale

        p.x += p.vx * dt
        p.y += p.vy * dt

        # --- UPDATED WALL COLLISIONS ---
        V = 0.9
        D = WALL_BOUNDARY

        # Left Wall
        if p.x < D:
            p.x = D
            p.vx *= -V

        # Right Wall
        if p.x > SIM_DIM[0] - D:
            p.x = SIM_DIM[0] - D
            p.vx *= -V

        # Top Wall
        if p.y < D:
            p.y = D
            p.vy *= -V

        # Bottom Wall
        if p.y > SIM_DIM[1] - D:
            p.y = SIM_DIM[1] - D
            p.vy *= -V

def instantiateGroup(
    num: int,
    c: tuple,
    frame: Tuple[Tuple[int, int], Tuple[int, int]],
    cluster_id: int
    ) -> List[Particle]:
    """Method to instantiate a group of particles"""
    random.seed()
    group = []

    for _ in range(num):
        x = random.randint(frame[0][0], frame[0][1])
        y = random.randint(frame[1][0], frame[1][1])

        # Pass cluster_id directly
        group.append(Particle(x=(x, y), v=(0.0, 0.0), c=c, cluster_id=cluster_id))

    return group
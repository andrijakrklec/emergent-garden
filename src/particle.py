"""
AUTHOR: Vishal Paudel
(Modificirano za CFL simulaciju)
"""
from typing import Tuple, List
from ctypes import c_ubyte
from collections import defaultdict

import pygame
import random
import math
import numpy as np
from sklearn.cluster import KMeans


from src.constants import (
    SIM_DIM,
    PARTICLE_DEFAULT_RADIUS, SCREEN_DIM, WALL_HEAT, WALL_BOUNDARY,
    PARTICLE_FORCE_LOWER_RANGE, PARTICLE_FORCE_UPPER_RANGE,
    PARTICLE_POWER_OF_DISTANCE, PARTICLE_DEFAULT_UPDATE_TIME, PARTICLE_LOSE_ENERGY, PARTICLE_MAX_SPEED
)

class Particle:
    def __init__(
        self,
        x: Tuple[int, int],
        v: Tuple[float, float],
        c: Tuple[c_ubyte, c_ubyte, c_ubyte],
        target_idx: int,
        r: float = PARTICLE_DEFAULT_RADIUS
        ):
        """Initializes a particle

        Args:
            x   (Tuple[int, int]):      The postion     vector  of the particle
            v   (Tuple[float, float]):  The velocity    vector  of the particle
            c   (Tuple[int, int, int]): The color       RGB     of the particle
            r   (int):                  The radius      pixel   of the particle
            target_pos (Tuple[int, int]): The "hidden" target for this particle (its "local data")
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

        self.target_idx = target_idx # "Lokalni podaci"

        self.model = np.random.rand(2) * 2 - 1
        self.model = self.model / np.linalg.norm(self.model) # Normaliziramo

        self.cluster_id = -1 # Pripadnost klasteru (-1 = nije dodijeljen)


    def update(self, dt: float):
        """Updates the attributes of the particle

        Args:
            dt (float): The delta time, time since the last frame update
        """
        # (Ova funkcija se trenutno ne koristi, fizika se rješava centralno)
        pass


    def draw(self, screen: pygame.Surface):
        """Draws the particle(a circle)

        Args:
            screen (pygame.Surface):    The     screen  to draw onto
        """
        pygame.draw.circle(screen, self.c, (self.x, self.y), self.r)


def local_train(particle: Particle, current_target_pos: Tuple[int, int], obstacles: List[Tuple[int, int, int]],
                learning_rate: float = 0.1):
    """
    Simulates local training with obstacle avoidance.
    """
    # 1. Calculate ideal vector to the target (Attraction)
    tx = current_target_pos[0] - particle.x
    ty = current_target_pos[1] - particle.y
    dist_t = math.hypot(tx, ty)

    if dist_t > 0:
        tx, ty = tx / dist_t, ty / dist_t

    # 2. Calculate vector away from obstacles (Repulsion)
    ox_total, oy_total = 0.0, 0.0
    for ox, oy, orad in obstacles:
        dx = particle.x - ox
        dy = particle.y - oy
        dist_o = math.hypot(dx, dy)

        # Sensor range (e.g., obstacle radius + 60 pixels)
        sense_radius = orad + 60
        if 0 < dist_o < sense_radius:
            # The closer they are, the harder they push away
            push_strength = (sense_radius - dist_o) / sense_radius
            ox_total += (dx / dist_o) * push_strength
            oy_total += (dy / dist_o) * push_strength

    # 3. Combine vectors to find the new 'Ideal' direction
    # We weight the obstacle avoidance heavily (x2.5) so they prioritize survival
    ideal_x = tx + (ox_total * 2.5)
    ideal_y = ty + (oy_total * 2.5)

    norm_ideal = math.hypot(ideal_x, ideal_y)
    if norm_ideal > 0:
        ideal_x, ideal_y = ideal_x / norm_ideal, ideal_y / norm_ideal

    # 4. Gradient descent: Move current learned model towards the ideal direction
    particle.model[0] += learning_rate * (ideal_x - particle.model[0])
    particle.model[1] += learning_rate * (ideal_y - particle.model[1])

    # Re-normalize model to keep it a unit direction vector
    norm_model = np.linalg.norm(particle.model)
    if norm_model > 0:
        particle.model = particle.model / norm_model

def apply_physics_rules(particles: List[Particle], obstacles: List[Tuple[int, int, int]], g_attract: float, g_repel: float, dt: float):
    """
    Revised Physics: Prevents stacking by adding emergency repulsion.
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

                # Only interact if both have a valid cluster
                if a.cluster_id != -1 and b.cluster_id != -1:
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

        # --- HARD PHYSICAL COLLISIONS WITH OBSTACLES ---
        for ox, oy, orad in obstacles:
            dx, dy = p.x - ox, p.y - oy
            dist = math.hypot(dx, dy)

            # If they physically hit the obstacle, push them out
            if dist < orad + p.r:
                overlap = (orad + p.r) - dist
                if dist > 0:
                    # Positional correction ONLY. Do not touch vx/vy!
                    p.x += (dx / dist) * overlap
                    p.y += (dy / dist) * overlap

        # --- UPDATED WALL COLLISIONS ---
        # Use SIM_DIM[0] and SIM_DIM[1] instead of SCREEN_DIM

        V = 0.9
        D = WALL_BOUNDARY

        # Left Wall
        if p.x < D:
            p.x = D
            p.vx *= -V

        # Right Wall (Now uses simulation width, not window width)
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

def run_cfl_round(particles: List[Particle], kmeans_model: KMeans):
    """
    Runs CFL and returns a dictionary of transfers: {(old_id, new_id): count}
    """
    if not particles:
        return {}

    # 1. Capture Old State (Before training changes anything)
    old_ids = [p.cluster_id for p in particles]

    # 2. Standard KMeans Steps
    all_models = np.array([p.model for p in particles])
    new_labels = kmeans_model.fit_predict(all_models)
    cluster_centers = kmeans_model.cluster_centers_

    # 3. Update Particles and Track Transfers
    transfers = defaultdict(int)

    for i, p in enumerate(particles):
        old_id = old_ids[i]
        new_id = new_labels[i]

        # Record Transfer if the cluster changed
        if old_id != new_id:
            transfers[(old_id, new_id)] += 1

        # Apply Updates
        p.cluster_id = new_id

        # Federated Aggregation (Average the model)
        new_model = cluster_centers[new_id]
        norm = np.linalg.norm(new_model)
        if norm > 0:
            p.model = new_model / norm
        else:
            p.model = new_model

    return transfers

def instantiateGroup(
    num: int,
    c: tuple,
    frame: Tuple[Tuple[int, int], Tuple[int, int]],
    target_idx: int
    ) -> List[Particle]:
    """Method to instantiate a group of particles

    Args:
        num     (int):                                      The number          of particles in the group
        c       (tuple):                                    The color           of the group
        frame   (Tuple[Tuple[int, int], Tuple[int, int]]):  The cordinate frame for the group to spawn in
        target_pos (Tuple[int, int]):                       The hidden target   for this group

    Returns:
        group   List[Particle]:                             The list    of particles        in the group
    """
    random.seed()
    group = []

    for _ in range(num):
        x = random.randint(frame[0][0], frame[0][1])
        y = random.randint(frame[1][0], frame[1][1])

        # Pass target_idx to the Particle constructor
        group.append(Particle(x=(x, y), v=(0.0, 0.0), c=c, target_idx=target_idx))

    return group
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
    def __init__(self, x, v, c, target_idx, r=PARTICLE_DEFAULT_RADIUS):
        self.x = x[0]
        self.y = x[1]
        self.vx = v[0]
        self.vy = v[1]
        self.c = c
        self.r = r
        self.target_idx = target_idx
        self.cluster_id = -1

        # 8-dimensional model:
        # [0] dir_x, [1] dir_y — learned heading (unit vector, as before)
        # [2] confidence       — 0..1, how consistently this particle converges
        # [3] obstacle_pressure — 0..1, decaying memory of recent obstacle hits
        # [4] peer_alignment   — 0..1, cosine sim to same-cluster neighbors
        # [5] rounds_stable    — normalised count of rounds without cluster change
        # [6] local_loss       — normalised distance to personal target
        # [7] drift_velocity   — normalised recent average speed
        direction = np.random.randn(2)
        direction /= np.linalg.norm(direction)
        self.model = np.array([
            direction[0],
            direction[1],
            0.5,  # start with middling confidence
            0.0,  # no obstacle pressure yet
            0.0,  # no peer alignment computed yet
            0.0,  # not stable yet
            1.0,  # assume far from target initially
            0.0,  # not moving yet
        ], dtype=np.float64)

        # Internal bookkeeping
        self._prev_cluster_id = -1
        self._stable_rounds = 0
        self._speed_ema = 0.0  # exponential moving average of speed


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


def local_train(particle, current_target_pos, obstacles, learning_rate=0.1):
    tx = current_target_pos[0] - particle.x
    ty = current_target_pos[1] - particle.y
    dist_t = math.hypot(tx, ty)

    if dist_t > 0:
        tx_n, ty_n = tx / dist_t, ty / dist_t
    else:
        tx_n, ty_n = 0.0, 0.0

    # --- Obstacle repulsion ---
    ox_total, oy_total = 0.0, 0.0
    raw_pressure = 0.0
    for ox, oy, orad in obstacles:
        dx = particle.x - ox
        dy = particle.y - oy
        dist_o = math.hypot(dx, dy)
        sense_radius = orad + 60
        if 0 < dist_o < sense_radius:
            push = (sense_radius - dist_o) / sense_radius
            ox_total += (dx / dist_o) * push
            oy_total += (dy / dist_o) * push
            raw_pressure += push

    ideal_x = tx_n + ox_total * 2.5
    ideal_y = ty_n + oy_total * 2.5
    norm = math.hypot(ideal_x, ideal_y)
    if norm > 0:
        ideal_x, ideal_y = ideal_x / norm, ideal_y / norm

    # --- Update [0:2]: direction (gradient descent as before) ---
    particle.model[0] += learning_rate * (ideal_x - particle.model[0])
    particle.model[1] += learning_rate * (ideal_y - particle.model[1])
    norm_m = np.linalg.norm(particle.model[0:2])
    if norm_m > 0:
        particle.model[0:2] /= norm_m

    # --- Update [2]: confidence ---
    # Alignment between current heading and ideal direction is our proxy for "is training working?"
    alignment = particle.model[0] * ideal_x + particle.model[1] * ideal_y  # -1..1
    confidence_signal = (alignment + 1) / 2  # remap to 0..1
    # Decay confidence if under obstacle pressure
    confidence_signal *= max(0.0, 1.0 - raw_pressure)
    particle.model[2] += 0.05 * (confidence_signal - particle.model[2])
    particle.model[2] = np.clip(particle.model[2], 0.01, 1.0)

    # --- Update [3]: obstacle_pressure (decaying memory) ---
    particle.model[3] = 0.9 * particle.model[3] + 0.1 * min(raw_pressure, 1.0)

    # --- Update [6]: local_loss (normalised distance to target) ---
    max_dist = math.hypot(SIM_DIM[0], SIM_DIM[1])
    particle.model[6] = min(dist_t / max_dist, 1.0)

    # --- Update [7]: drift_velocity (EMA of speed) ---
    speed = math.hypot(particle.vx, particle.vy)
    particle._speed_ema = 0.95 * particle._speed_ema + 0.05 * speed
    particle.model[7] = min(particle._speed_ema / PARTICLE_MAX_SPEED, 1.0)

def update_peer_alignment(particles, neighbor_radius=90.0):
    """Updates model[4] for every particle based on directional consensus with cluster neighbors."""
    for p in particles:
        if p.cluster_id == -1:
            p.model[4] = 0.0
            continue

        neighbors = [
            q for q in particles
            if q is not p
            and q.cluster_id == p.cluster_id
            and math.hypot(q.x - p.x, q.y - p.y) < neighbor_radius
        ]

        if not neighbors:
            p.model[4] = 0.0
            continue

        # Average direction of neighbors
        avg_dx = sum(q.model[0] for q in neighbors) / len(neighbors)
        avg_dy = sum(q.model[1] for q in neighbors) / len(neighbors)
        norm = math.hypot(avg_dx, avg_dy)
        if norm > 0:
            avg_dx, avg_dy = avg_dx / norm, avg_dy / norm

        # Cosine similarity between this particle's heading and the group average
        cos_sim = p.model[0] * avg_dx + p.model[1] * avg_dy  # -1..1
        p.model[4] = (cos_sim + 1) / 2  # remap to 0..1

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

def run_cfl_round(particles, kmeans_model):
    if not particles:
        return {}

    old_ids = [p.cluster_id for p in particles]

    # KMeans still clusters on the full 8D model —
    # this means particles with similar confidence, pressure, and alignment
    # will cluster together, not just heading.
    all_models = np.array([p.model for p in particles])
    new_labels = kmeans_model.fit_predict(all_models)

    transfers = defaultdict(int)

    for i, p in enumerate(particles):
        old_id = old_ids[i]
        new_id = int(new_labels[i])
        if old_id != new_id:
            transfers[(old_id, new_id)] += 1
        p.cluster_id = new_id

    # Confidence-weighted federated aggregation per cluster
    for cluster_id in range(kmeans_model.n_clusters):
        members = [p for p in particles if p.cluster_id == cluster_id]
        if not members:
            continue

        weights = np.array([p.model[2] for p in members])  # confidence
        weights = weights / (weights.sum() + 1e-8)

        aggregated = sum(w * p.model for w, p in zip(weights, members))

        # Re-normalise the direction component after averaging
        dir_norm = np.linalg.norm(aggregated[0:2])
        if dir_norm > 0:
            aggregated[0:2] /= dir_norm

        for p in members:
            # Blend: 40% own knowledge, 60% federated consensus
            p.model = 0.4 * p.model + 0.6 * aggregated
            # Clamp non-direction dimensions to valid ranges
            p.model[2] = np.clip(p.model[2], 0.01, 1.0)
            p.model[3] = np.clip(p.model[3], 0.0, 1.0)
            p.model[4] = np.clip(p.model[4], 0.0, 1.0)

    # Update rounds_stable [5]
    for p in particles:
        if p.cluster_id == p._prev_cluster_id:
            p._stable_rounds = min(p._stable_rounds + 1, 50)
        else:
            p._stable_rounds = 0
        p._prev_cluster_id = p.cluster_id
        p.model[5] = p._stable_rounds / 50.0

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
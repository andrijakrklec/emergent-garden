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
    PARTICLE_POWER_OF_DISTANCE, PARTICLE_DEFAULT_UPDATE_TIME, PARTICLE_LOSE_ENERGY, PARTICLE_MAX_SPEED,
    SIM_HEIGHT, SIM_WIDTH
)

MIN_CLUSTERS = 2
MAX_CLUSTERS = 6
SPLIT_LOSS_THRESHOLD = 0.45    # avg local_loss above this triggers a split
MERGE_SIMILARITY_THRESHOLD = 0.97  # cosine sim above this triggers a merge
MIN_CLUSTER_SIZE = 3           # clusters smaller than this get absorbed

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

def compute_cluster_stats(particles, n_clusters):
    """Returns per-cluster: avg_loss, avg_confidence, size, mean_model."""
    stats = {}
    for cid in range(n_clusters):
        members = [p for p in particles if p.cluster_id == cid]
        if not members:
            stats[cid] = {'size': 0, 'avg_loss': 0.0,
                          'avg_confidence': 0.0, 'mean_model': None}
            continue
        avg_loss = sum(p.model[6] for p in members) / len(members)
        avg_conf = sum(p.model[2] for p in members) / len(members)
        mean_model = sum(p.model for p in members) / len(members)
        stats[cid] = {
            'size': len(members),
            'avg_loss': avg_loss,
            'avg_confidence': avg_conf,
            'mean_model': mean_model,
        }
    return stats

def run_cfl_round(particles, kmeans_model, cluster_targets, cluster_colors, cooldown_counter):
    """
    Returns: (transfers, new_kmeans, new_targets, new_colors, new_n, event, cooldown)
    event is one of: None, 'split', 'merge'
    """
    if not particles:
        return {}, kmeans_model, cluster_targets, cluster_colors, kmeans_model.n_clusters, None, cooldown_counter

    n = kmeans_model.n_clusters
    old_ids = [p.cluster_id for p in particles]

    all_models = np.array([p.model for p in particles])
    new_labels = kmeans_model.fit_predict(all_models)

    transfers = defaultdict(int)
    for i, p in enumerate(particles):
        old_id = old_ids[i]
        new_id = int(new_labels[i])
        if old_id != new_id:
            transfers[(old_id, new_id)] += 1
        p.cluster_id = new_id

    # Weighted aggregation (unchanged)
    for cid in range(n):
        members = [p for p in particles if p.cluster_id == cid]
        if not members:
            continue
        weights = np.array([p.model[2] for p in members])
        weights /= weights.sum() + 1e-8
        aggregated = sum(w * p.model for w, p in zip(weights, members))
        norm = np.linalg.norm(aggregated[0:2])
        if norm > 0:
            aggregated[0:2] /= norm
        for p in members:
            p.model = 0.4 * p.model + 0.6 * aggregated
            p.model[2] = np.clip(p.model[2], 0.01, 1.0)
            p.model[3:5] = np.clip(p.model[3:5], 0.0, 1.0)

    # Stable round counter
    for p in particles:
        if p.cluster_id == p._prev_cluster_id:
            p._stable_rounds = min(p._stable_rounds + 1, 50)
        else:
            p._stable_rounds = 0
        p._prev_cluster_id = p.cluster_id
        p.model[5] = p._stable_rounds / 50.0

    # --- RESTRUCTURING (only if cooldown has expired) ---
    if cooldown_counter > 0:
        return transfers, kmeans_model, cluster_targets, cluster_colors, n, None, cooldown_counter - 1

    stats = compute_cluster_stats(particles, n)

    # 1. MERGE — find two clusters whose mean models are nearly identical
    if n > MIN_CLUSTERS:
        best_sim, merge_pair = -1.0, None
        for a in range(n):
            for b in range(a + 1, n):
                ma = stats[a]['mean_model']
                mb = stats[b]['mean_model']
                if ma is None or mb is None:
                    continue
                # Only compare direction components for similarity
                sim = float(np.dot(ma[0:2], mb[0:2]))
                if sim > best_sim:
                    best_sim, merge_pair = sim, (a, b)

        if best_sim >= MERGE_SIMILARITY_THRESHOLD and merge_pair:
            keep, drop = merge_pair
            # Reassign all particles from drop -> keep
            for p in particles:
                if p.cluster_id == drop:
                    p.cluster_id = keep
                    p._prev_cluster_id = keep
            # Compact cluster ids: remap everything above drop down by 1
            for p in particles:
                if p.cluster_id > drop:
                    p.cluster_id -= 1
                    p._prev_cluster_id = p.cluster_id

            new_targets = [t for i, t in enumerate(cluster_targets) if i != drop]
            new_colors  = {(i if i < drop else i - 1): c
                           for i, c in cluster_colors.items() if i != drop and i != -1}
            new_colors[-1] = cluster_colors[-1]
            new_n = n - 1
            new_kmeans = KMeans(n_clusters=new_n, n_init=10, random_state=0)
            return transfers, new_kmeans, new_targets, new_colors, new_n, 'merge', 10

    # 2. SPLIT — find the cluster with the highest avg_loss
    if n < MAX_CLUSTERS:
        worst_cid = max(
            (cid for cid in range(n) if stats[cid]['size'] >= MIN_CLUSTER_SIZE * 2),
            key=lambda cid: stats[cid]['avg_loss'],
            default=None
        )
        if worst_cid is not None and stats[worst_cid]['avg_loss'] > SPLIT_LOSS_THRESHOLD:
            members = [p for p in particles if p.cluster_id == worst_cid]
            # Split by direction: above/below median dir_x
            median_dx = np.median([p.model[0] for p in members])
            new_cid = n  # the new cluster gets the next index
            split_count = 0
            for p in members:
                if p.model[0] < median_dx:
                    p.cluster_id = new_cid
                    p._prev_cluster_id = new_cid
                    split_count += 1

            # New target: perturb the original
            ox, oy = cluster_targets[worst_cid]
            new_targets = cluster_targets + [(
                int(np.clip(ox + random.randint(-150, 150), 50, SIM_WIDTH - 50)),
                int(np.clip(oy + random.randint(-150, 150), 50, SIM_HEIGHT - 50))
            )]
            # New color: pick one not already in use
            used = set(cluster_colors.values()) - {cluster_colors[-1]}
            new_color = _pick_unused_color(used)
            new_colors = dict(cluster_colors)
            new_colors[new_cid] = new_color
            new_n = n + 1
            new_kmeans = KMeans(n_clusters=new_n, n_init=10, random_state=0)
            return transfers, new_kmeans, new_targets, new_colors, new_n, 'split', 10

    return transfers, kmeans_model, cluster_targets, cluster_colors, n, None, 0

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

CLUSTER_PALETTE = [
    (220, 80,  80),
    (80,  160, 220),
    (80,  200, 120),
    (220, 180, 60),
    (180, 80,  220),
    (60,  210, 210),
]

def _pick_unused_color(used_colors):
    for c in CLUSTER_PALETTE:
        if c not in used_colors:
            return c
    return (200, 200, 200)  # fallback gray if palette exhausted
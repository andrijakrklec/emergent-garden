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

BLEND_MATURITY_ROUNDS = 15  # rounds until a new cluster reaches full global blend
BLEND_LOCAL_NEW   = 0.75    # local weight for a freshly split cluster

BEHAVIORAL_FORCE  = 3.0    # goal-directed push from model[0:2], scaled by confidence
STRAGGLER_LOSS    = 0.60   # local_loss threshold to consider a particle stranded
STRAGGLER_DRIFT   = 0.04   # drift_velocity threshold below which a particle is "stuck"

OBSTACLE_SOFT_ZONE     = 20    # px beyond physical edge where pre-contact repulsion starts
OBSTACLE_SOFT_STRENGTH = 60.0 # base strength of the soft-zone push
BLEND_LOCAL_MATURE = 0.40   # local weight once matured (your existing value)

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
        sense_radius = orad + 100  # wider sensing range so avoidance starts earlier
        if 0 < dist_o < sense_radius:
            t = (sense_radius - dist_o) / sense_radius  # 0 at edge, 1 at surface
            push = t * t * t  # cubic: steep near surface, gentle at distance
            ox_total += (dx / dist_o) * push
            oy_total += (dy / dist_o) * push
            raw_pressure += t  # linear for EMA (represents zone depth, not push force)

    # Obstacle weight scales up as particle gets closer — escape dominates target near surface
    obstacle_weight = 1.25 + raw_pressure * 0.75
    ideal_x = tx_n + ox_total * obstacle_weight
    ideal_y = ty_n + oy_total * obstacle_weight
    norm = math.hypot(ideal_x, ideal_y)
    if norm > 0:
        ideal_x, ideal_y = ideal_x / norm, ideal_y / norm

    # --- Update [0:2]: direction ---
    # Learning rate scales with sustained pressure so the direction adapts faster when stuck
    pressure_lr = min(learning_rate * (1.0 + particle.model[3] * 4.0), 0.5)
    particle.model[0] += pressure_lr * (ideal_x - particle.model[0])
    particle.model[1] += pressure_lr * (ideal_y - particle.model[1])
    norm_m = np.linalg.norm(particle.model[0:2])
    if norm_m > 0:
        particle.model[0:2] /= norm_m

    # --- Update [2]: confidence ---
    alignment = particle.model[0] * ideal_x + particle.model[1] * ideal_y  # -1..1
    confidence_signal = (alignment + 1) / 2  # remap to 0..1
    # Softer pressure penalty: killing confidence when stuck weakens the escape force
    confidence_signal *= max(0.0, 1.0 - raw_pressure * 0.4)
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
    if not particles:
        return

    n = len(particles)
    positions  = np.array([(p.x, p.y) for p in particles])        # (n, 2)
    directions = np.array([p.model[0:2] for p in particles])       # (n, 2)
    cids       = np.array([p.cluster_id for p in particles])       # (n,)

    for i, p in enumerate(particles):
        if p.cluster_id == -1:
            p.model[4] = 0.0
            continue

        same_cluster = (cids == p.cluster_id)
        same_cluster[i] = False

        if not np.any(same_cluster):
            p.model[4] = 0.0
            continue

        diffs = positions[same_cluster] - positions[i]             # (k, 2)
        in_radius = np.linalg.norm(diffs, axis=1) < neighbor_radius

        neighbor_dirs = directions[same_cluster][in_radius]        # (m, 2)
        if len(neighbor_dirs) == 0:
            p.model[4] = 0.0
            continue

        avg_dir = neighbor_dirs.mean(axis=0)
        norm = np.linalg.norm(avg_dir)
        if norm > 0:
            avg_dir /= norm

        cos_sim = float(np.dot(directions[i], avg_dir))
        p.model[4] = (cos_sim + 1) / 2

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
                F_scalar = 200.0  # Massive repulsion to unclump them

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

    # Soft obstacle zone: pre-contact repulsion so particles redirect before touching.
    # Quadratic falloff — very strong at the physical edge, tapers over OBSTACLE_SOFT_ZONE px.
    for i, p in enumerate(particles):
        for ox, oy, orad in obstacles:
            dx, dy = p.x - ox, p.y - oy
            dist = math.hypot(dx, dy)
            if dist == 0:
                continue
            edge = orad + p.r
            if dist < edge + OBSTACLE_SOFT_ZONE:
                t = max(0.0, (edge + OBSTACLE_SOFT_ZONE - dist) / OBSTACLE_SOFT_ZONE)
                push = OBSTACLE_SOFT_STRENGTH * t * t
                forces[i][0] += (dx / dist) * push
                forces[i][1] += (dy / dist) * push

    # Behavioral force: model-directed push scaled by confidence.
    # model[3] (obstacle_pressure) amplifies the force so sustained pressure
    # translates directly into a stronger escape push.
    for i, p in enumerate(particles):
        if p.cluster_id != -1 and p.model[2] > 0.15:
            pressure_boost = 1.0 + float(p.model[3]) * 1.25
            strength = BEHAVIORAL_FORCE * float(p.model[2]) * pressure_boost
            forces[i][0] += p.model[0] * strength
            forces[i][1] += p.model[1] * strength

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
                    nx, ny = dx / dist, dy / dist
                    p.x += nx * overlap
                    p.y += ny * overlap
                    # Reflect velocity along surface normal (only if moving toward obstacle)
                    dot = p.vx * nx + p.vy * ny
                    if dot < 0:
                        p.vx -= 2 * dot * nx
                        p.vy -= 2 * dot * ny
                        p.vx *= 0.8
                        p.vy *= 0.8

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

def run_cfl_round(particles, kmeans_model, cluster_targets, cluster_colors, cluster_ages, cooldown_counter):
    """
    cluster_ages: dict {cluster_id: rounds_since_created}
    """
    if not particles:
        return {}, kmeans_model, cluster_targets, cluster_colors, cluster_ages, kmeans_model.n_clusters, None, cooldown_counter

    n = kmeans_model.n_clusters
    old_ids = [p.cluster_id for p in particles]

    all_models = np.array([p.model for p in particles])

    # Warm-start: use current cluster means as initial centroids to reduce label permutation
    try:
        init_centroids = np.array([
            np.mean([p.model for p in particles if p.cluster_id == cid], axis=0)
            if any(p.cluster_id == cid for p in particles)
            else all_models[np.random.randint(len(all_models))]
            for cid in range(n)
        ])
        warm_kmeans = KMeans(n_clusters=n, init=init_centroids, n_init=1, random_state=0)
        new_labels = warm_kmeans.fit_predict(all_models)
        kmeans_model = warm_kmeans
    except Exception:
        new_labels = kmeans_model.fit_predict(all_models)

    transfers = defaultdict(int)
    for i, p in enumerate(particles):
        old_id = old_ids[i]
        new_id = int(new_labels[i])
        if old_id != new_id:
            transfers[(old_id, new_id)] += 1
        p.cluster_id = new_id

    # --- Weighted aggregation with adaptive blend ratio ---
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

        # Blend ratio depends on cluster age
        age = cluster_ages.get(cid, BLEND_MATURITY_ROUNDS)
        t = min(age / BLEND_MATURITY_ROUNDS, 1.0)  # 0.0 = newborn, 1.0 = mature
        local_weight  = BLEND_LOCAL_NEW + t * (BLEND_LOCAL_MATURE - BLEND_LOCAL_NEW)
        global_weight = 1.0 - local_weight

        for p in members:
            p.model = local_weight * p.model + global_weight * aggregated
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

    # Increment age for all existing clusters
    for cid in range(n):
        cluster_ages[cid] = cluster_ages.get(cid, 0) + 1

    # Straggler rescue: particles with high loss and near-zero drift are stuck.
    # Re-assign them to the spatially nearest cluster target every round
    # (runs even during cooldown so stuck particles aren't left orphaned).
    for p in particles:
        if p.model[6] > STRAGGLER_LOSS and p.model[7] < STRAGGLER_DRIFT:
            nearest = min(
                range(len(cluster_targets)),
                key=lambda i: math.hypot(p.x - cluster_targets[i][0],
                                         p.y - cluster_targets[i][1])
            )
            if nearest != p.cluster_id:
                p.cluster_id = nearest
                p.target_idx = nearest
                p._prev_cluster_id = -1
                p._stable_rounds = 0

    # --- Restructuring ---
    if cooldown_counter > 0:
        return transfers, kmeans_model, cluster_targets, cluster_colors, cluster_ages, n, None, cooldown_counter - 1

    stats = compute_cluster_stats(particles, n)

    # MERGE
    if n > MIN_CLUSTERS:
        best_sim, merge_pair = -1.0, None
        for a in range(n):
            for b in range(a + 1, n):
                ma = stats[a]['mean_model']
                mb = stats[b]['mean_model']
                if ma is None or mb is None:
                    continue
                # Don't merge clusters that are still young — let them diverge first
                if cluster_ages.get(a, 0) < BLEND_MATURITY_ROUNDS or \
                   cluster_ages.get(b, 0) < BLEND_MATURITY_ROUNDS:
                    continue
                sim = float(np.dot(ma[0:2], mb[0:2]))
                if sim > best_sim:
                    best_sim, merge_pair = sim, (a, b)

        if best_sim >= MERGE_SIMILARITY_THRESHOLD and merge_pair:
            keep, drop = merge_pair
            for p in particles:
                if p.cluster_id == drop:
                    p.cluster_id = keep
                    p._prev_cluster_id = -1  # force stability reset next round
                    p._stable_rounds = 0
            for p in particles:
                if p.cluster_id > drop:
                    p.cluster_id -= 1
                    p._prev_cluster_id = p.cluster_id

            new_targets = [t for i, t in enumerate(cluster_targets) if i != drop]
            new_colors  = {(i if i < drop else i - 1): c
                           for i, c in cluster_colors.items() if i != drop and i != -1}
            new_colors[-1] = cluster_colors[-1]

            # Remap ages, drop the merged cluster
            new_ages = {}
            for cid, age in cluster_ages.items():
                if cid == drop:
                    continue
                new_cid = cid if cid < drop else cid - 1
                new_ages[new_cid] = age
            # The surviving cluster inherits the older age
            new_ages[keep if keep < drop else keep - 1] = max(
                cluster_ages.get(keep, 0), cluster_ages.get(drop, 0)
            )

            new_n = n - 1
            new_kmeans = KMeans(n_clusters=new_n, n_init=10, random_state=0)

            max_target_idx = len(new_targets) - 1
            for p in particles:
                p.target_idx = min(p.target_idx, max_target_idx)

            return transfers, new_kmeans, new_targets, new_colors, new_ages, new_n, 'merge', 10

    # SPLIT
    if n < MAX_CLUSTERS:
        worst_cid = max(
            (cid for cid in range(n) if stats[cid]['size'] >= MIN_CLUSTER_SIZE * 2),
            key=lambda cid: stats[cid]['avg_loss'],
            default=None
        )
        if worst_cid is not None and stats[worst_cid]['avg_loss'] > SPLIT_LOSS_THRESHOLD:
            members = [p for p in particles if p.cluster_id == worst_cid]
            # Split on spatial x-position: more stable than model heading
            median_x = np.median([p.x for p in members])
            new_cid = n
            for p in members:
                if p.x < median_x:
                    p.cluster_id = new_cid
                    p._prev_cluster_id = -1  # force stability reset
                    p._stable_rounds = 0

            ox, oy = cluster_targets[worst_cid]
            new_targets = cluster_targets + [(
                int(np.clip(ox + random.randint(-150, 150), 50, SIM_WIDTH - 50)),
                int(np.clip(oy + random.randint(-150, 150), 50, SIM_HEIGHT - 50))
            )]
            used = set(cluster_colors.values()) - {cluster_colors[-1]}
            new_colors = dict(cluster_colors)
            new_colors[new_cid] = _pick_unused_color(used)

            # New cluster starts at age 0, parent keeps its age
            new_ages = dict(cluster_ages)
            new_ages[new_cid] = 0

            new_n = n + 1
            new_kmeans = KMeans(n_clusters=new_n, n_init=10, random_state=0)

            new_cid = n
            for p in particles:
                if p.cluster_id == new_cid:
                    p.target_idx = len(new_targets) - 1

            return transfers, new_kmeans, new_targets, new_colors, new_ages, new_n, 'split', 10

    return transfers, kmeans_model, cluster_targets, cluster_colors, cluster_ages, n, None, 0

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
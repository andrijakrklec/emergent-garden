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
    SIM_HEIGHT, SIM_WIDTH,
    PARTICLE_COLOR_RED, PARTICLE_COLOR_YELLOW, PARTICLE_COLOR_GREEN,
    PARTICLE_COLOR_BLUE, PARTICLE_COLOR_WHITE,
)

MIN_CLUSTERS = 2
MAX_CLUSTERS = 6
# Split fires when EITHER (a) avg_loss exceeds threshold (a cluster genuinely
# can't serve its members) OR (b) directional coherence among members drops
# below threshold (members disagree on heading — the natural IFCA split signal,
# since IFCA already pins avg_loss near the local minimum).
SPLIT_LOSS_THRESHOLD = 0.30
SPLIT_COHERENCE_THRESHOLD = 0.70   # mean cos-sim to cluster-mean direction
# Merge thresholds: cosine sim of mean direction (loosened from 0.97 — IFCA's
# bias-averaged means cluster more cleanly so 0.92 is a reasonable bar) plus
# spatial-target proximity (two clusters whose targets have drifted close
# together are functionally redundant under IFCA's loss).
MERGE_SIMILARITY_THRESHOLD = 0.92
MERGE_TARGET_DISTANCE = 120.0      # px — targets closer than this favour merge
MIN_CLUSTER_SIZE = 3               # clusters smaller than this get absorbed

BLEND_MATURITY_ROUNDS = 15  # rounds until a new cluster reaches full global blend
BLEND_LOCAL_NEW   = 0.75    # local weight for a freshly split cluster

# Identity slice used for KMeans assignment.
# [0:2] direction is the persistent learned identity; [2] confidence is a slow EMA.
# Features [3:8] are situational (pressure / alignment / loss / drift / stability)
# and would cause spurious migrations if used for clustering.
IDENTITY_SLICE = slice(0, 2)

# Hysteresis: a particle only migrates if the new cluster's centroid is at least
# this fraction closer (in identity-space) than its current cluster's centroid.
# 0.0 = no hysteresis (every KMeans flip migrates), 0.20 = need 20% improvement.
MIGRATION_HYSTERESIS = 0.35

BEHAVIORAL_FORCE  = 30.0    # goal-directed push from model[0:2], scaled by confidence
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

        # Per-particle directional bias — the non-IID analog from the CFL paper.
        # Each "client" systematically misperceives the ideal direction by a random
        # fixed offset.  Federation averages out these biases; solo learning cannot.
        _bias_angle    = random.uniform(0, 2 * math.pi)
        _bias_strength = random.uniform(0.25, 0.60)
        self._local_bias = np.array([
            math.cos(_bias_angle) * _bias_strength,
            math.sin(_bias_angle) * _bias_strength,
        ])


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

    # Save the unbiased target direction so we can compute true directional error later.
    # (alignment vs. biased ideal would DECREASE when CFL corrects the model, which
    # is the opposite of what we want the loss metric to show.)
    _true_tx, _true_ty = tx_n, ty_n

    # Non-IID bias: each particle persistently misperceives the ideal direction.
    # Without CFL aggregation the bias accumulates; federation averages it out.
    _bx = tx_n + particle._local_bias[0]
    _by = ty_n + particle._local_bias[1]
    _bn = math.hypot(_bx, _by)
    if _bn > 0:
        tx_n, ty_n = _bx / _bn, _by / _bn

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

    # --- Update [6]: local_loss (geometric distance + directional error) ---
    # directional_error uses the TRUE target direction (pre-bias), not the biased ideal.
    # When CFL corrects the model toward the true direction, true_alignment rises and
    # this error falls — making the CFL benefit visible in the loss metric.
    max_dist = math.hypot(SIM_DIM[0], SIM_DIM[1])
    geo_loss = min(dist_t / max_dist, 1.0)
    true_alignment = particle.model[0] * _true_tx + particle.model[1] * _true_ty
    directional_error = 1.0 - (true_alignment + 1.0) / 2.0
    particle.model[6] = min(0.5 * geo_loss + 0.5 * directional_error, 1.0)

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

def apply_physics_rules(particles: List[Particle], obstacles: List[Tuple[int, int, int]], g_attract: float, g_repel: float, dt: float, rules: dict = None, attraction_enabled: bool = True):
    """
    Revised Physics: Prevents stacking by adding emergency repulsion.
    rules: optional dict mapping (min(ci,cj), max(ci,cj)) -> float for per-pair forces.
    attraction_enabled: if False, the inter-cluster/intra-cluster gravitational forces
        are disabled (only emergency anti-stacking repulsion remains).  Lets you
        isolate the effect of the simple emergent physics rules on learning.
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

            # 2. EMERGENCY REPULSION (always on — prevents visual stacking)
            if d < PARTICLE_FORCE_LOWER_RANGE * 2:
                F_scalar = 200.0  # Massive repulsion to unclump them

            # 3. NORMAL ATTRACTION/REPULSION (only when emergent rules are enabled)
            elif attraction_enabled:
                g = 0.0

                # Only interact if both have a valid cluster
                if a.cluster_id != -1 and b.cluster_id != -1:
                    ci, cj = a.cluster_id, b.cluster_id
                    if rules is not None:
                        key = (min(ci, cj), max(ci, cj))
                        default = g_attract if ci == cj else g_repel
                        g = rules.get(key, default)
                    elif ci == cj:
                        g = g_attract
                    else:
                        g = g_repel

                # Standard Gravity Formula
                denom = (d ** PARTICLE_POWER_OF_DISTANCE) * len(particles)
                if denom == 0: denom = 0.001
                F_scalar = g * (1 / denom)
            else:
                continue  # attraction off and outside emergency range — no force

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

def _direction_coherence(members):
    """Mean cosine similarity of members' directions to the cluster's own
    mean direction. 1.0 = perfectly aligned, 0 = chaotic, -1 = anti-aligned.
    Used as IFCA's split signal: under IFCA, avg_loss is pinned near the
    local minimum after assignment, so loss can't tell us a cluster is
    incoherent — directional disagreement among members can.
    """
    if len(members) < 2:
        return 1.0
    dirs = np.array([p.model[0:2] for p in members])
    mean_dir = dirs.mean(axis=0)
    norm = np.linalg.norm(mean_dir)
    if norm == 0:
        return 0.0
    mean_dir = mean_dir / norm
    return float((dirs @ mean_dir).mean())


def _ifca_score(particle, target, theta, max_dist):
    """IFCA loss for evaluating cluster k on `particle`.

    Lower is better. Mirrors the per-particle loss from `local_train`'s
    model[6] but evaluated under cluster k's *broadcast* model θ_k and
    cluster k's target. This is what IFCA clients compute locally for
    every candidate model to pick their cluster.
    """
    dx = target[0] - particle.x
    dy = target[1] - particle.y
    dist = math.hypot(dx, dy)
    geo_loss = min(dist / max_dist, 1.0)

    if dist > 0 and theta is not None:
        true_dx, true_dy = dx / dist, dy / dist
        # Alignment of θ_k with the true direction from particle to target_k.
        alignment = float(theta[0] * true_dx + theta[1] * true_dy)  # -1..1
        dir_loss = (1.0 - alignment) / 2.0
    else:
        dir_loss = 0.5  # no info → neutral

    return 0.5 * geo_loss + 0.5 * dir_loss


def _compute_cluster_models(particles, n_clusters):
    """Confidence-weighted mean model per cluster (the broadcast θ_k).

    Returns a list of length n_clusters; entries are length-8 numpy
    arrays with [0:2] re-normalised, or None for empty clusters.
    """
    cluster_models = [None] * n_clusters
    for k in range(n_clusters):
        members = [p for p in particles if p.cluster_id == k]
        if not members:
            continue
        weights = np.array([p.model[2] for p in members])
        weights /= weights.sum() + 1e-8
        theta = sum(w * p.model for w, p in zip(weights, members))
        norm = np.linalg.norm(theta[0:2])
        if norm > 0:
            theta[0:2] = theta[0:2] / norm
        cluster_models[k] = theta
    return cluster_models


def run_cfl_round(particles, kmeans_model, cluster_targets, cluster_colors, cluster_ages, cooldown_counter):
    """IFCA-style federated clustering round (Ghosh et al. 2020).

    Each particle evaluates every cluster's broadcast model θ_k against
    its own data (distance + directional alignment to target_k) and
    picks the cluster with the lowest loss. After assignment, each
    cluster aggregates its members' models with the existing
    age-adaptive blend.

    `kmeans_model` is retained in the signature for backward compat with
    callers that read `.n_clusters` and `.inertia_`. We fit a fresh
    KMeans on identity features at the end purely to keep that contract.
    """
    if not particles:
        return {}, kmeans_model, cluster_targets, cluster_colors, cluster_ages, kmeans_model.n_clusters, None, cooldown_counter

    n = kmeans_model.n_clusters
    old_ids = [p.cluster_id for p in particles]

    # --- IFCA broadcast: build θ_k from current cluster membership ---
    cluster_models = _compute_cluster_models(particles, n)
    max_dist = math.hypot(SIM_DIM[0], SIM_DIM[1])

    # Fallback for clusters that are currently empty: seed θ_k with a
    # unit vector pointing from the simulation centre toward target_k,
    # so the loss is well-defined and the cluster can still attract members.
    cx, cy = SIM_DIM[0] / 2.0, SIM_DIM[1] / 2.0
    for k in range(n):
        if cluster_models[k] is None and k < len(cluster_targets):
            tx, ty = cluster_targets[k]
            dx, dy = tx - cx, ty - cy
            d = math.hypot(dx, dy)
            theta = np.zeros(8, dtype=np.float64)
            if d > 0:
                theta[0], theta[1] = dx / d, dy / d
            theta[2] = 0.5  # middling confidence
            cluster_models[k] = theta

    # --- IFCA assignment: each particle picks argmin loss across θ_k ---
    transfers = defaultdict(int)
    for i, p in enumerate(particles):
        old_id = old_ids[i]

        losses = [
            _ifca_score(p, cluster_targets[k], cluster_models[k], max_dist)
            for k in range(n)
        ]
        new_id = int(np.argmin(losses))

        # Hysteresis: only switch if the chosen cluster's loss is meaningfully
        # below the current cluster's loss. Stops particles flipping between
        # nearly-equivalent clusters when geometry barely favours one.
        if 0 <= old_id < n and old_id != new_id:
            if losses[new_id] >= losses[old_id] * (1.0 - MIGRATION_HYSTERESIS):
                new_id = old_id

        if old_id != new_id:
            transfers[(old_id, new_id)] += 1
        p.cluster_id = new_id

    # --- Aggregation: recompute θ_k from new assignments and blend ---
    aggregated_models = _compute_cluster_models(particles, n)
    for cid in range(n):
        aggregated = aggregated_models[cid]
        if aggregated is None:
            continue
        members = [p for p in particles if p.cluster_id == cid]

        # Blend ratio depends on cluster age (newborn clusters keep more
        # local state so they don't immediately re-merge with the parent).
        age = cluster_ages.get(cid, BLEND_MATURITY_ROUNDS)
        t = min(age / BLEND_MATURITY_ROUNDS, 1.0)  # 0.0 = newborn, 1.0 = mature
        local_weight  = BLEND_LOCAL_NEW + t * (BLEND_LOCAL_MATURE - BLEND_LOCAL_NEW)
        global_weight = 1.0 - local_weight

        for p in members:
            p.model = local_weight * p.model + global_weight * aggregated
            p.model[2] = np.clip(p.model[2], 0.01, 1.0)
            p.model[3:5] = np.clip(p.model[3:5], 0.0, 1.0)

    # --- Re-fit KMeans purely for the .inertia_ / .n_clusters contract ---
    # IFCA itself doesn't need this; downstream logging/printing reads it.
    try:
        identity_models = np.array([p.model[IDENTITY_SLICE] for p in particles])
        init_centroids = np.array([
            np.mean([p.model[IDENTITY_SLICE] for p in particles if p.cluster_id == cid], axis=0)
            if any(p.cluster_id == cid for p in particles)
            else identity_models[np.random.randint(len(identity_models))]
            for cid in range(n)
        ])
        warm_kmeans = KMeans(n_clusters=n, init=init_centroids, n_init=1, random_state=0)
        warm_kmeans.fit(identity_models)
        kmeans_model = warm_kmeans
    except Exception:
        pass

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
        merge_pair = None

        # Path 1: Absorb tiny clusters. A cluster with fewer than MIN_CLUSTER_SIZE
        # members is effectively dead under IFCA (its θ_k is too noisy) — fold
        # it into the cluster whose target is nearest, regardless of age.
        tiny = [cid for cid in range(n) if stats[cid]['size'] < MIN_CLUSTER_SIZE]
        for tcid in tiny:
            tx, ty = cluster_targets[tcid]
            other = [cid for cid in range(n) if cid != tcid and stats[cid]['size'] > 0]
            if not other:
                continue
            nearest = min(
                other,
                key=lambda cid: math.hypot(cluster_targets[cid][0] - tx,
                                           cluster_targets[cid][1] - ty),
            )
            merge_pair = (min(nearest, tcid), max(nearest, tcid))
            print(f"   > MERGE-tiny: cluster {tcid} (size={stats[tcid]['size']}) absorbed into {nearest}")
            break

        # Path 2: Standard similarity-based merge for mature, redundant clusters.
        if merge_pair is None:
            best_score, candidate = -float('inf'), None
            for a in range(n):
                for b in range(a + 1, n):
                    ma = stats[a]['mean_model']
                    mb = stats[b]['mean_model']
                    if ma is None or mb is None:
                        continue
                    if cluster_ages.get(a, 0) < BLEND_MATURITY_ROUNDS or \
                       cluster_ages.get(b, 0) < BLEND_MATURITY_ROUNDS:
                        continue

                    sim = float(np.dot(ma[0:2], mb[0:2]))
                    # Spatial-target component: 1.0 when targets coincide,
                    # 0.0 when they're MERGE_TARGET_DISTANCE or farther apart.
                    tdist = math.hypot(
                        cluster_targets[a][0] - cluster_targets[b][0],
                        cluster_targets[a][1] - cluster_targets[b][1],
                    )
                    target_score = max(0.0, 1.0 - tdist / MERGE_TARGET_DISTANCE)
                    # Combined score: high direction sim OR very close targets.
                    score = max(sim, target_score)
                    if score > best_score:
                        best_score, candidate = score, (a, b)

            if candidate and best_score >= MERGE_SIMILARITY_THRESHOLD:
                merge_pair = candidate
                print(f"   > MERGE-redundant: {candidate} score={best_score:.3f}")

        if merge_pair:
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

            return transfers, new_kmeans, new_targets, new_colors, new_ages, new_n, ('merge', drop), 10

    # SPLIT — under IFCA the right signal is directional incoherence among
    # members (avg_loss is mechanically pinned at the local min after assign).
    if n < MAX_CLUSTERS:
        eligible = [cid for cid in range(n)
                    if stats[cid]['size'] >= MIN_CLUSTER_SIZE * 2]

        worst_cid, worst_score = None, 0.0
        for cid in eligible:
            members = [p for p in particles if p.cluster_id == cid]
            coherence = _direction_coherence(members)
            avg_loss = stats[cid]['avg_loss']

            # Combined "wants to split" score: how much each criterion exceeds
            # its threshold. Either alone can trigger.
            coh_excess  = max(0.0, SPLIT_COHERENCE_THRESHOLD - coherence)
            loss_excess = max(0.0, avg_loss - SPLIT_LOSS_THRESHOLD)
            score = coh_excess + loss_excess
            if score > worst_score:
                worst_score, worst_cid = score, cid

        if worst_cid is not None and worst_score > 0.0:
            members = [p for p in particles if p.cluster_id == worst_cid]
            members_coh = _direction_coherence(members)
            members_avg = stats[worst_cid]['avg_loss']
            print(f"   > SPLIT-trigger: cluster {worst_cid} "
                  f"coherence={members_coh:.2f} avg_loss={members_avg:.2f}")

            # Split along the direction-disagreement axis: 2-means on member
            # heading vectors. Members in one heading group form the new cluster.
            # Falls back to spatial-x split when 2-means is too unbalanced.
            new_members = None
            if len(members) >= MIN_CLUSTER_SIZE * 2:
                try:
                    dirs = np.array([p.model[0:2] for p in members])
                    km2 = KMeans(n_clusters=2, n_init=3, random_state=0)
                    labels = km2.fit_predict(dirs)
                    group_a = [m for m, l in zip(members, labels) if l == 0]
                    group_b = [m for m, l in zip(members, labels) if l == 1]
                    if len(group_a) >= MIN_CLUSTER_SIZE and len(group_b) >= MIN_CLUSTER_SIZE:
                        # Smaller group becomes the new cluster.
                        new_members = group_b if len(group_b) <= len(group_a) else group_a
                except Exception:
                    pass

            if new_members is None:
                members_sorted = sorted(members, key=lambda p: p.x)
                split_count = max(MIN_CLUSTER_SIZE, len(members_sorted) // 2)
                new_members = members_sorted[:split_count]

            new_cid = n
            for p in new_members:
                p.cluster_id = new_cid
                p._prev_cluster_id = -1
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
    PARTICLE_COLOR_RED,
    PARTICLE_COLOR_YELLOW,
    PARTICLE_COLOR_GREEN,
    PARTICLE_COLOR_BLUE,
    PARTICLE_COLOR_WHITE,
    (0, 255, 255),   # cyan — 6th slot for MAX_CLUSTERS=6
]

def _pick_unused_color(used_colors):
    for c in CLUSTER_PALETTE:
        if c not in used_colors:
            return c
    return (200, 200, 200)  # fallback gray if palette exhausted
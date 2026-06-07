"""@file particle.py
@brief Agent model and local training (the cognitive domain).

This module is the agent half of the simulation. It defines:
  - Particle — a federated agent coupling an 8-D cognitive model with 2-D physics state.
  - local_train() — one local training step per agent (cognitive domain).
  - update_peer_alignment() — spatial-neighborhood feedback into the model.

The physical-domain integrator (apply_physics_rules) lives in physics.py and the
IFCA clustered-federated-learning round lives in cfl.py — both import Particle
from here, so the dependency is one-directional.

Original emergent-physics base by Vishal Paudel; extended for clustered federated
learning (CFL) as part of the thesis. See README.md for the conceptual mapping.
"""
from typing import Tuple

import random
import math
import numpy as np

from src.constants import (
    SIM_DIM, SIM_WIDTH, SIM_HEIGHT,
    PARTICLE_DEFAULT_RADIUS, WALL_BOUNDARY, PARTICLE_MAX_SPEED,
)
from src import cfl_params

class Particle:
    """@brief A single federated agent: cognitive model + physical body.

    Each particle is simultaneously a federated-learning *client* (it owns an
    8-D @c model vector and trains it locally) and a physical body (position,
    velocity) that moves under emergent forces. The two are coupled: the learned
    heading drives motion, and motion/neighborhood feed back into the model.

    Model layout (indices into @c self.model):
      - [0,1] dir_x, dir_y     — learned heading (unit vector); the clustering identity.
      - [2]   confidence       — 0..1 EMA of how consistently it converges.
      - [3]   obstacle_pressure— 0..1 decaying memory of recent obstacle proximity.
      - [4]   peer_alignment   — 0..1 cosine similarity to same-cluster neighbors.
      - [5]   rounds_stable    — normalized rounds since the last cluster change.
      - [6]   local_loss       — 0..1 distance + directional error to the target.
      - [7]   drift_velocity   — 0..1 EMA of speed.
    """

    def __init__(self, x, v, c, target_idx, true_cluster_id: int,
                 cluster_bias_dir: Tuple[float, float], r=PARTICLE_DEFAULT_RADIUS):
        """@brief Construct an agent with a random heading, target, and non-IID bias.

        @param x (x, y) initial position.
        @param v (vx, vy) initial velocity.
        @param c RGB color tuple (overwritten by the cluster color at render time).
        @param target_idx Index of the cluster target this agent initially chases.
        @param true_cluster_id Ground-truth latent cluster label (the structure CFL is
            meant to recover).
        @param cluster_bias_dir Unit vector for this true cluster's shared bias direction.
            @c _local_bias is drawn as a mix of this direction and per-particle
            noise (see @c cfl_params.get_bias_cluster_fraction).
        @param r Draw radius in pixels.
        """
        self.x = x[0]
        self.y = x[1]
        self.vx = v[0]
        self.vy = v[1]
        self.c = c
        self.r = r
        self.target_idx = target_idx
        self.cluster_id = -1
        self._true_cluster_id = true_cluster_id

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

        # Per-particle inner target — what this particle actually chases.
        # The group/cluster target is just the average of these for visualization.
        _margin = WALL_BOUNDARY * 4
        self._target_x     = random.uniform(_margin, SIM_WIDTH  - _margin)
        self._target_y     = random.uniform(_margin, SIM_HEIGHT - _margin)
        self._target_angle = random.uniform(0, 2 * math.pi)

        # Per-particle directional bias — the non-IID analog from the CFL paper.
        # Each "client" systematically misperceives the ideal direction by a fixed
        # offset. When a cluster_bias_dir is provided, the bias direction is a mix
        # of that cluster-shared signal and per-particle noise — so members of the
        # same true cluster share a latent direction that CFL aggregation can
        # recover, while individual learning cannot. Magnitude stays random per
        # particle so even within-cluster, no two agents are identical.
        _bias_strength = random.uniform(cfl_params.get_bias_mag_lo(), cfl_params.get_bias_mag_hi())
        _noise_angle = random.uniform(0, 2 * math.pi)
        _nx, _ny = math.cos(_noise_angle), math.sin(_noise_angle)
        _cx, _cy = cluster_bias_dir
        _bias_frac = cfl_params.get_bias_cluster_fraction()
        _mx = _bias_frac * _cx + (1.0 - _bias_frac) * _nx
        _my = _bias_frac * _cy + (1.0 - _bias_frac) * _ny
        _mn = math.hypot(_mx, _my)
        if _mn > 0:
            _mx, _my = _mx / _mn, _my / _mn
        else:
            _mx, _my = _nx, _ny
        self._local_bias = np.array([_mx * _bias_strength, _my * _bias_strength])


def local_train(particle, current_target_pos, obstacles, learning_rate=0.1,
                emergent_pull=(0.0, 0.0)):
    """@brief One local training step for a single agent (the cognitive update).

    Nudges the agent's learned heading @c model[0:2] toward its ideal direction —
    the true direction to its target corrupted by the agent's fixed non-IID bias,
    then blended with obstacle-avoidance AND any net emergent pull (attract/
    repel from neighbors) — and refreshes the situational model dimensions:
    confidence [2], obstacle_pressure [3], local_loss [6] and drift_velocity
    [7]. The loss uses the *true* (unbiased) target direction so that
    federation correcting the bias is visible as a loss decrease.

    Wiring the emergent pull into the ideal direction (not just into velocity,
    where apply_physics_rules already does it) means the LEARNED MODEL also
    reflects the physical push — direction_precision drops under
    attraction_enabled=True, decoupling the cognitive and physical contributions
    of the configuration.

    @param particle The Particle to train; mutated in place.
    @param current_target_pos (x, y) of this agent's personal inner target.
    @param obstacles List of (ox, oy, radius) obstacle circles to avoid.
    @param learning_rate Base step size for the heading update (scaled up under pressure).
    @param emergent_pull (ex, ey) net emergent force vector for this particle this
        step. Direction is what matters — the vector is normalized to a unit
        direction before being added to the ideal with @c cfl_params.get_emergent_model_weight.
        Pass (0.0, 0.0) when attraction_enabled=False to keep behavior
        identical to the pre-emergent-model code.
    @return None — @p particle.model is updated in place.
    """
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

    # Emergent pull: normalized neighbor-force direction added to the ideal so
    # the learned heading also reflects where physics is dragging this agent.
    # Magnitude of the input vector is discarded — only direction is kept, then
    # scaled by cfl_params.get_emergent_model_weight(). Comparable in scale to the unit-length
    # target direction so it noticeably bends model[0:2] when present.
    ex_in, ey_in = emergent_pull
    em_norm = math.hypot(ex_in, ey_in)
    if em_norm > 0.0:
        _emergent_w = cfl_params.get_emergent_model_weight()
        ideal_x += (ex_in / em_norm) * _emergent_w
        ideal_y += (ey_in / em_norm) * _emergent_w

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
    """@brief Feed spatial-neighborhood consensus back into each agent's model (physical -> cognitive).

    Sets @c model[4] (peer_alignment) of every agent to the cosine similarity
    between its own heading and the mean heading of same-cluster neighbors within
    @p neighbor_radius, remapped to 0..1. Agents with no cluster or no neighbors
    get 0.

    @param particles List of all Particle agents; mutated in place.
    @param neighbor_radius Pixel radius defining the spatial neighborhood.
    @return None.
    """
    if not particles:
        return

    positions  = np.array([(p.x, p.y) for p in particles])
    directions = np.array([p.model[0:2] for p in particles])
    cids       = np.array([p.cluster_id for p in particles])

    for i, p in enumerate(particles):
        if p.cluster_id == -1:
            p.model[4] = 0.0
            continue

        same_cluster = (cids == p.cluster_id)
        same_cluster[i] = False

        if not np.any(same_cluster):
            p.model[4] = 0.0
            continue

        diffs = positions[same_cluster] - positions[i]
        in_radius = np.linalg.norm(diffs, axis=1) < neighbor_radius

        neighbor_dirs = directions[same_cluster][in_radius]
        if len(neighbor_dirs) == 0:
            p.model[4] = 0.0
            continue

        avg_dir = neighbor_dirs.mean(axis=0)
        norm = np.linalg.norm(avg_dir)
        if norm > 0:
            avg_dir /= norm

        cos_sim = float(np.dot(directions[i], avg_dir))
        p.model[4] = (cos_sim + 1) / 2

"""@file cfl.py
@brief Clustered federated learning (IFCA) round, clustering helpers, and group setup.

Defines:
  - instantiate_group() — spawn a group of agents sharing a latent cluster bias.
  - compute_cluster_stats() — per-cluster health for split/merge and the logger.
  - run_cfl_round() — one IFCA-style clustered federated learning round
    (confidence-weighted aggregation + spontaneous cluster split / merge).

Depends on particle.Particle (one-directional: particle.py does not import this
module). CFL tunables come from cfl_params via its getters.
"""
from typing import Tuple, List
from collections import defaultdict

import random
import math
import numpy as np
from sklearn.cluster import KMeans

from src.constants import (
    SIM_DIM, SIM_WIDTH, SIM_HEIGHT, CLUSTER_PALETTE,
)
from src import cfl_params
from src.particle import Particle


def compute_cluster_stats(particles, n_clusters):
    """@brief Aggregate per-cluster health used by the split/merge logic and logger.

    @param particles List of all Particle agents.
    @param n_clusters Number of clusters to report on (ids 0..n_clusters-1).
    @return dict cluster_id -> {size, avg_loss, avg_confidence, mean_model}, where
        @c mean_model is the element-wise mean 8-D model (None for empty clusters).
    """
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
    """@brief Directional coherence of a cluster — the IFCA split signal.

    Mean cosine similarity of members' headings to the cluster's own mean
    heading: 1.0 = perfectly aligned, 0 = chaotic, -1 = anti-aligned. Under IFCA
    avg_loss is pinned near the local minimum after assignment, so loss cannot
    reveal an incoherent cluster — directional disagreement among members can.

    @param members List of Particle agents in one cluster.
    @return float coherence in [-1, 1] (1.0 when fewer than two members).
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
    """@brief IFCA loss for evaluating cluster k on a single agent (lower is better).

    Mirrors the per-agent loss from local_train() (model[6]) but evaluated
    under cluster k's *broadcast* model theta and cluster k's target. This is the
    score each IFCA client computes locally for every candidate model in order to
    pick its cluster (argmin over k).

    @param particle The agent being evaluated.
    @param target (x, y) spatial target of cluster k.
    @param theta Cluster k's broadcast 8-D model (its heading is theta[0:2]); may be None.
    @param max_dist Diagonal of the simulation area, used to normalize distance.
    @return float loss = 0.5 * geometric_loss + 0.5 * directional_loss.
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
    """@brief Confidence-weighted mean model per cluster — the broadcast theta_k.

    Each cluster's model is the mean of its members' 8-D models weighted by
    confidence (model[2]), with the heading [0:2] re-normalized to a unit vector.

    @param particles List of all Particle agents.
    @param n_clusters Number of clusters.
    @return list of length @p n_clusters; each entry is a length-8 numpy array, or
        None for an empty cluster.
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


def _merge_clusters(particles, n, stats, cluster_targets, cluster_colors, cluster_ages, transfers):
    """@brief One merge step: absorb a tiny or redundant cluster, if any.

    Two paths, tried in order: (1) absorb any cluster smaller than
    @c min_cluster_size into its spatially nearest neighbor; (2) merge the most
    redundant mature pair (high heading similarity OR near-coincident targets).

    @param particles List of all Particle agents; cluster_id mutated in place on a merge.
    @param n Current number of clusters.
    @param stats Per-cluster stats from compute_cluster_stats().
    @param cluster_targets List of (x, y) cluster targets.
    @param cluster_colors dict cluster_id -> RGB (key -1 reserved for unassigned).
    @param cluster_ages dict cluster_id -> age in rounds.
    @param transfers dict (src, dst) -> migration count, threaded into the return tuple.
    @return The 8-tuple run_cfl_round returns on a merge (event = ('merge', dropped_id)),
        or None when no merge fires.
    """
    if n <= cfl_params.get_min_clusters():
        return None

    min_cluster_size = cfl_params.get_min_cluster_size()
    merge_pair = None

    # Path 1: Absorb tiny clusters. A cluster with fewer than min_cluster_size
    # members is effectively dead under IFCA (its θ_k is too noisy) — fold
    # it into the cluster whose target is nearest, regardless of age.
    tiny = [cid for cid in range(n) if stats[cid]['size'] < min_cluster_size]
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
                maturity_rounds = cfl_params.get_blend_maturity_rounds()
                if (cluster_ages.get(a, 0) < maturity_rounds or
                        cluster_ages.get(b, 0) < maturity_rounds):
                    continue

                sim = float(np.dot(ma[0:2], mb[0:2]))
                # Spatial-target component: 1.0 when targets coincide,
                # 0.0 when they're the merge-target distance or farther apart.
                tdist = math.hypot(
                    cluster_targets[a][0] - cluster_targets[b][0],
                    cluster_targets[a][1] - cluster_targets[b][1],
                )
                target_score = max(0.0, 1.0 - tdist / cfl_params.get_merge_target_distance())
                # Combined score: high direction sim OR very close targets.
                score = max(sim, target_score)
                if score > best_score:
                    best_score, candidate = score, (a, b)

        if candidate and best_score >= cfl_params.get_merge_similarity_threshold():
            merge_pair = candidate
            print(f"   > MERGE-redundant: {candidate} score={best_score:.3f}")

    if not merge_pair:
        return None

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


def _split_clusters(particles, n, stats, cluster_targets, cluster_colors, cluster_ages, transfers):
    """@brief One split step: carve an incoherent cluster in two, if one qualifies.

    Under IFCA the split signal is directional incoherence among members
    (avg_loss is mechanically pinned at the local min after assignment). The
    worst-scoring eligible cluster is split via 2-means on member heading
    vectors, falling back to a spatial-x split when 2-means is too unbalanced.

    @param particles List of all Particle agents; cluster_id mutated in place on a split.
    @param n Current number of clusters.
    @param stats Per-cluster stats from compute_cluster_stats().
    @param cluster_targets List of (x, y) cluster targets.
    @param cluster_colors dict cluster_id -> RGB (key -1 reserved for unassigned).
    @param cluster_ages dict cluster_id -> age in rounds.
    @param transfers dict (src, dst) -> migration count, threaded into the return tuple.
    @return The 8-tuple run_cfl_round returns on a split (event = 'split'), or None
        when no split fires.
    """
    if n >= cfl_params.get_max_clusters():
        return None

    min_cluster_size = cfl_params.get_min_cluster_size()
    eligible = [cid for cid in range(n)
                if stats[cid]['size'] >= min_cluster_size * 2]

    worst_cid, worst_score = None, 0.0
    for cid in eligible:
        members = [p for p in particles if p.cluster_id == cid]
        coherence = _direction_coherence(members)
        avg_loss = stats[cid]['avg_loss']

        # Combined "wants to split" score: how much each criterion exceeds
        # its threshold. Either alone can trigger.
        coh_excess  = max(0.0, cfl_params.get_split_coherence_threshold() - coherence)
        loss_excess = max(0.0, avg_loss - cfl_params.get_split_loss_threshold())
        score = coh_excess + loss_excess
        if score > worst_score:
            worst_score, worst_cid = score, cid

    if worst_cid is None or worst_score <= 0.0:
        return None

    members = [p for p in particles if p.cluster_id == worst_cid]
    members_coh = _direction_coherence(members)
    members_avg = stats[worst_cid]['avg_loss']
    print(f"   > SPLIT-trigger: cluster {worst_cid} "
          f"coherence={members_coh:.2f} avg_loss={members_avg:.2f}")

    # Split along the direction-disagreement axis: 2-means on member
    # heading vectors. Members in one heading group form the new cluster.
    # Falls back to spatial-x split when 2-means is too unbalanced.
    new_members = None
    if len(members) >= min_cluster_size * 2:
        try:
            dirs = np.array([p.model[0:2] for p in members])
            km2 = KMeans(n_clusters=2, n_init=3, random_state=0)
            labels = km2.fit_predict(dirs)
            group_a = [m for m, l in zip(members, labels) if l == 0]
            group_b = [m for m, l in zip(members, labels) if l == 1]
            if len(group_a) >= min_cluster_size and len(group_b) >= min_cluster_size:
                # Smaller group becomes the new cluster.
                new_members = group_b if len(group_b) <= len(group_a) else group_a
        except Exception:
            pass

    if new_members is None:
        members_sorted = sorted(members, key=lambda p: p.x)
        split_count = max(min_cluster_size, len(members_sorted) // 2)
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


def run_cfl_round(particles, kmeans_model, cluster_targets, cluster_colors, cluster_ages, cooldown_counter):
    """@brief One IFCA-style clustered federated learning round (Ghosh et al. 2020).

    Steps: (1) broadcast a confidence-weighted model theta_k per cluster; (2) each
    agent re-assigns to the cluster whose theta_k gives the lowest loss (argmin),
    subject to migration hysteresis; (3) aggregate members toward theta_k with an
    age-adaptive blend; (4) rescue stuck stragglers; (5) after a cooldown, possibly
    @b merge redundant/tiny clusters or @b split an incoherent one. A fresh KMeans
    is fit at the end purely to preserve the @c .n_clusters / @c .inertia_ contract
    that the logger reads — IFCA itself does not need it.

    @param particles List of all Particle agents; cluster_id and model mutated in place.
    @param kmeans_model Previous KMeans (read for @c .n_clusters; re-fit and returned).
    @param cluster_targets List of (x, y) spatial targets, one per cluster.
    @param cluster_colors dict cluster_id -> RGB (key -1 reserved for unassigned).
    @param cluster_ages dict cluster_id -> age in rounds (drives the blend ratio).
    @param cooldown_counter Rounds remaining before another split/merge may fire.
    @return 8-tuple (transfers, kmeans_model, cluster_targets, cluster_colors,
        cluster_ages, num_clusters, event, cooldown_counter), where @c transfers maps
        (src, dst) -> migration count and @c event is None, 'split', or ('merge', dropped_id).
    """
    if not particles:
        return {}, kmeans_model, cluster_targets, cluster_colors, cluster_ages, kmeans_model.n_clusters, None, cooldown_counter

    n = kmeans_model.n_clusters
    old_ids = [p.cluster_id for p in particles]

    # --- IFCA broadcast: build θ_k from current cluster membership ---
    cluster_models = _compute_cluster_models(particles, n)
    max_dist = math.hypot(SIM_DIM[0], SIM_DIM[1])

    # Fallback for clusters that are currently empty: seed θ_k with a
    # unit vector pointing from the simulation center toward target_k,
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
        # nearly-equivalent clusters when geometry barely favors one.
        if 0 <= old_id < n and old_id != new_id:
            if losses[new_id] >= losses[old_id] * (1.0 - cfl_params.get_migration_hysteresis()):
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
        maturity_rounds = cfl_params.get_blend_maturity_rounds()
        age = cluster_ages.get(cid, maturity_rounds)
        t = min(age / maturity_rounds, 1.0)  # 0.0 = newborn, 1.0 = mature
        local_new = cfl_params.get_blend_local_new()
        local_weight  = local_new + t * (cfl_params.get_blend_local_mature() - local_new)
        global_weight = 1.0 - local_weight

        for p in members:
            p.model = local_weight * p.model + global_weight * aggregated
            p.model[2] = np.clip(p.model[2], 0.01, 1.0)
            p.model[3:5] = np.clip(p.model[3:5], 0.0, 1.0)

    # --- Re-fit KMeans purely for the .inertia_ / .n_clusters contract ---
    # IFCA itself doesn't need this; downstream logging/printing reads it.
    try:
        identity_slice = cfl_params.get_identity_slice()
        identity_models = np.array([p.model[identity_slice] for p in particles])
        init_centroids = np.array([
            np.mean([p.model[identity_slice] for p in particles if p.cluster_id == cid], axis=0)
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
    straggler_loss = cfl_params.get_straggler_loss()
    straggler_drift = cfl_params.get_straggler_drift()
    for p in particles:
        if p.model[6] > straggler_loss and p.model[7] < straggler_drift:
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
    merged = _merge_clusters(particles, n, stats, cluster_targets,
                             cluster_colors, cluster_ages, transfers)
    if merged is not None:
        return merged

    # SPLIT
    split = _split_clusters(particles, n, stats, cluster_targets,
                            cluster_colors, cluster_ages, transfers)
    if split is not None:
        return split

    return transfers, kmeans_model, cluster_targets, cluster_colors, cluster_ages, n, None, 0

def instantiate_group(
    num: int,
    c: tuple,
    frame: Tuple[Tuple[int, int], Tuple[int, int]],
    target_idx: int,
    true_cluster_id: int,
    cluster_bias_dir: Tuple[float, float],
    ) -> List[Particle]:
    """@brief Instantiate a group of agents spawned uniformly within a frame.

    @param num Number of particles to create.
    @param c RGB color tuple for the group.
    @param frame Spawn box ((x_min, x_max), (y_min, y_max)).
    @param target_idx Index of the cluster target this group initially chases.
    @param true_cluster_id Ground-truth latent label shared by all agents in this group.
    @param cluster_bias_dir Unit vector for this true cluster's shared bias direction.
    @return List[Particle] the newly created agents.
    """
    random.seed()
    group = []

    for _ in range(num):
        x = random.randint(frame[0][0], frame[0][1])
        y = random.randint(frame[1][0], frame[1][1])

        group.append(Particle(
            x=(x, y), v=(0.0, 0.0), c=c, target_idx=target_idx,
            true_cluster_id=true_cluster_id,
            cluster_bias_dir=cluster_bias_dir,
        ))

    return group


def _pick_unused_color(used_colors):
    """@brief Pick the first palette color not already in use (for a new cluster).

    @param used_colors Set/collection of RGB tuples currently assigned.
    @return RGB tuple — an unused palette color, or gray if the palette is exhausted.
    """
    for c in CLUSTER_PALETTE:
        if c not in used_colors:
            return c
    return (200, 200, 200)  # fallback gray if palette exhausted
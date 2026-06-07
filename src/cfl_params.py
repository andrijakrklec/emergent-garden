"""@file cfl_params.py
@brief Clustered-federated-learning (IFCA) tunables for the simulation.

These parameters are used in particle.py: cluster-count bounds,
split/merge thresholds, the non-IID bias structure, migration hysteresis, the
behavioral / obstacle forces, the local<->global blend schedule, and the
emergent-pull weight.
"""

# --- CLUSTER-COUNT BOUNDS ---
_MIN_CLUSTERS = 2
_MAX_CLUSTERS = 6

# Split fires when EITHER (a) avg_loss exceeds threshold (a cluster genuinely
# can't serve its members) OR (b) directional coherence among members drops
# below threshold (members disagree on heading — the natural IFCA split signal,
# since IFCA already pins avg_loss near the local minimum).
_SPLIT_LOSS_THRESHOLD = 0.30
_SPLIT_COHERENCE_THRESHOLD = 0.70   # mean cos-sim to cluster-mean direction

# Merge thresholds: cosine sim of mean direction (loosened from 0.97 — IFCA's
# bias-averaged means cluster more cleanly so 0.92 is a reasonable bar) plus
# spatial-target proximity (two clusters whose targets have drifted close
# together are functionally redundant under IFCA's loss).
_MERGE_SIMILARITY_THRESHOLD = 0.92
_MERGE_TARGET_DISTANCE = 120.0      # px — targets closer than this favor merge
_MIN_CLUSTER_SIZE = 3               # clusters smaller than this get absorbed

_BLEND_MATURITY_ROUNDS = 15  # rounds until a new cluster reaches full global blend
_BLEND_LOCAL_NEW   = 0.75    # local weight for a freshly split cluster

# Non-IID bias structure: each particle's _local_bias direction is a mix of
# its true cluster's shared bias direction and a per-particle random direction.
# BIAS_CLUSTER_FRACTION controls how much of the bias direction is the latent
# cluster signal vs per-particle noise. Higher α → more recoverable latent
# structure → bigger CFL benefit in the ablation.
#
# Magnitude is modest (subdominant to the unit-length target direction) so
# direction precision and loss remain meaningful: model still navigates toward
# the target, bias just perturbs the heading. Recoverability of the latent
# cluster signal comes from cluster members SHARING a target (handled in
# game.py via per-true-cluster anchors), not from oversized bias.
_BIAS_CLUSTER_FRACTION = 0.7
_BIAS_MAG_LO = 0.25
_BIAS_MAG_HI = 0.60

# Identity slice used for KMeans assignment.
# [0:2] direction is the persistent learned identity; [2] confidence is a slow EMA.
# Features [3:8] are situational (pressure / alignment / loss / drift / stability)
# and would cause spurious migrations if used for clustering.
_IDENTITY_SLICE = slice(0, 2)

# Hysteresis: a particle only migrates if the new cluster's loss is at least
# this fraction lower than the current cluster's loss.
# 0.0 = no hysteresis (every flip migrates), 0.5 = need 50% improvement,
# 0.6 = need 60% (current — middle ground: moderate per-round churn while
# letting IFCA recover purity in <30 rounds).
_MIGRATION_HYSTERESIS = 0.6

_BEHAVIORAL_FORCE  = 30.0    # goal-directed push from model[0:2], scaled by confidence
_STRAGGLER_LOSS    = 0.60   # local_loss threshold to consider a particle stranded
_STRAGGLER_DRIFT   = 0.04   # drift_velocity threshold below which a particle is "stuck"

_OBSTACLE_SOFT_ZONE     = 20    # px beyond physical edge where pre-contact repulsion starts
_OBSTACLE_SOFT_STRENGTH = 60.0 # base strength of the soft-zone push
_BLEND_LOCAL_MATURE = 0.40   # local weight once matured (your existing value)

# How strongly the emergent pull (net attract/repel force from neighbors)
# biases the learned heading in local_train. The pull is normalized to a unit
# vector first, then added to the ideal direction with this weight. ~1.5 means
# emergent influence is comparable to the unit-length target direction —
# enough to noticeably drag direction_precision when attraction physics is on.
_EMERGENT_MODEL_WEIGHT = 1.5


# --- GETTERS ---

def get_min_clusters() -> int:
    """@brief Minimum number of clusters the federation may collapse to."""
    return _MIN_CLUSTERS


def get_max_clusters() -> int:
    """@brief Maximum number of clusters the federation may split into."""
    return _MAX_CLUSTERS


def get_split_loss_threshold() -> float:
    """@brief avg_loss above which a cluster is a split candidate."""
    return _SPLIT_LOSS_THRESHOLD


def get_split_coherence_threshold() -> float:
    """@brief Mean cos-sim to cluster-mean direction below which a cluster splits."""
    return _SPLIT_COHERENCE_THRESHOLD


def get_merge_similarity_threshold() -> float:
    """@brief Combined direction/target score at/above which two clusters merge."""
    return _MERGE_SIMILARITY_THRESHOLD


def get_merge_target_distance() -> float:
    """@brief Target separation (px) beyond which the spatial merge score is zero."""
    return _MERGE_TARGET_DISTANCE


def get_min_cluster_size() -> int:
    """@brief Clusters smaller than this are absorbed; splits must beat 2x this."""
    return _MIN_CLUSTER_SIZE


def get_blend_maturity_rounds() -> int:
    """@brief Rounds until a new cluster reaches full global blend."""
    return _BLEND_MATURITY_ROUNDS


def get_blend_local_new() -> float:
    """@brief Local weight for a freshly split (newborn) cluster."""
    return _BLEND_LOCAL_NEW


def get_blend_local_mature() -> float:
    """@brief Local weight once a cluster has matured."""
    return _BLEND_LOCAL_MATURE


def get_bias_cluster_fraction() -> float:
    """@brief Fraction of the non-IID bias direction that is the shared cluster signal."""
    return _BIAS_CLUSTER_FRACTION


def get_bias_mag_lo() -> float:
    """@brief Lower bound of the per-particle bias magnitude."""
    return _BIAS_MAG_LO


def get_bias_mag_hi() -> float:
    """@brief Upper bound of the per-particle bias magnitude."""
    return _BIAS_MAG_HI


def get_identity_slice() -> slice:
    """@brief Model slice used as the clustering identity (learned direction)."""
    return _IDENTITY_SLICE


def get_migration_hysteresis() -> float:
    """@brief Fractional loss improvement required before a particle migrates."""
    return _MIGRATION_HYSTERESIS


def get_behavioral_force() -> float:
    """@brief Goal-directed push from model[0:2], scaled by confidence."""
    return _BEHAVIORAL_FORCE


def get_straggler_loss() -> float:
    """@brief local_loss threshold above which a particle is considered stranded."""
    return _STRAGGLER_LOSS


def get_straggler_drift() -> float:
    """@brief drift_velocity threshold below which a particle is considered stuck."""
    return _STRAGGLER_DRIFT


def get_obstacle_soft_zone() -> int:
    """@brief Distance (px) beyond the physical edge where soft repulsion starts."""
    return _OBSTACLE_SOFT_ZONE


def get_obstacle_soft_strength() -> float:
    """@brief Base strength of the obstacle soft-zone push."""
    return _OBSTACLE_SOFT_STRENGTH


def get_emergent_model_weight() -> float:
    """@brief Weight of the normalized emergent pull when bending the learned heading."""
    return _EMERGENT_MODEL_WEIGHT

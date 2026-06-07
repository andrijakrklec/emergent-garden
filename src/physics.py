"""@file physics.py
@brief Pure emergent-physics force library — standalone, no project dependencies.

A dependency-free toolkit of the force/collision primitives used by the
simulation's physics tick. Every function is a pure scalar/tuple computation
with no knowledge of Particle, the config, or the simulation — callers pass in
plain numbers, so these can be unit-tested or reused in isolation.

Primitives:
  - gravity_force()           — the standard inverse-distance force scalar.
  - emergency_repulsion()     — the fixed anti-stacking repulsion scalar.
  - interaction_coefficient() — pick the attract/repel coefficient g for a pair.
  - pair_force()              — full pairwise scalar (emergency + normal combined).
  - soft_obstacle_push()      — quadratic pre-contact obstacle repulsion magnitude.
  - reflect_velocity()        — bounce a velocity off a surface normal.
  - clamp_speed()             — cap a velocity vector's magnitude.
"""

# Fixed repulsion applied when two bodies are nearly coincident, to stop them
# stacking into a single point. Large relative to the normal gravity force.
EMERGENCY_REPULSION_STRENGTH = 200.0


def gravity_force(g, distance, count, power=1):
    """@brief Standard inverse-distance force scalar: g / (distance**power * count).

    Multiply the returned scalar by the (dx, dy) component delta between the two
    bodies to get the force vector. @p count divides the force so denser swarms
    don't blow up. Guards a zero denominator.

    @param g Interaction coefficient (negative attracts, positive repels).
    @param distance Center-to-center distance between the two bodies.
    @param count Number of interacting bodies (normalizes the total force).
    @param power Exponent on distance (1 = inverse-linear falloff).
    @return float force scalar.
    """
    denom = (distance ** power) * count
    if denom == 0:
        denom = 0.001
    return g * (1.0 / denom)


def emergency_repulsion(strength=EMERGENCY_REPULSION_STRENGTH):
    """@brief The fixed anti-stacking repulsion scalar (positive = repels).

    @param strength Magnitude of the emergency push.
    @return float — @p strength, as the force scalar.
    """
    return strength


def interaction_coefficient(cluster_a, cluster_b, g_attract, g_repel, rules=None):
    """@brief Resolve the attract/repel coefficient g for a pair of bodies.

    Same-cluster pairs attract (@p g_attract), cross-cluster pairs repel
    (@p g_repel), unless an explicit @p rules entry overrides the pair. Bodies
    with no cluster (id -1) don't interact (returns 0).

    @param cluster_a Cluster id of the first body (-1 = unassigned).
    @param cluster_b Cluster id of the second body (-1 = unassigned).
    @param g_attract Default same-cluster coefficient (negative attracts).
    @param g_repel Default cross-cluster coefficient (positive repels).
    @param rules Optional dict (min(a, b), max(a, b)) -> coefficient override.
    @return float coefficient g (0.0 when either body is unassigned).
    """
    if cluster_a == -1 or cluster_b == -1:
        return 0.0
    if rules is not None:
        key = (min(cluster_a, cluster_b), max(cluster_a, cluster_b))
        default = g_attract if cluster_a == cluster_b else g_repel
        return rules.get(key, default)
    if cluster_a == cluster_b:
        return g_attract
    return g_repel


def pair_force(distance, coefficient, count, lower_range, power=1,
               attraction_enabled=True, emergency_strength=EMERGENCY_REPULSION_STRENGTH):
    """@brief Full pairwise force scalar: emergency repulsion + standard gravity.

    Emergency anti-stacking repulsion dominates when @p distance is below
    @p lower_range * 2 (always on, even with attraction disabled). Otherwise,
    when @p attraction_enabled, the standard gravity force applies with
    @p coefficient; with attraction off and outside the emergency range, the
    force is zero.

    @param distance Center-to-center distance between the two bodies.
    @param coefficient Attract/repel coefficient g (see interaction_coefficient()).
    @param count Number of interacting bodies (normalizes the gravity force).
    @param lower_range Emergency repulsion triggers below 2 * this distance.
    @param power Exponent on distance for the gravity force.
    @param attraction_enabled When False, only emergency repulsion can act.
    @param emergency_strength Magnitude of the emergency repulsion.
    @return float force scalar (multiply by the dx/dy delta to get the vector).
    """
    if distance < lower_range * 2:
        return emergency_repulsion(emergency_strength)
    if attraction_enabled:
        return gravity_force(coefficient, distance, count, power)
    return 0.0


def soft_obstacle_push(distance, edge, soft_zone, soft_strength):
    """@brief Quadratic pre-contact obstacle repulsion magnitude.

    Zero outside the soft zone; rises quadratically from the zone's outer edge to
    full @p soft_strength at the physical edge. Multiply by the outward unit
    vector (dx/dist, dy/dist) to get the push vector.

    @param distance Body-to-obstacle-center distance.
    @param edge Obstacle radius + body radius (the physical contact distance).
    @param soft_zone Width (px) of the pre-contact zone beyond @p edge.
    @param soft_strength Push magnitude at the physical edge.
    @return float push magnitude (0.0 outside the soft zone).
    """
    if soft_zone <= 0 or distance >= edge + soft_zone:
        return 0.0
    t = max(0.0, (edge + soft_zone - distance) / soft_zone)
    return soft_strength * t * t


def reflect_velocity(vx, vy, nx, ny, restitution=1.0):
    """@brief Reflect a velocity off a surface with unit normal (nx, ny).

    Only reflects when the body is moving into the surface (negative normal
    component); otherwise the velocity is returned unchanged. @p restitution
    scales the reflected speed (1.0 = elastic, < 1 = energy loss).

    @param vx, vy Current velocity components.
    @param nx, ny Unit surface normal pointing away from the surface.
    @param restitution Bounce energy retention in [0, 1].
    @return (vx, vy) the (possibly) reflected velocity.
    """
    dot = vx * nx + vy * ny
    if dot < 0:
        vx -= 2 * dot * nx
        vy -= 2 * dot * ny
        vx *= restitution
        vy *= restitution
    return vx, vy


def clamp_speed(vx, vy, max_speed):
    """@brief Cap a velocity vector's magnitude at @p max_speed (direction kept).

    @param vx, vy Current velocity components.
    @param max_speed Maximum allowed speed.
    @return (vx, vy) the speed-limited velocity.
    """
    speed = (vx ** 2 + vy ** 2) ** 0.5
    if speed > max_speed:
        scale = max_speed / speed
        vx *= scale
        vy *= scale
    return vx, vy

"""
Scene Graph Module: Dynamic spatio-temporal scene graph construction.

Constructs graph G_t = (V_t, E_t) from detected traffic participants,
with kinematic-annotated nodes and proximity-based edges.
"""

from typing import List, Optional, Tuple

import numpy as np

from pc_csg.config import PCCSGConfig
from pc_csg.types import KinematicState, ObjectClass, SceneGraph, SceneNode


def compute_ttc(
    ego_x: float, ego_y: float, ego_vx: float, ego_vy: float,
    obj_x: float, obj_y: float, obj_vx: float, obj_vy: float,
    collision_radius: float = 2.5,
) -> float:
    """
    Compute Time-to-Collision between ego vehicle and a scene participant.

    Uses Closest Point of Approach (CPA) kinematics:
    If relative trajectory closest approach distance exceeds the collision corridor,
    vehicles safely clear each other and TTC is infinite.
    """
    dx = obj_x - ego_x
    dy = obj_y - ego_y
    dvx = obj_vx - ego_vx
    dvy = obj_vy - ego_vy

    dist = np.sqrt(dx**2 + dy**2)
    if dist < 1e-6:
        return 0.0

    v_rel_sq = dvx**2 + dvy**2
    if v_rel_sq < 1e-6:
        return float("inf")

    t_cpa = -(dx * dvx + dy * dvy) / v_rel_sq
    if t_cpa <= 0:
        return float("inf")  # Diverging — no collision

    # Distance at closest point of approach
    dx_cpa = dx + dvx * t_cpa
    dy_cpa = dy + dvy * t_cpa
    d_cpa = np.sqrt(dx_cpa**2 + dy_cpa**2)
    if d_cpa > collision_radius:
        return float("inf")  # Safe lateral passing corridor

    return float(t_cpa)


def compute_hazard_weight(
    ttc: float,
    tau_crit: float = 3.0,
    lambda_scale: float = 0.5,
) -> float:
    """
    Compute intrinsic hazard score H(v_i) ∈ [0, 1] via sigmoid TTC mapping.

    H(v_i) = σ((τ_crit - TTC) / λ_scale) · 𝟙(TTC > 0)
    """
    if ttc <= 0 or ttc == float("inf"):
        return 0.0
    exponent = (ttc - tau_crit) / lambda_scale
    return float(1.0 / (1.0 + np.exp(exponent)))


def build_edge_features(
    node_i: SceneNode, node_j: SceneNode,
) -> np.ndarray:
    """
    Compute 6-D edge feature: [rel_x, rel_y, rel_vx, rel_vy, dist, bearing].
    """
    dx = node_j.kinematics.x - node_i.kinematics.x
    dy = node_j.kinematics.y - node_i.kinematics.y
    dvx = node_j.kinematics.vx - node_i.kinematics.vx
    dvy = node_j.kinematics.vy - node_i.kinematics.vy
    dist = np.sqrt(dx**2 + dy**2)
    bearing = np.arctan2(dy, dx)
    return np.array([dx, dy, dvx, dvy, dist, bearing], dtype=np.float32)


def build_scene_graph(
    detections: List[dict],
    ego_state: Optional[KinematicState] = None,
    config: Optional[PCCSGConfig] = None,
    timestamp: float = 0.0,
) -> SceneGraph:
    """
    Construct a dynamic scene graph from raw detections.

    Args:
        detections: List of dicts with keys:
            track_id, class_id, bbox (cx,cy,w,h), vx, vy, ax, ay, heading
        ego_state: Kinematic state of the ego vehicle.
        config: Framework configuration.
        timestamp: Frame timestamp.

    Returns:
        SceneGraph with kinematic-annotated nodes and proximity edges.
    """
    if config is None:
        config = PCCSGConfig()

    if ego_state is None:
        ego_state = KinematicState(x=0.0, y=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0, heading=0.0)

    nodes: List[SceneNode] = []

    for det in detections[:config.max_nodes]:
        kin = KinematicState(
            x=det.get("x", det.get("bbox", [0, 0, 0, 0])[0]),
            y=det.get("y", det.get("bbox", [0, 0, 0, 0])[1]),
            vx=det.get("vx", 0.0),
            vy=det.get("vy", 0.0),
            ax=det.get("ax", 0.0),
            ay=det.get("ay", 0.0),
            heading=det.get("heading", 0.0),
        )

        ttc = compute_ttc(
            ego_state.x, ego_state.y, ego_state.vx, ego_state.vy,
            kin.x, kin.y, kin.vx, kin.vy,
        )
        hazard = compute_hazard_weight(ttc, config.tau_crit, config.lambda_scale)

        node = SceneNode(
            track_id=det.get("track_id", len(nodes)),
            class_id=ObjectClass(det.get("class_id", ObjectClass.UNKNOWN)),
            bbox=tuple(det.get("bbox", (0.0, 0.0, 0.0, 0.0))),
            kinematics=kin,
            ttc=ttc,
            hazard_weight=hazard,
        )
        nodes.append(node)

    # Build proximity-based edges
    src_list: List[int] = []
    dst_list: List[int] = []
    edge_attrs: List[np.ndarray] = []

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue
            dx = nodes[j].kinematics.x - nodes[i].kinematics.x
            dy = nodes[j].kinematics.y - nodes[i].kinematics.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= config.proximity_threshold:
                src_list.append(i)
                dst_list.append(j)
                edge_attrs.append(build_edge_features(nodes[i], nodes[j]))

    if len(src_list) > 0:
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr = np.stack(edge_attrs, axis=0)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, config.edge_feature_dim), dtype=np.float32)

    return SceneGraph(
        nodes=nodes,
        edge_index=edge_index,
        edge_attr=edge_attr,
        timestamp=timestamp,
    )

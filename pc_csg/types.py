"""
Core data contracts and type definitions for the PC-CSG framework.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


class InterventionLevel(IntEnum):
    """Graduated preemptive safety intervention levels."""
    NOMINAL = 0                   # CRT < 0.25: Safe nominal driving
    ADVISORY = 1                  # CRT ∈ [0.25, 0.50): HUD advisory overlay
    ALERT = 2                     # CRT ∈ [0.50, 0.70): Audible + trajectory display
    PRE_INTERVENTION = 3          # CRT ∈ [0.70, 0.85): Pre-braking + lane centering
    EMERGENCY = 4                 # CRT ≥ 0.85: Autonomous AEB + MRM


class ObjectClass(IntEnum):
    """Semantic object categories for scene graph nodes."""
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    TRUCK = 3
    BUS = 4
    BARRIER = 5
    CONE = 6
    UNKNOWN = 7


class CounterfactualMode(IntEnum):
    """Counterfactual trajectory perturbation modes."""
    SUDDEN_BRAKE = 0
    HARD_SWERVE_LEFT = 1
    HARD_SWERVE_RIGHT = 2
    SUDDEN_ACCELERATION = 3
    LANE_DEPARTURE = 4
    STATIONARY_OBSTACLE = 5


@dataclass
class KinematicState:
    """Full kinematic state vector for a traffic participant."""
    x: float                      # Global X position (meters)
    y: float                      # Global Y position (meters)
    vx: float                     # X velocity (m/s)
    vy: float                     # Y velocity (m/s)
    ax: float                     # X acceleration (m/s²)
    ay: float                     # Y acceleration (m/s²)
    heading: float                # Heading angle (radians)
    yaw_rate: float = 0.0         # Yaw rate (rad/s)
    speed: float = 0.0            # Scalar speed (m/s)

    def __post_init__(self) -> None:
        self.speed = float(np.sqrt(self.vx**2 + self.vy**2))


@dataclass
class SceneNode:
    """A single traffic participant node in the scene graph."""
    track_id: int
    class_id: ObjectClass
    bbox: Tuple[float, float, float, float]   # (cx, cy, w, h) normalized [0,1]
    kinematics: KinematicState
    ttc: float                                # Time-to-Collision (seconds, inf if safe)
    hazard_weight: float = 0.0                # H(v_i) ∈ [0, 1] computed by hazard scorer

    def to_feature_vector(self) -> np.ndarray:
        """Convert to 12-D feature vector for neural input."""
        return np.array([
            float(self.class_id),
            self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
            self.kinematics.vx, self.kinematics.vy,
            self.kinematics.ax, self.kinematics.ay,
            self.kinematics.heading,
            min(self.ttc, 30.0),              # Cap TTC at 30s
            self.kinematics.speed,
        ], dtype=np.float32)


@dataclass
class SceneGraph:
    """Dynamic scene graph at a single time step."""
    nodes: List[SceneNode]
    edge_index: np.ndarray                    # Shape (2, E) — adjacency pairs
    edge_attr: np.ndarray                     # Shape (E, D_edge) — edge features
    timestamp: float = 0.0                    # Frame timestamp

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def to_node_features(self) -> np.ndarray:
        """Stack all node feature vectors → (N, 12)."""
        if len(self.nodes) == 0:
            return np.zeros((0, 12), dtype=np.float32)
        return np.stack([n.to_feature_vector() for n in self.nodes], axis=0)


@dataclass
class CounterfactualTrajectory:
    """A single counterfactual trajectory hypothesis."""
    agent_id: int                             # Which agent is perturbed
    mode: CounterfactualMode                  # Type of perturbation
    trajectory: np.ndarray                    # Shape (H, 2) — predicted (x, y) positions
    is_physically_valid: bool = True          # Passes PIKV kinematic constraints
    collision_risk: float = 0.0               # Risk score ∈ [0, 1]
    ttc_under_cf: float = float("inf")        # TTC under this counterfactual scenario


@dataclass
class CounterfactualRiskTensor:
    """The Counterfactual Risk Tensor (CRT) — novel metric."""
    crt_value: float                          # Aggregated scalar CRT ∈ [0, 1]
    per_agent_risk: np.ndarray                # Risk per agent: (N,)
    per_cf_risk: np.ndarray                   # Risk per counterfactual: (N, K)
    critical_agent_id: Optional[int] = None   # Most dangerous agent
    critical_cf_mode: Optional[CounterfactualMode] = None
    worst_case_ttc: float = float("inf")


@dataclass
class InterventionDecision:
    """Final preemptive safety intervention decision."""
    level: InterventionLevel
    crt: CounterfactualRiskTensor
    action_description: str
    preemptive_lead_time: float = 0.0         # Time before predicted collision (seconds)
    target_agent_id: Optional[int] = None     # Agent targeted for overlay/warning

"""
Core data contracts and types for ST-HGST framework.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple
import numpy as np

class TakeoverAction(IntEnum):
    SAFE_HANDOVER = 0
    TARGETED_SPATIAL_CUE = 1
    MINIMUM_RISK_MANEUVER = 2

class FixationState(IntEnum):
    UNSEEN = 0
    SACCADIC_GLANCE = 1
    COMPREHENDED = 2

@dataclass
class SceneNode:
    track_id: int
    class_id: int
    bbox: Tuple[float, float, float, float]  # (x, y, w, h) normalized
    velocity: Tuple[float, float]             # (vx, vy)
    ttc: float                               # Time-To-Collision (seconds)
    hazard_weight: float                     # H(v_i) in [0, 1]

@dataclass
class SceneGraph:
    nodes: List[SceneNode]
    edge_index: np.ndarray                   # Shape (2, E)
    edge_attr: np.ndarray                    # Shape (E, D_edge)

@dataclass
class GazeVector:
    unit_vector: np.ndarray                  # 3D vector (x, y, z) on unit sphere
    pitch: float
    yaw: float
    windshield_point: Optional[Tuple[float, float]] = None  # (u_g, v_g) in [0, 1]

    @classmethod
    def from_raw(cls, raw: np.ndarray, pitch: float = 0.0, yaw: float = 0.0):
        norm = float(np.linalg.norm(raw))
        unit = raw / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return cls(unit_vector=unit.astype(np.float32), pitch=float(pitch), yaw=float(yaw))

@dataclass
class CognitiveState:
    attention_weights: np.ndarray            # alpha_i per node
    accumulated_cognition: np.ndarray        # C_i per node
    fixation_states: List[FixationState]     # state per node

@dataclass
class ArbitrationDecision:
    action: TakeoverAction
    eag: float
    causal_hazard_id: Optional[int]
    action_description: str

"""
Global configuration dataclass for PC-CSG framework.

Physics-Constrained Counterfactual Scene Graph Transformer
for Real-Time Autonomous Anomaly Anticipation.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class PCCSGConfig:
    """Master configuration for the PC-CSG framework."""

    # ── Temporal Sequence ──────────────────────────────────────────────
    sequence_length: int = 30           # T = 30 frames of scene graph history
    fps: float = 30.0                   # Camera frame rate
    delta_t: float = 1.0 / 30.0        # Time step (seconds)
    prediction_horizon: int = 15        # Future prediction horizon (frames = 0.5s)

    # ── Scene Graph ────────────────────────────────────────────────────
    max_nodes: int = 32                 # Maximum traffic participants per frame
    node_feature_dim: int = 12          # Per-node: [class, x, y, w, h, vx, vy, ax, ay, heading, ttc, dist]
    edge_feature_dim: int = 6           # Per-edge: [rel_x, rel_y, rel_vx, rel_vy, dist, bearing]
    num_object_classes: int = 8         # Vehicle, Pedestrian, Cyclist, Truck, Bus, Barrier, Cone, Unknown
    proximity_threshold: float = 50.0   # Scene graph edge connection radius (meters)

    # ── Backbone (EfficientDet-Lite or YOLOv8-Nano) ────────────────────
    backbone_out_dim: int = 128         # Feature dimension from visual backbone
    backbone_type: str = "yolov8n"      # Detection backbone

    # ── Counterfactual Scene Graph Attention (CSGA) ────────────────────
    d_model: int = 128                  # Transformer model dimension
    num_heads: int = 4                  # Multi-head attention heads
    num_encoder_layers: int = 3         # Spatio-temporal GNN encoder layers
    num_decoder_layers: int = 2         # Counterfactual decoder layers
    d_ff: int = 512                     # Feed-forward hidden dimension
    dropout: float = 0.1               # Attention dropout

    # ── Counterfactual Hypothesis Generation ───────────────────────────
    num_counterfactuals: int = 6        # K counterfactual hypotheses per critical agent
    cf_perturbation_modes: List[str] = field(default_factory=lambda: [
        "sudden_brake",                 # Delta_v = -v (full stop)
        "hard_swerve_left",             # Delta_heading = +30°
        "hard_swerve_right",            # Delta_heading = -30°
        "sudden_acceleration",          # Delta_v = +5 m/s
        "lane_departure",               # Lateral drift 2.0m
        "stationary_obstacle",          # v → 0 instantaneously
    ])

    # ── Physics-Informed Kinematic Validator (PIKV) ────────────────────
    # Kinematic Bicycle Model constraints
    wheelbase: float = 2.7             # Average passenger car wheelbase (meters)
    max_steering_angle: float = 0.6    # Maximum steering angle (radians, ~34°)
    max_acceleration: float = 8.0      # Maximum longitudinal acceleration (m/s²)
    max_deceleration: float = 10.0     # Maximum braking deceleration (m/s²)
    max_lateral_accel: float = 6.0     # Maximum lateral acceleration (m/s²)
    max_yaw_rate: float = 1.2          # Maximum yaw rate (rad/s)
    friction_coefficient: float = 0.7  # Tire-road friction coefficient (dry asphalt)
    gravity: float = 9.81             # Gravitational acceleration (m/s²)

    # ── Counterfactual Risk Tensor (CRT) ───────────────────────────────
    tau_crit: float = 3.0              # Critical TTC threshold (seconds)
    lambda_scale: float = 0.5          # Sigmoid scaling for hazard scores
    risk_decay: float = 0.9            # Temporal risk decay factor
    crt_aggregation: str = "max"       # How to aggregate CRT: "max", "mean", "weighted"

    # ── Safety Intervention Thresholds ─────────────────────────────────
    crt_safe: float = 0.25            # CRT below = nominal driving
    crt_caution: float = 0.50         # CRT caution = advisory warning
    crt_warning: float = 0.70         # CRT warning = active alert
    crt_critical: float = 0.85        # CRT critical = preemptive intervention

    # ── Graduated Intervention Actions ─────────────────────────────────
    # Level 0: CRT < 0.25 → Safe nominal operation
    # Level 1: 0.25 ≤ CRT < 0.50 → HUD advisory overlay
    # Level 2: 0.50 ≤ CRT < 0.70 → Audible alert + predictive trajectory display
    # Level 3: 0.70 ≤ CRT < 0.85 → Active pre-braking + lane centering ready
    # Level 4: CRT ≥ 0.85 → Autonomous Emergency Intervention (AEB + MRM)

    # ── Training ───────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 16
    num_epochs: int = 100
    warmup_epochs: int = 5
    focal_gamma: float = 2.0           # Focal loss focusing parameter

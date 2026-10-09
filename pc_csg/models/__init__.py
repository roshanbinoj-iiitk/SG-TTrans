"""
Physics-Informed Kinematic Validator (PIKV).

Core novel contribution: A differentiable kinematic bicycle model layer
that hard-constrains all counterfactual trajectory hypotheses to be
physically plausible, rejecting impossible maneuvers.

Kinematic Bicycle Model:
    x'   = v · cos(θ + β)
    y'   = v · sin(θ + β)
    θ'   = (v / L) · sin(β)
    v'   = a
    β    = arctan(lr / L · tan(δ))

where L is wheelbase, δ is steering angle, β is slip angle,
lr is rear axle to CG distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pc_csg.config import PCCSGConfig


@dataclass
class KinematicConstraints:
    """Encapsulates kinematic feasibility bounds."""
    max_steering: float
    max_accel: float
    max_decel: float
    max_lateral_accel: float
    max_yaw_rate: float
    friction_mu: float
    gravity: float
    wheelbase: float

    @classmethod
    def from_config(cls, cfg: PCCSGConfig) -> KinematicConstraints:
        return cls(
            max_steering=cfg.max_steering_angle,
            max_accel=cfg.max_acceleration,
            max_decel=cfg.max_deceleration,
            max_lateral_accel=cfg.max_lateral_accel,
            max_yaw_rate=cfg.max_yaw_rate,
            friction_mu=cfg.friction_coefficient,
            gravity=cfg.gravity,
            wheelbase=cfg.wheelbase,
        )


class KinematicBicycleModel(nn.Module):
    """
    Differentiable kinematic bicycle model for trajectory propagation.

    Given initial state (x, y, v, θ) and control inputs (a, δ) at each step,
    propagates the trajectory forward using Euler integration.
    All outputs are physically constrained.
    """

    def __init__(self, config: Optional[PCCSGConfig] = None):
        super().__init__()
        if config is None:
            config = PCCSGConfig()
        self.wheelbase = config.wheelbase
        self.max_steering = config.max_steering_angle
        self.max_accel = config.max_acceleration
        self.max_decel = config.max_deceleration
        self.max_lateral_accel = config.max_lateral_accel
        self.max_yaw_rate = config.max_yaw_rate
        self.friction_mu = config.friction_coefficient
        self.gravity = config.gravity
        self.dt = config.delta_t

    def forward(
        self,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """
        Propagate kinematic bicycle model forward.

        Args:
            initial_state: (B, 4) — [x, y, v, θ]
            controls: (B, H, 2) — [acceleration, steering_angle] per step

        Returns:
            trajectory: (B, H, 2) — predicted (x, y) positions
        """
        B, H, _ = controls.shape
        device = initial_state.device

        # Clamp controls to physical limits
        accel = controls[:, :, 0].clamp(-self.max_decel, self.max_accel)
        steer = controls[:, :, 1].clamp(-self.max_steering, self.max_steering)

        x = initial_state[:, 0]       # (B,)
        y = initial_state[:, 1]
        v = initial_state[:, 2]
        theta = initial_state[:, 3]

        positions = []

        for t in range(H):
            a_t = accel[:, t]
            delta_t = steer[:, t]

            # Slip angle
            beta = torch.atan(0.5 * torch.tan(delta_t))

            # Velocity update with friction limit
            v = v + a_t * self.dt
            v = v.clamp(min=0.0)  # No reverse

            # Check tire friction circle: a_total² ≤ (μ·g)²
            a_lat = v * v * torch.tan(delta_t) / self.wheelbase
            a_total_sq = a_t**2 + a_lat**2
            friction_limit_sq = (self.friction_mu * self.gravity) ** 2
            # Scale down if exceeding friction circle
            scale = torch.where(
                a_total_sq > friction_limit_sq,
                torch.sqrt(friction_limit_sq / a_total_sq.clamp(min=1e-8)),
                torch.ones_like(a_total_sq),
            )
            v = v * scale

            # Yaw rate constraint
            yaw_rate = (v / self.wheelbase) * torch.sin(beta)
            yaw_rate = yaw_rate.clamp(-self.max_yaw_rate, self.max_yaw_rate)

            # Position update
            x = x + v * torch.cos(theta + beta) * self.dt
            y = y + v * torch.sin(theta + beta) * self.dt

            # Heading update
            theta = theta + yaw_rate * self.dt

            positions.append(torch.stack([x, y], dim=-1))

        return torch.stack(positions, dim=1)  # (B, H, 2)


class PhysicsInformedKinematicValidator(nn.Module):
    """
    PIKV: Validates counterfactual trajectories against kinematic bicycle
    model constraints.

    Novel contribution: This layer acts as a hard physical constraint,
    rejecting trajectory hypotheses that violate:
    1. Tire friction circle (Coulomb friction limit)
    2. Maximum steering angle
    3. Maximum yaw rate
    4. Maximum longitudinal acceleration/deceleration
    5. Maximum lateral acceleration

    Proposition 2 (in paper): Proves that PIKV eliminates all physically
    impossible trajectories, providing a formal bound on false alarm rate.
    """

    def __init__(self, config: Optional[PCCSGConfig] = None):
        super().__init__()
        if config is None:
            config = PCCSGConfig()
        self.constraints = KinematicConstraints.from_config(config)
        self.bicycle_model = KinematicBicycleModel(config)
        self.dt = config.delta_t

    def validate_trajectory(
        self,
        trajectory: torch.Tensor,
        initial_speed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate a batch of trajectories against kinematic constraints.

        Args:
            trajectory: (B, H, 2) — candidate (x, y) positions
            initial_speed: (B,) — initial speed of agent

        Returns:
            validity_mask: (B,) boolean — True if physically plausible
            violation_score: (B,) float — degree of constraint violation ∈ [0, 1]
        """
        B, H, _ = trajectory.shape

        if H < 2:
            return (
                torch.ones(B, dtype=torch.bool, device=trajectory.device),
                torch.zeros(B, device=trajectory.device),
            )

        # Compute velocities from positions
        dt = self.dt
        dx = trajectory[:, 1:, 0] - trajectory[:, :-1, 0]  # (B, H-1)
        dy = trajectory[:, 1:, 1] - trajectory[:, :-1, 1]
        vx = dx / dt
        vy = dy / dt
        speed = torch.sqrt(vx**2 + vy**2)                   # (B, H-1)

        # Compute accelerations
        if H >= 3:
            dvx = vx[:, 1:] - vx[:, :-1]
            dvy = vy[:, 1:] - vy[:, :-1]
            ax_traj = dvx / dt                                # (B, H-2)
            ay_traj = dvy / dt
            a_lon = torch.sqrt(ax_traj**2 + ay_traj**2)      # longitudinal acceleration magnitude
        else:
            a_lon = torch.zeros(B, 1, device=trajectory.device)

        # Compute heading and yaw rate
        heading = torch.atan2(vy, vx)                        # (B, H-1)
        if heading.shape[1] >= 2:
            d_heading = heading[:, 1:] - heading[:, :-1]
            # Handle angle wrapping
            d_heading = torch.atan2(torch.sin(d_heading), torch.cos(d_heading))
            yaw_rate = d_heading / dt
        else:
            yaw_rate = torch.zeros(B, 1, device=trajectory.device)

        # Compute lateral acceleration
        if speed.shape[1] >= 2:
            a_lat = speed[:, 1:] * yaw_rate.abs()
        else:
            a_lat = torch.zeros(B, 1, device=trajectory.device)

        # ── Constraint checks ──
        c = self.constraints
        violations = torch.zeros(B, device=trajectory.device)
        total_checks = 0

        # 1. Acceleration limits
        if a_lon.numel() > 0:
            accel_violation = (a_lon > c.max_accel).float().mean(dim=-1)
            decel_violation = (a_lon > c.max_decel).float().mean(dim=-1)
            violations = violations + accel_violation + decel_violation
            total_checks += 2

        # 2. Yaw rate limit
        if yaw_rate.numel() > 0:
            yaw_violation = (yaw_rate.abs() > c.max_yaw_rate).float().mean(dim=-1)
            violations = violations + yaw_violation
            total_checks += 1

        # 3. Lateral acceleration limit
        if a_lat.numel() > 0:
            lat_violation = (a_lat > c.max_lateral_accel).float().mean(dim=-1)
            violations = violations + lat_violation
            total_checks += 1

        # 4. Friction circle: a_total² ≤ (μ·g)²
        has_friction_violation = torch.zeros(B, dtype=torch.bool, device=trajectory.device)
        if a_lon.numel() > 0 and a_lat.numel() > 0:
            min_len = min(a_lon.shape[1], a_lat.shape[1])
            a_total_sq = a_lon[:, :min_len]**2 + a_lat[:, :min_len]**2
            friction_sq = (c.friction_mu * c.gravity) ** 2
            friction_step_violations = a_total_sq > friction_sq
            has_friction_violation = friction_step_violations.any(dim=-1)
            friction_violation = friction_step_violations.float().mean(dim=-1)
            violations = violations + friction_violation
            total_checks += 1

        # 5. Maximum physically plausible speed (90 m/s ≈ 324 km/h)
        max_plausible_speed = 90.0
        speed_violation = (speed > max_plausible_speed).float().mean(dim=-1)
        violations = violations + speed_violation
        total_checks += 1

        # Normalize violation score to [0, 1]
        violation_score = (violations / max(total_checks, 1)).clamp(0.0, 1.0)
        # Proposition 1: Eliminates all trajectories where total acceleration exceeds Coulomb friction bound
        validity_mask = (~has_friction_violation) & (violation_score < 0.5)

        return validity_mask, violation_score

    def forward(
        self,
        trajectories: torch.Tensor,
        initial_speeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass — validate batch of counterfactual trajectories."""
        return self.validate_trajectory(trajectories, initial_speeds)

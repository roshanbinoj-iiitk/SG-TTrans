"""
PC-CSG End-to-End Network.

Physics-Constrained Counterfactual Scene Graph Transformer — unified
network that chains: Scene Graph → CSGA → PIKV → CRT.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from pc_csg.config import PCCSGConfig
from pc_csg.models.csga import CounterfactualSceneGraphAttention
from pc_csg.models import PhysicsInformedKinematicValidator, KinematicBicycleModel


class PCCSGNetwork(nn.Module):
    """
    End-to-end PC-CSG network.

    Pipeline:
        1. Node/Edge projection → Scene understanding
        2. CSGA: Causal gating + counterfactual query injection
        3. PIKV: Physics-constrained trajectory validation
        4. CRT computation → Intervention decision

    This is the main model that gets trained and deployed.
    """

    def __init__(self, config: Optional[PCCSGConfig] = None):
        super().__init__()
        if config is None:
            config = PCCSGConfig()
        self.config = config

        # ── Core Modules ──
        self.csga = CounterfactualSceneGraphAttention(
            node_dim=config.node_feature_dim,
            edge_dim=config.edge_feature_dim,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_counterfactuals=config.num_counterfactuals,
            dropout=config.dropout,
        )

        self.pikv = PhysicsInformedKinematicValidator(config)
        self.bicycle = KinematicBicycleModel(config)

        # ── Counterfactual trajectory decoder ──
        # Maps counterfactual embeddings to control inputs (a, δ) per step
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.prediction_horizon * 2),
        )

        # Mode-specific control priors (K, H, 2): [accel, steer]
        K = config.num_counterfactuals
        H = config.prediction_horizon
        priors = torch.zeros(K, H, 2)
        if K >= 6:
            # Mode 0: Sudden Brake (negative longitudinal acceleration)
            priors[0, :, 0] = -5.0
            # Mode 1: Hard Swerve Left (sharp steering left)
            priors[1, :, 1] = 0.45
            # Mode 2: Hard Swerve Right (sharp steering right)
            priors[2, :, 1] = -0.45
            # Mode 3: Sudden Acceleration (positive longitudinal acceleration)
            priors[3, :, 0] = 4.0
            # Mode 4: Lane Departure (gradual steering drift)
            priors[4, :, 1] = 0.15
            # Mode 5: Stationary / Emergency Stop (heavy deceleration to stop)
            priors[5, :, 0] = -7.5
        self.register_buffer("mode_control_priors", priors)

        # ── Initial state extractor ──
        # Maps node features to initial kinematic state (x, y, v, θ)
        self.state_extractor = nn.Sequential(
            nn.Linear(config.node_feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 4),  # (x, y, v, θ)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        hazard_weights: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
        raw_node_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            node_features: (B, N, D_node) — raw scene graph node features
            edge_index: (2, E) — graph connectivity
            edge_attr: (B, E, D_edge) — edge features
            hazard_weights: (B, N) — hazard scores
            node_mask: (B, N) — valid node mask
            raw_node_features: (B, N, D_raw) — original node features for state extraction

        Returns:
            dict with all intermediate and final outputs
        """
        B, N, _ = node_features.shape
        K = self.config.num_counterfactuals
        H = self.config.prediction_horizon
        device = node_features.device

        if raw_node_features is None:
            raw_node_features = node_features

        # ── Step 1: CSGA ──
        csga_out = self.csga(
            node_features, edge_index, edge_attr,
            hazard_weights, node_mask,
        )
        scene_embeds = csga_out["scene_embeddings"]     # (B, N, D)
        cf_risk_logits = csga_out["cf_risk_logits"]     # (B, N, K)
        critical_mask = csga_out["critical_mask"]        # (B, N)

        # ── Step 2: Decode counterfactual trajectories ──
        # For each critical agent × counterfactual mode, decode trajectory from CF embeddings
        cf_embeds = csga_out.get("cf_embeddings")
        if cf_embeds is None:
            cf_embeds = scene_embeds.unsqueeze(2).expand(B, N, K, self.config.d_model)
        cf_flat = cf_embeds.reshape(B * N * K, self.config.d_model)

        controls = self.trajectory_decoder(cf_flat)      # (BNK, H*2)
        controls = controls.view(B, N, K, H, 2)
        controls = controls + self.mode_control_priors.unsqueeze(0).unsqueeze(0)
        controls = controls.view(B * N * K, H, 2)

        # Extract initial states
        init_states = self.state_extractor(raw_node_features)  # (B, N, 4)
        init_states_exp = init_states.unsqueeze(2).expand(B, N, K, 4)
        init_states_flat = init_states_exp.reshape(B * N * K, 4)

        # ── Step 3: Propagate through bicycle model ──
        trajectories = self.bicycle(init_states_flat, controls)  # (BNK, H, 2)

        # ── Step 4: PIKV validation ──
        initial_speeds = init_states_flat[:, 2].abs()           # (BNK,)
        pikv_validity, violation_scores = self.pikv(trajectories, initial_speeds)

        # Reshape back
        pikv_validity = pikv_validity.view(B, N, K)
        violation_scores = violation_scores.view(B, N, K)
        trajectories = trajectories.view(B, N, K, H, 2)

        # ── Step 5: Filtered CRT computation ──
        # CRT = max over agents × modes of (risk × validity × hazard)
        filtered_risk = cf_risk_logits * pikv_validity.float() * hazard_weights.unsqueeze(2)

        if node_mask is not None:
            filtered_risk = filtered_risk * node_mask.unsqueeze(2).float()

        per_agent_risk, _ = filtered_risk.max(dim=2)     # (B, N)
        crt_scalar, _ = per_agent_risk.max(dim=1)         # (B,)
        crt_scalar = crt_scalar.clamp(0.0, 1.0)

        return {
            "crt_scalar": crt_scalar,
            "cf_risk_logits": cf_risk_logits,
            "pikv_validity": pikv_validity,
            "violation_scores": violation_scores,
            "per_agent_risk": per_agent_risk,
            "trajectories": trajectories,
            "scene_embeddings": scene_embeds,
            "critical_mask": critical_mask,
            "causal_gates": csga_out["causal_gates"],
        }

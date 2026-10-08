"""
Comprehensive test suite for PC-CSG framework.

Tests all core components: config, types, scene graph, CSGA, PIKV, CRT,
end-to-end network, and synthetic data generation.
"""

import numpy as np
import pytest
import torch

from pc_csg.config import PCCSGConfig
from pc_csg.types import (
    CounterfactualMode,
    CounterfactualRiskTensor,
    InterventionDecision,
    InterventionLevel,
    KinematicState,
    ObjectClass,
    SceneGraph,
    SceneNode,
)
from pc_csg.scene_graph import (
    build_scene_graph,
    compute_hazard_weight,
    compute_ttc,
)
from pc_csg.models import (
    KinematicBicycleModel,
    PhysicsInformedKinematicValidator,
)
from pc_csg.models.csga import (
    CausalAttentionGating,
    CounterfactualQueryInjector,
    CounterfactualSceneGraphAttention,
)
from pc_csg.models.pc_csg_net import PCCSGNetwork
from pc_csg.risk_engine import CRTEngine, PreemptiveInterventionArbiter
from pc_csg.data import PCCSGDataset, SyntheticTrafficScenarioGenerator


# ═══════════════════════════════════════════════════════════
#  Configuration Tests
# ═══════════════════════════════════════════════════════════

class TestConfig:
    def test_default_config_creation(self):
        cfg = PCCSGConfig()
        assert cfg.d_model == 128
        assert cfg.num_counterfactuals == 6
        assert cfg.max_nodes == 32
        assert cfg.prediction_horizon == 15

    def test_physics_constraints_valid(self):
        cfg = PCCSGConfig()
        assert cfg.max_acceleration > 0
        assert cfg.max_deceleration > 0
        assert cfg.friction_coefficient > 0
        assert cfg.wheelbase > 0

    def test_intervention_thresholds_ordered(self):
        cfg = PCCSGConfig()
        assert cfg.crt_safe < cfg.crt_caution < cfg.crt_warning < cfg.crt_critical


# ═══════════════════════════════════════════════════════════
#  Type Tests
# ═══════════════════════════════════════════════════════════

class TestTypes:
    def test_kinematic_state_speed(self):
        ks = KinematicState(x=0, y=0, vx=3.0, vy=4.0, ax=0, ay=0, heading=0)
        assert abs(ks.speed - 5.0) < 1e-6

    def test_scene_node_feature_vector(self):
        kin = KinematicState(x=10, y=5, vx=2, vy=1, ax=0.5, ay=0, heading=0.3)
        node = SceneNode(
            track_id=0, class_id=ObjectClass.VEHICLE,
            bbox=(0.5, 0.5, 0.1, 0.1), kinematics=kin,
            ttc=3.0, hazard_weight=0.6,
        )
        fv = node.to_feature_vector()
        assert fv.shape == (12,)
        assert fv[0] == 0.0  # VEHICLE class

    def test_intervention_levels(self):
        assert InterventionLevel.NOMINAL < InterventionLevel.ADVISORY
        assert InterventionLevel.ADVISORY < InterventionLevel.ALERT
        assert InterventionLevel.ALERT < InterventionLevel.PRE_INTERVENTION
        assert InterventionLevel.PRE_INTERVENTION < InterventionLevel.EMERGENCY

    def test_counterfactual_modes(self):
        assert len(CounterfactualMode) == 6
        assert CounterfactualMode.SUDDEN_BRAKE == 0


# ═══════════════════════════════════════════════════════════
#  Scene Graph Tests
# ═══════════════════════════════════════════════════════════

class TestSceneGraph:
    def test_ttc_closing(self):
        """Two vehicles approaching — finite TTC."""
        ttc = compute_ttc(0, 0, 10, 0, 50, 0, -5, 0)
        assert 0 < ttc < 10

    def test_ttc_diverging(self):
        """Two vehicles moving apart — infinite TTC."""
        ttc = compute_ttc(0, 0, -5, 0, 50, 0, 10, 0)
        assert ttc == float("inf")

    def test_hazard_weight_imminent(self):
        """Low TTC → high hazard."""
        h = compute_hazard_weight(1.0, tau_crit=3.0, lambda_scale=0.5)
        assert h > 0.95

    def test_hazard_weight_safe(self):
        """High TTC → low hazard."""
        h = compute_hazard_weight(10.0, tau_crit=3.0, lambda_scale=0.5)
        assert h < 0.01

    def test_hazard_weight_infinite_ttc(self):
        h = compute_hazard_weight(float("inf"))
        assert h == 0.0

    def test_build_scene_graph(self):
        detections = [
            {"track_id": 0, "class_id": 0, "bbox": (0.5, 0.5, 0.1, 0.1),
             "x": 30, "y": 0, "vx": -5, "vy": 0, "ax": 0, "ay": 0, "heading": 3.14},
            {"track_id": 1, "class_id": 1, "bbox": (0.3, 0.3, 0.05, 0.05),
             "x": 15, "y": 2, "vx": -1, "vy": 0.5, "ax": 0, "ay": 0, "heading": 2.0},
        ]
        sg = build_scene_graph(detections)
        assert sg.num_nodes == 2
        assert sg.edge_index.shape[0] == 2
        assert sg.to_node_features().shape == (2, 12)


# ═══════════════════════════════════════════════════════════
#  PIKV (Kinematic Bicycle Model) Tests
# ═══════════════════════════════════════════════════════════

class TestPIKV:
    def test_bicycle_model_output_shape(self):
        cfg = PCCSGConfig()
        model = KinematicBicycleModel(cfg)
        B, H = 4, 15
        init = torch.randn(B, 4)
        ctrl = torch.randn(B, H, 2)
        traj = model(init, ctrl)
        assert traj.shape == (B, H, 2)

    def test_bicycle_model_zero_control(self):
        """Zero controls → straight-line motion."""
        cfg = PCCSGConfig()
        model = KinematicBicycleModel(cfg)
        init = torch.tensor([[0.0, 0.0, 10.0, 0.0]])  # x=0, y=0, v=10, θ=0
        ctrl = torch.zeros(1, 15, 2)
        traj = model(init, ctrl)
        # Should move in x direction
        assert traj[0, -1, 0].item() > 0
        assert abs(traj[0, -1, 1].item()) < 0.1

    def test_pikv_validates_straight_line(self):
        """Straight line at constant speed should be valid."""
        cfg = PCCSGConfig()
        pikv = PhysicsInformedKinematicValidator(cfg)
        B, H = 1, 15
        dt = cfg.delta_t
        # Constant velocity: 10 m/s in x
        x = torch.arange(H, dtype=torch.float32).unsqueeze(0) * 10.0 * dt
        y = torch.zeros(B, H)
        traj = torch.stack([x, y], dim=-1)
        speeds = torch.tensor([10.0])
        valid, violation = pikv(traj, speeds)
        assert valid[0].item() is True
        assert violation[0].item() < 0.1

    def test_pikv_rejects_teleportation(self):
        """Instantaneous teleportation should be invalid."""
        cfg = PCCSGConfig()
        pikv = PhysicsInformedKinematicValidator(cfg)
        dt = cfg.delta_t
        H = 15
        # Jump 500m per frame — vastly exceeds any kinematic constraint
        traj = torch.zeros(1, H, 2)
        for t in range(H):
            traj[0, t, 0] = 500.0 * t
            traj[0, t, 1] = 0.0
        speeds = torch.tensor([10.0])
        valid, violation = pikv(traj, speeds)
        assert violation[0].item() > 0.1


# ═══════════════════════════════════════════════════════════
#  CSGA Tests
# ═══════════════════════════════════════════════════════════

class TestCSGA:
    def test_causal_gating_output_shape(self):
        gate = CausalAttentionGating(node_dim=128, edge_dim=128)
        B, N, D = 2, 5, 128
        E = 20
        nodes = torch.randn(B, N, D)
        edge_idx = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(B, E, D)
        out = gate(nodes, edge_idx, edge_attr)
        assert out.shape == (B, E, 1)
        assert (out >= 0).all() and (out <= 1).all()

    def test_cf_injector_output_shape(self):
        inj = CounterfactualQueryInjector(d_model=128, num_counterfactuals=6)
        B, N, D = 2, 5, 128
        nodes = torch.randn(B, N, D)
        mask = torch.tensor([[True, True, False, False, False],
                             [True, False, False, False, False]])
        cf, modes = inj(nodes, mask)
        assert cf.shape == (B, N, 6, D)
        assert modes.shape == (6,)

    def test_csga_forward(self):
        cfg = PCCSGConfig()
        csga = CounterfactualSceneGraphAttention(
            node_dim=cfg.node_feature_dim,
            edge_dim=cfg.edge_feature_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_counterfactuals=cfg.num_counterfactuals,
        )
        B, N = 2, 5
        E = 20
        node_feat = torch.randn(B, N, cfg.node_feature_dim)
        edge_idx = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(B, E, cfg.edge_feature_dim)
        hazard = torch.rand(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)

        out = csga(node_feat, edge_idx, edge_attr, hazard, mask)
        assert out["scene_embeddings"].shape == (B, N, cfg.d_model)
        assert out["cf_risk_logits"].shape == (B, N, cfg.num_counterfactuals)
        assert out["critical_mask"].shape == (B, N)

    def test_csga_risk_logits_bounded(self):
        cfg = PCCSGConfig()
        csga = CounterfactualSceneGraphAttention(
            node_dim=cfg.node_feature_dim,
            edge_dim=cfg.edge_feature_dim,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_counterfactuals=cfg.num_counterfactuals,
        )
        B, N, E = 1, 3, 6
        out = csga(
            torch.randn(B, N, cfg.node_feature_dim),
            torch.randint(0, N, (2, E)),
            torch.randn(B, E, cfg.edge_feature_dim),
            torch.rand(B, N),
            torch.ones(B, N, dtype=torch.bool),
        )
        risk = out["cf_risk_logits"]
        assert (risk >= 0).all() and (risk <= 1).all()


# ═══════════════════════════════════════════════════════════
#  CRT Engine Tests
# ═══════════════════════════════════════════════════════════

class TestCRTEngine:
    def test_crt_output_range(self):
        cfg = PCCSGConfig()
        engine = CRTEngine(cfg)
        B, N, K = 2, 5, 6
        risk = torch.rand(B, N, K)
        valid = torch.ones(B, N, K, dtype=torch.bool)
        hazard = torch.rand(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        crt = engine(risk, valid, hazard, mask)
        assert crt.shape == (B,)
        assert (crt >= 0).all() and (crt <= 1).all()

    def test_crt_zero_when_no_risk(self):
        cfg = PCCSGConfig()
        engine = CRTEngine(cfg)
        B, N, K = 1, 5, 6
        risk = torch.zeros(B, N, K)
        valid = torch.ones(B, N, K, dtype=torch.bool)
        hazard = torch.zeros(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        crt = engine(risk, valid, hazard, mask)
        assert crt[0].item() < 0.01

    def test_intervention_arbiter_nominal(self):
        arbiter = PreemptiveInterventionArbiter()
        decision = arbiter.arbitrate(
            crt_scalar=0.10,
            per_agent_risk=torch.zeros(5),
            per_cf_risk=torch.zeros(5, 6),
            hazard_weights=torch.zeros(5),
        )
        assert decision.level == InterventionLevel.NOMINAL

    def test_intervention_arbiter_emergency(self):
        arbiter = PreemptiveInterventionArbiter()
        decision = arbiter.arbitrate(
            crt_scalar=0.92,
            per_agent_risk=torch.tensor([0.9, 0.1, 0.05, 0.02, 0.01]),
            per_cf_risk=torch.rand(5, 6),
            hazard_weights=torch.tensor([0.95, 0.1, 0.05, 0.02, 0.01]),
            worst_case_ttc=0.8,
        )
        assert decision.level == InterventionLevel.EMERGENCY


# ═══════════════════════════════════════════════════════════
#  End-to-End Network Tests
# ═══════════════════════════════════════════════════════════

class TestPCCSGNetwork:
    def test_forward_shape(self):
        cfg = PCCSGConfig()
        net = PCCSGNetwork(cfg)
        B, N = 2, 5
        E = 20
        node_feat = torch.randn(B, N, cfg.node_feature_dim)
        edge_idx = torch.randint(0, N, (2, E))
        edge_attr = torch.randn(B, E, cfg.edge_feature_dim)
        hazard = torch.rand(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)

        out = net(node_feat, edge_idx, edge_attr, hazard, mask, node_feat)
        assert out["crt_scalar"].shape == (B,)
        assert out["cf_risk_logits"].shape == (B, N, cfg.num_counterfactuals)
        assert out["pikv_validity"].shape == (B, N, cfg.num_counterfactuals)
        assert out["trajectories"].shape == (B, N, cfg.num_counterfactuals, cfg.prediction_horizon, 2)

    def test_crt_scalar_bounded(self):
        cfg = PCCSGConfig()
        net = PCCSGNetwork(cfg)
        B, N, E = 1, 3, 6
        out = net(
            torch.randn(B, N, cfg.node_feature_dim),
            torch.randint(0, N, (2, E)),
            torch.randn(B, E, cfg.edge_feature_dim),
            torch.rand(B, N),
            torch.ones(B, N, dtype=torch.bool),
            torch.randn(B, N, cfg.node_feature_dim),
        )
        crt = out["crt_scalar"]
        assert (crt >= 0).all() and (crt <= 1).all()

    def test_gradient_flow(self):
        """Verify gradients flow through entire pipeline."""
        cfg = PCCSGConfig()
        net = PCCSGNetwork(cfg)
        B, N, E = 1, 3, 6
        node_feat = torch.randn(B, N, cfg.node_feature_dim, requires_grad=True)
        out = net(
            node_feat,
            torch.randint(0, N, (2, E)),
            torch.randn(B, E, cfg.edge_feature_dim),
            torch.rand(B, N),
            torch.ones(B, N, dtype=torch.bool),
            node_feat,
        )
        loss = out["crt_scalar"].sum()
        loss.backward()
        assert node_feat.grad is not None
        assert node_feat.grad.abs().sum().item() > 0


# ═══════════════════════════════════════════════════════════
#  Data Generation Tests
# ═══════════════════════════════════════════════════════════

class TestDataGeneration:
    def test_synthetic_generator(self):
        gen = SyntheticTrafficScenarioGenerator()
        sample = gen.generate_nominal_scenario()
        assert "node_features" in sample
        assert "crt_label" in sample
        assert sample["crt_label"] < 0.25

    def test_critical_scenario_high_risk(self):
        gen = SyntheticTrafficScenarioGenerator()
        sample = gen.generate_critical_scenario()
        assert sample["crt_label"] > 0.75

    def test_dataset_length(self):
        ds = PCCSGDataset(num_samples=20)
        assert len(ds) == 20

    def test_dataset_item_keys(self):
        ds = PCCSGDataset(num_samples=4)
        item = ds[0]
        assert "node_features" in item
        assert "hazard_weights" in item
        assert "crt_label" in item
        assert item["node_features"].shape[0] == PCCSGConfig().max_nodes


# ═══════════════════════════════════════════════════════════
#  Proposition 2 Verification (Physics Plausibility Bound)
# ═══════════════════════════════════════════════════════════

class TestProposition2:
    """
    Verify Proposition 2: PIKV eliminates physically impossible trajectories.

    Any trajectory where a_total > μ·g (tire friction limit exceeded)
    must be flagged as invalid by PIKV.
    """

    def test_friction_circle_violation(self):
        """Trajectory exceeding friction circle must be invalid."""
        cfg = PCCSGConfig()
        pikv = PhysicsInformedKinematicValidator(cfg)
        mu_g = cfg.friction_coefficient * cfg.gravity
        dt = cfg.delta_t

        # Create trajectory with extreme lateral acceleration
        # Agent going 20 m/s, then instantly turning 90°
        H = 10
        traj = torch.zeros(1, H, 2)
        for t in range(H):
            if t < 3:
                traj[0, t, 0] = 20.0 * dt * t  # Forward
                traj[0, t, 1] = 0.0
            else:
                traj[0, t, 0] = traj[0, 2, 0]
                traj[0, t, 1] = 20.0 * dt * (t - 2)  # Sudden lateral

        valid, violation = pikv(traj, torch.tensor([20.0]))
        # Should detect the violation
        assert violation[0].item() > 0.1

    def test_gentle_curve_valid(self):
        """Gentle curve within friction limits should be valid."""
        cfg = PCCSGConfig()
        pikv = PhysicsInformedKinematicValidator(cfg)
        dt = cfg.delta_t
        H = 15
        speed = 10.0
        radius = 50.0  # Gentle curve

        traj = torch.zeros(1, H, 2)
        for t in range(H):
            theta = speed * dt * t / radius
            traj[0, t, 0] = radius * np.sin(theta)
            traj[0, t, 1] = radius * (1 - np.cos(theta))

        valid, violation = pikv(traj, torch.tensor([speed]))
        assert valid[0].item() is True


# ═══════════════════════════════════════════════════════════
#  Real-World Dataset & Video Inference Tests
# ═══════════════════════════════════════════════════════════

class TestRealWorldModules:
    def test_real_accident_dataset_loading(self):
        """Verify real-world dataset loading and graph collation."""
        from pc_csg.data import RealTrafficAccidentDataset
        from pc_csg.training.trainer import collate_graph_batch

        ds = RealTrafficAccidentDataset(split="train", max_samples=4)
        assert len(ds) == 4
        item = ds[0]
        assert "node_features" in item
        assert "hazard_weights" in item
        assert "crt_label" in item
        assert "level_label" in item

        batch = collate_graph_batch([ds[0], ds[1]])
        assert batch["node_features"].shape[0] == 2
        assert batch["edge_index"].shape[0] == 2
        assert batch["edge_attr"].shape[0] == 2

    def test_pccsg_loss_computation(self):
        """Verify multi-task loss forward and backward pass."""
        from pc_csg.models.pc_csg_net import PCCSGNetwork
        from pc_csg.training.losses import PCCSGLoss

        cfg = PCCSGConfig()
        model = PCCSGNetwork(cfg)
        loss_fn = PCCSGLoss(cfg)

        B, N = 2, cfg.max_nodes
        node_feat = torch.randn(B, N, cfg.node_feature_dim)
        edge_idx = torch.zeros(2, 10, dtype=torch.long)
        edge_attr = torch.randn(B, 10, cfg.edge_feature_dim)
        hazard = torch.rand(B, N)
        mask = torch.ones(B, N, dtype=torch.bool)
        crt_targets = torch.tensor([0.2, 0.8])
        level_targets = torch.tensor([0, 4])

        outputs = model(node_feat, edge_idx, edge_attr, hazard, mask)
        losses = loss_fn(outputs, crt_targets, level_targets)

        assert "loss" in losses
        assert "loss_crt" in losses
        assert "loss_level" in losses
        assert torch.isfinite(losses["loss"]).all()
        assert losses["loss"].item() > 0

    def test_real_video_engine_frame_processing(self):
        """Verify real video inference engine processes a frame and produces HUD."""
        from pc_csg.inference import RealVideoInferenceEngine

        engine = RealVideoInferenceEngine(checkpoint_path=None, device=torch.device("cpu"))
        dummy_frame = np.full((360, 640, 3), 100, dtype=np.uint8)

        annotated_frame, metrics = engine.process_frame(dummy_frame, t_sec=1.0)
        assert annotated_frame.shape == (360, 640, 3)
        assert "crt" in metrics
        assert "level" in metrics
        assert "latency_ms" in metrics
        assert 0.0 <= metrics["crt"] <= 1.0


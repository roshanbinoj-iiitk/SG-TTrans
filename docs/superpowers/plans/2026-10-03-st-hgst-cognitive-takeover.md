# ST-HGST: Cognitive Saliency Alignment & Causal Risk Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the Spatio-Temporal Hypergraph Gaze-Scene Transformer (ST-HGST) framework for cognitive saliency alignment, causal hazard grounding, and context-aware Level-3 takeover arbitration.

**Architecture:** Dual-stream pipeline combining an interior 3D gaze kinematics estimator and camera-windshield homography projection with an exterior dynamic spatio-temporal scene graph. A bipartite multi-head cross-attention transformer couples driver attention to exterior hazard nodes, filtered through a neuro-visual leaky cognitive accumulator to quantify the Epistemic Attention Gap (EAG) for Euro NCAP 2026 Level-3 Takeover Arbitration (Safe Handover, Targeted Spatial HUD Cueing, Autonomous Minimum Risk Maneuver).

**Tech Stack:** Python 3.12, PyTorch 2.6+cu124, NumPy, OpenCV, MediaPipe, PyTest (running in `/home/roshanbinoj/Documents/BTP/venv/`).

**Spec:** `docs/superpowers/specs/2026-10-03-st-hgst-cognitive-takeover-design.md`

## Global Constraints
- Target Hardware: Real-time throughput (>45 FPS, <21.6 ms latency) on NVIDIA GeForce RTX 3050 Laptop GPU under 1.0 GB VRAM.
- Virtual Environment: All execution and pytest runs MUST use `/home/roshanbinoj/Documents/BTP/venv/bin/python` and `/home/roshanbinoj/Documents/BTP/venv/bin/pytest`.
- Mathematical Invariants: Gaze unit norm $\|\mathbf{g}\|_2 = 1.0 \pm 10^{-6}$; Cross-attention weights sum to $1.0 \pm 10^{-6}$; $EAG(t) \in [0.0, 1.0]$.
- Perception Latency Threshold: Transient gaze ($\le 100\text{ ms}$, 3 frames) MUST NOT achieve Comprehension ($\mathcal{C} < 0.60$); sustained fixation ($\ge 250\text{ ms}$, 8 frames) MUST achieve Comprehension ($\mathcal{C} \ge 0.60$).
- Code Modularity: All modules must be decoupled dataclasses/nn.Modules with explicit tensor contracts.

## Review Focus
1. Empty Scene ($N=0$ objects): System must not divide by zero or crash; must output $EAG = 0.0$ and nominal hold.
2. In-Cabin Tracking Loss / Gaze Occlusion: Gaze vector loss must inflate uncertainty and elevate $EAG$, preventing accidental manual handover.
3. Saccadic Transience vs. Sustained Fixation: Ensure rapid glance across a hazard does not trigger handover; driver must dwell for $\ge 250\text{ ms}$.
4. High Collision Imminence Override: If $TTC < 1.2\text{ s}$, arbitrator MUST trigger Level 2 MRM immediately, overriding any partial comprehension.
5. Dynamic Object Cardinality ($N_t$ variation): Cross-attention kernel and graph encoder must gracefully support variable object counts from $N=1$ to $N=30$ per frame without recompilation or batch crashes.

---

### Task 1: Global Configuration & Core Data Contracts

**Files:**
- Create: `sg_ttrans/types.py`
- Modify: `sg_ttrans/config.py`
- Test: `tests/test_types_config.py`

**Interfaces:**
- Consumes: None (base types).
- Produces: `SceneNode`, `SceneGraph`, `GazeVector`, `CognitiveState`, `ArbitrationDecision`, `STHGSTConfig`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types_config.py
import pytest
import numpy as np
import torch
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import (
    SceneNode,
    SceneGraph,
    GazeVector,
    CognitiveState,
    ArbitrationDecision,
    TakeoverAction,
)

def test_config_initialization():
    cfg = STHGSTConfig()
    assert cfg.tau_crit == 2.5
    assert cfg.tau_cog == 0.25
    assert cfg.theta_comp == 0.60
    assert cfg.theta_sacc == 0.20
    assert cfg.eag_handover_thresh == 0.20
    assert cfg.eag_cue_thresh == 0.65

def test_scene_node_contract():
    node = SceneNode(
        track_id=1,
        class_id=0,
        bbox=(0.1, 0.2, 0.3, 0.4),
        velocity=(0.0, -2.5),
        ttc=1.8,
        hazard_weight=0.82,
    )
    assert node.track_id == 1
    assert node.ttc == 1.8
    assert node.hazard_weight == 0.82

def test_gaze_vector_normalization():
    raw_vec = np.array([1.0, 1.0, 1.0])
    gaze = GazeVector.from_raw(raw_vec, pitch=0.1, yaw=-0.2)
    assert np.isclose(np.linalg.norm(gaze.unit_vector), 1.0, atol=1e-5)
    assert gaze.pitch == 0.1
    assert gaze.yaw == -0.2

def test_arbitration_decision_contract():
    dec = ArbitrationDecision(
        action=TakeoverAction.SAFE_HANDOVER,
        eag=0.08,
        causal_hazard_id=None,
        action_description="Safe torque handover authorized",
    )
    assert dec.action == TakeoverAction.SAFE_HANDOVER
    assert dec.eag < 0.20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_types_config.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sg_ttrans.types'`

- [ ] **Step 3: Write minimal implementation**

```python
# sg_ttrans/types.py
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
        norm = np.linalg.norm(raw)
        unit = raw / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
        return cls(unit_vector=unit, pitch=pitch, yaw=yaw)

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
```

```python
# sg_ttrans/config.py
from dataclasses import dataclass

@dataclass
class STHGSTConfig:
    # Temporal & Kinematics
    fps: float = 30.0
    delta_t: float = 1.0 / 30.0
    tau_crit: float = 2.5                    # Critical TTC reaction threshold (seconds)
    lambda_scale: float = 0.5                # Sigmoid scaling factor for hazard scoring
    ttc_imminent_threshold: float = 1.2      # TTC below which emergency MRM is forced
    
    # Cognitive Latency Parameters
    tau_cog: float = 0.25                    # Cognitive comprehension time constant (seconds)
    theta_comp: float = 0.60                 # Accumulation threshold for Comprehended state
    theta_sacc: float = 0.20                 # Accumulation threshold for Saccade state
    
    # Takeover Arbitration Thresholds
    eag_handover_thresh: float = 0.20        # EAG below which handover is safe
    eag_cue_thresh: float = 0.65             # EAG above which MRM is mandated
    cue_hold_duration_sec: float = 1.5       # Hold duration for spatial HUD cue
    
    # Model Hyperparameters
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    max_nodes: int = 30

cfg = STHGSTConfig()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_types_config.py -v`  
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sg_ttrans/types.py sg_ttrans/config.py tests/test_types_config.py
git commit -m "feat(core): add ST-HGST data contracts and global configuration"
```

---

### Task 2: In-Cabin 3D Gaze Geometry & Windshield Projection

**Files:**
- Create: `sg_ttrans/cabin/gaze_estimator.py`
- Create: `tests/test_gaze_projection.py`

**Interfaces:**
- Consumes: `GazeVector` from `sg_ttrans.types`.
- Produces: `project_gaze_to_windshield(gaze, homography_matrix) -> Tuple[float, float]`, `generate_fixation_cone(point, sigma, resolution) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gaze_projection.py
import pytest
import numpy as np
from sg_ttrans.types import GazeVector
from sg_ttrans.cabin.gaze_estimator import (
    project_gaze_to_windshield,
    generate_fixation_cone,
    GazeProjector,
)

def test_gaze_projection_center():
    projector = GazeProjector()
    # Looking directly along z-axis (pitch=0, yaw=0)
    gaze = GazeVector(unit_vector=np.array([0.0, 0.0, 1.0]), pitch=0.0, yaw=0.0)
    pt = projector.project(gaze)
    assert 0.0 <= pt[0] <= 1.0
    assert 0.0 <= pt[1] <= 1.0
    # Optical center should project near (0.5, 0.5)
    assert np.isclose(pt[0], 0.5, atol=0.05)
    assert np.isclose(pt[1], 0.5, atol=0.05)

def test_gaze_projection_clamping():
    projector = GazeProjector()
    # Extreme gaze looking far right/up
    gaze = GazeVector(unit_vector=np.array([0.9, 0.9, 0.1]), pitch=1.2, yaw=1.2)
    pt = projector.project(gaze)
    assert 0.0 <= pt[0] <= 1.0
    assert 0.0 <= pt[1] <= 1.0

def test_fixation_cone_density():
    center = (0.5, 0.5)
    cone = generate_fixation_cone(center, sigma=0.08, grid_size=(64, 64))
    assert cone.shape == (64, 64)
    assert np.isclose(cone.sum(), 1.0, atol=1e-3)
    # Peak density should be at grid center (32, 32)
    assert np.argmax(cone) == 32 * 64 + 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_gaze_projection.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sg_ttrans.cabin'`

- [ ] **Step 3: Write minimal implementation**

```python
# sg_ttrans/cabin/__init__.py
# In-cabin driver perception and 3D gaze kinematics
```

```python
# sg_ttrans/cabin/gaze_estimator.py
from typing import Tuple
import numpy as np
from sg_ttrans.types import GazeVector

class GazeProjector:
    """Projects 3D in-cabin gaze vectors to normalized 2D road camera windshield coordinates."""
    def __init__(self, homography: np.ndarray = None):
        if homography is None:
            # Default canonical projection mapping center gaze [0, 0, 1] to (0.5, 0.5)
            self.H = np.array([
                [0.6, 0.0, 0.5],
                [0.0, 0.6, 0.5],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
        else:
            self.H = homography

    def project(self, gaze: GazeVector) -> Tuple[float, float]:
        v = gaze.unit_vector
        z = max(v[2], 1e-4)
        x_proj = v[0] / z
        y_proj = v[1] / z
        
        # Homogeneous transformation
        p_homo = self.H @ np.array([x_proj, y_proj, 1.0])
        u = float(np.clip(p_homo[0] / p_homo[2], 0.0, 1.0))
        v_coord = float(np.clip(p_homo[1] / p_homo[2], 0.0, 1.0))
        gaze.windshield_point = (u, v_coord)
        return (u, v_coord)

def project_gaze_to_windshield(gaze: GazeVector, H: np.ndarray = None) -> Tuple[float, float]:
    return GazeProjector(H).project(gaze)

def generate_fixation_cone(
    center: Tuple[float, float],
    sigma: float = 0.08,
    grid_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Generates normalized 2D Gaussian fixation cone over image plane."""
    h, w = grid_size
    y = np.linspace(0.0, 1.0, h)
    x = np.linspace(0.0, 1.0, w)
    xx, yy = np.meshgrid(x, y)
    
    dist_sq = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    gaussian = np.exp(-dist_sq / (2.0 * (sigma ** 2)))
    sum_val = gaussian.sum()
    if sum_val > 1e-8:
        gaussian /= sum_val
    return gaussian.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_gaze_projection.py -v`  
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sg_ttrans/cabin/ tests/test_gaze_projection.py
git commit -m "feat(cabin): implement 3D gaze vector projection and fixation cone generator"
```

---

### Task 3: Exterior Dynamic Scene Graph & TTC Hazard Engine

**Files:**
- Create: `sg_ttrans/scene_graph/ttc_calculator.py`
- Create: `sg_ttrans/scene_graph/graph_builder.py`
- Create: `tests/test_scene_graph.py`

**Interfaces:**
- Consumes: `SceneNode`, `SceneGraph`, `STHGSTConfig`.
- Produces: `compute_ttc(dist, rel_vel) -> float`, `compute_hazard_weight(ttc, cfg) -> float`, `build_scene_graph(detections, velocities) -> SceneGraph`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scene_graph.py
import pytest
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.scene_graph.ttc_calculator import compute_ttc, compute_hazard_weight
from sg_ttrans.scene_graph.graph_builder import DynamicSceneGraphBuilder

def test_ttc_calculation():
    # Approaching at 5 m/s from 10m away -> TTC = 2.0s
    assert np.isclose(compute_ttc(distance=10.0, relative_velocity=-5.0), 2.0)
    # Moving away (+3 m/s) -> TTC = infinity
    assert compute_ttc(distance=10.0, relative_velocity=3.0) == float("inf")
    # Zero velocity -> TTC = infinity
    assert compute_ttc(distance=10.0, relative_velocity=0.0) == float("inf")

def test_hazard_weight_sigmoid():
    cfg = STHGSTConfig()
    # At critical horizon (TTC = 2.5s) -> H = 0.5
    h_crit = compute_hazard_weight(ttc=2.5, cfg=cfg)
    assert np.isclose(h_crit, 0.5, atol=0.02)
    # Imminent collision (TTC = 1.0s) -> H close to 1.0
    h_imm = compute_hazard_weight(ttc=1.0, cfg=cfg)
    assert h_imm > 0.90
    # Far obstacle (TTC = 10.0s) -> H close to 0.0
    h_far = compute_hazard_weight(ttc=10.0, cfg=cfg)
    assert h_far < 0.05

def test_dynamic_scene_graph_builder():
    builder = DynamicSceneGraphBuilder()
    detections = [
        {"id": 1, "class_id": 0, "bbox": (0.2, 0.4, 0.1, 0.2), "distance": 12.0, "rel_vel": -6.0},
        {"id": 2, "class_id": 1, "bbox": (0.6, 0.3, 0.2, 0.3), "distance": 25.0, "rel_vel": -2.0},
    ]
    graph = builder.build(detections)
    assert len(graph.nodes) == 2
    assert graph.nodes[0].ttc == 2.0
    assert graph.nodes[0].hazard_weight > 0.60
    assert graph.edge_index.shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_scene_graph.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sg_ttrans.scene_graph'`

- [ ] **Step 3: Write minimal implementation**

```python
# sg_ttrans/scene_graph/__init__.py
# Exterior dynamic scene graph and TTC hazard calculation
```

```python
# sg_ttrans/scene_graph/ttc_calculator.py
import numpy as np
from sg_ttrans.config import STHGSTConfig

def compute_ttc(distance: float, relative_velocity: float) -> float:
    """Computes Time-to-Collision (TTC) in seconds. Negative rel_vel indicates approach."""
    if relative_velocity < -1e-4:
        return float(distance / (-relative_velocity))
    return float("inf")

def compute_hazard_weight(ttc: float, cfg: STHGSTConfig = None) -> float:
    """Calculates continuous intrinsic hazard weight H(v_i) in [0, 1]."""
    if cfg is None:
        cfg = STHGSTConfig()
    if ttc == float("inf") or ttc <= 0:
        return 0.0
    val = (ttc - cfg.tau_crit) / cfg.lambda_scale
    sig = 1.0 / (1.0 + np.exp(val))
    return float(np.clip(sig, 0.0, 1.0))
```

```python
# sg_ttrans/scene_graph/graph_builder.py
from typing import Dict, List
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.scene_graph.ttc_calculator import compute_ttc, compute_hazard_weight
from sg_ttrans.types import SceneGraph, SceneNode

class DynamicSceneGraphBuilder:
    def __init__(self, cfg: STHGSTConfig = None):
        self.cfg = cfg or STHGSTConfig()

    def build(self, raw_detections: List[Dict]) -> SceneGraph:
        nodes = []
        positions = []
        for det in raw_detections:
            dist = float(det.get("distance", 15.0))
            rel_vel = float(det.get("rel_vel", 0.0))
            ttc = compute_ttc(dist, rel_vel)
            h_weight = compute_hazard_weight(ttc, self.cfg)
            bbox = det.get("bbox", (0.0, 0.0, 0.1, 0.1))
            
            node = SceneNode(
                track_id=int(det.get("id", len(nodes))),
                class_id=int(det.get("class_id", 0)),
                bbox=bbox,
                velocity=(float(det.get("vx", 0.0)), float(rel_vel)),
                ttc=ttc,
                hazard_weight=h_weight,
            )
            nodes.append(node)
            positions.append([bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0])

        n_nodes = len(nodes)
        if n_nodes < 2:
            return SceneGraph(
                nodes=nodes,
                edge_index=np.zeros((2, 0), dtype=np.int64),
                edge_attr=np.zeros((0, 2), dtype=np.float32),
            )

        # Build fully connected directed edges between distinct nodes
        edge_sources = []
        edge_targets = []
        edge_attrs = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_sources.append(i)
                    edge_targets.append(j)
                    p_i = np.array(positions[i])
                    p_j = np.array(positions[j])
                    dist = float(np.linalg.norm(p_i - p_j))
                    edge_attrs.append([dist, 1.0 / (dist + 1e-4)])

        return SceneGraph(
            nodes=nodes,
            edge_index=np.array([edge_sources, edge_targets], dtype=np.int64),
            edge_attr=np.array(edge_attrs, dtype=np.float32),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_scene_graph.py -v`  
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sg_ttrans/scene_graph/ tests/test_scene_graph.py
git commit -m "feat(scene_graph): implement TTC computation and dynamic scene graph builder"
```

---

### Task 4: Bipartite Gaze-Scene Cross-Attention Graph Transformer

**Files:**
- Create: `sg_ttrans/models/gaze_cross_attention.py`
- Create: `sg_ttrans/models/st_hgst_net.py`
- Create: `tests/test_cross_attention.py`

**Interfaces:**
- Consumes: `gaze_tensor` (B, 1, 5), `scene_nodes_tensor` (B, N, 8), `node_mask` (B, N).
- Produces: `attention_weights` (B, N) where $\sum_{i=1}^N \alpha_i = 1.0$, `node_embeddings` (B, N, D).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cross_attention.py
import pytest
import torch
from sg_ttrans.models.gaze_cross_attention import BipartiteGazeSceneAttention
from sg_ttrans.models.st_hgst_net import STHGSTNetwork

def test_cross_attention_normalization():
    model = BipartiteGazeSceneAttention(embed_dim=128, num_heads=4)
    B, N = 2, 5
    # Gaze token: [gx, gy, gz, ug, vg] -> (B, 1, 5)
    gaze_tokens = torch.randn(B, 1, 5)
    # Node features: [class, x, y, w, h, vx, vy, ttc] -> (B, N, 8)
    node_tokens = torch.randn(B, N, 8)
    
    attn_weights, out_nodes = model(gaze_tokens, node_tokens)
    assert attn_weights.shape == (B, N)
    # Must sum to 1.0 along object dimension
    sums = attn_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

def test_cross_attention_with_padding_mask():
    model = BipartiteGazeSceneAttention(embed_dim=128, num_heads=4)
    B, N = 1, 4
    gaze_tokens = torch.randn(B, 1, 5)
    node_tokens = torch.randn(B, N, 8)
    # Mask out the last 2 nodes (valid length = 2)
    mask = torch.tensor([[True, True, False, False]])
    
    attn_weights, _ = model(gaze_tokens, node_tokens, mask=mask)
    # Masked nodes must receive exactly zero attention
    assert torch.allclose(attn_weights[0, 2:], torch.zeros(2), atol=1e-6)
    assert torch.isclose(attn_weights[0, :2].sum(), torch.tensor(1.0), atol=1e-5)

def test_end_to_end_network_forward_pass():
    net = STHGSTNetwork(embed_dim=128, num_heads=4)
    gaze = torch.randn(2, 1, 5)
    nodes = torch.randn(2, 6, 8)
    out = net(gaze, nodes)
    assert "attention_weights" in out
    assert "node_embeddings" in out
    assert out["attention_weights"].shape == (2, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_cross_attention.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sg_ttrans.models.gaze_cross_attention'`

- [ ] **Step 3: Write minimal implementation**

```python
# sg_ttrans/models/gaze_cross_attention.py
import torch
import torch.nn as nn
from typing import Optional, Tuple

class BipartiteGazeSceneAttention(nn.Module):
    """Bipartite multi-head cross-attention between driver 3D gaze and exterior scene nodes."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.gaze_proj = nn.Linear(5, embed_dim)
        self.node_proj = nn.Linear(8, embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        gaze_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gaze_tokens: (B, 1, 5)
            node_tokens: (B, N, 8)
            mask: (B, N) boolean mask where True indicates valid node, False is padding.
        Returns:
            attn_weights: (B, N)
            updated_nodes: (B, N, embed_dim)
        """
        B, N, _ = node_tokens.shape
        q = self.gaze_proj(gaze_tokens)      # (B, 1, embed_dim)
        k = self.node_proj(node_tokens)      # (B, N, embed_dim)
        v = k
        
        key_padding_mask = ~mask if mask is not None else None
        out, raw_weights = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        # raw_weights has shape (B, 1, N)
        attn_weights = raw_weights.squeeze(1)  # (B, N)
        
        # Guarantee numerical simplex constraint
        if mask is not None:
            attn_weights = attn_weights * mask.float()
            sums = attn_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_weights = attn_weights / sums
            
        updated_nodes = self.norm(k)
        return attn_weights, updated_nodes
```

```python
# sg_ttrans/models/st_hgst_net.py
import torch
import torch.nn as nn
from typing import Dict, Optional
from sg_ttrans.models.gaze_cross_attention import BipartiteGazeSceneAttention

class STHGSTNetwork(nn.Module):
    """Unified Spatio-Temporal Hypergraph Gaze-Scene Transformer Network."""
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = BipartiteGazeSceneAttention(embed_dim, num_heads, dropout)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        gaze: torch.Tensor,
        nodes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        attn_weights, node_embeds = self.attention(gaze, nodes, mask)
        node_embeds = node_embeds + self.temporal_mlp(node_embeds)
        return {
            "attention_weights": attn_weights,
            "node_embeddings": node_embeds,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_cross_attention.py -v`  
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sg_ttrans/models/ tests/test_cross_attention.py
git commit -m "feat(models): implement bipartite cross-attention graph transformer kernel"
```

---

### Task 5: Neuro-Visual Leaky Cognitive Accumulator

**Files:**
- Create: `sg_ttrans/cognitive/leaky_accumulator.py`
- Create: `tests/test_cognitive_decay.py`

**Interfaces:**
- Consumes: `attention_weights` (N,), `prev_accumulation` (N,), `STHGSTConfig`.
- Produces: `CognitiveState` (accumulated values, fixation states).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cognitive_decay.py
import pytest
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import FixationState
from sg_ttrans.cognitive.leaky_accumulator import LeakyCognitiveAccumulator

def test_transient_saccade_rejection():
    cfg = STHGSTConfig(fps=30.0, tau_cog=0.25, theta_comp=0.60)
    accumulator = LeakyCognitiveAccumulator(cfg)
    
    # 3 frames of high glance (100 ms)
    state = None
    for _ in range(3):
        state = accumulator.step(attention_weights=np.array([0.9, 0.1]))
        
    # Must NOT reach Comprehended state in only 100 ms
    assert state.accumulated_cognition[0] < cfg.theta_comp
    assert state.fixation_states[0] != FixationState.COMPREHENDED

def test_sustained_fixation_comprehension():
    cfg = STHGSTConfig(fps=30.0, tau_cog=0.25, theta_comp=0.60)
    accumulator = LeakyCognitiveAccumulator(cfg)
    
    # 10 frames of sustained gaze (333 ms)
    state = None
    for _ in range(10):
        state = accumulator.step(attention_weights=np.array([0.95, 0.05]))
        
    # Must comfortably reach Comprehended state
    assert state.accumulated_cognition[0] >= cfg.theta_comp
    assert state.fixation_states[0] == FixationState.COMPREHENDED

def test_cognitive_decay_when_attention_shifts():
    cfg = STHGSTConfig(fps=30.0, tau_cog=0.25)
    accumulator = LeakyCognitiveAccumulator(cfg)
    # Establish fixation on node 0
    for _ in range(10):
        accumulator.step(attention_weights=np.array([1.0, 0.0]))
    
    # Attention shifts away to node 1 for 15 frames (500 ms)
    state = None
    for _ in range(15):
        state = accumulator.step(attention_weights=np.array([0.0, 1.0]))
        
    # Node 0 must decay below Comprehension
    assert state.accumulated_cognition[0] < cfg.theta_comp
    assert state.fixation_states[0] != FixationState.COMPREHENDED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_cognitive_decay.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sg_ttrans.cognitive'`

- [ ] **Step 3: Write minimal implementation**

```python
# sg_ttrans/cognitive/__init__.py
# Human cognitive modeling and takeover arbitration
```

```python
# sg_ttrans/cognitive/leaky_accumulator.py
from typing import Optional
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import CognitiveState, FixationState

class LeakyCognitiveAccumulator:
    """Models human neuro-visual cognitive processing latency (tau_cog = 250 ms)."""
    def __init__(self, cfg: Optional[STHGSTConfig] = None):
        self.cfg = cfg or STHGSTConfig()
        # beta = exp(-delta_t / tau_cog)
        self.beta = float(np.exp(-self.cfg.delta_t / self.cfg.tau_cog))
        self.c_prev: Optional[np.ndarray] = None

    def reset(self):
        self.c_prev = None

    def step(self, attention_weights: np.ndarray) -> CognitiveState:
        weights = np.asarray(attention_weights, dtype=np.float32)
        if self.c_prev is None or len(self.c_prev) != len(weights):
            self.c_prev = np.zeros_like(weights)

        # Leaky accumulation: C_i(t) = beta * C_i(t-1) + (1 - beta) * alpha_i(t)
        c_t = self.beta * self.c_prev + (1.0 - self.beta) * weights
        c_t = np.clip(c_t, 0.0, 1.0)
        self.c_prev = c_t

        states = []
        for val in c_t:
            if val >= self.cfg.theta_comp:
                states.append(FixationState.COMPREHENDED)
            elif val >= self.cfg.theta_sacc:
                states.append(FixationState.SACCADIC_GLANCE)
            else:
                states.append(FixationState.UNSEEN)

        return CognitiveState(
            attention_weights=weights,
            accumulated_cognition=c_t,
            fixation_states=states,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_cognitive_decay.py -v`  
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sg_ttrans/cognitive/ tests/test_cognitive_decay.py
git commit -m "feat(cognitive): implement leaky cognitive accumulator with perception latency decay"
```

---

### Task 6: Epistemic Attention Gap (EAG) & Level-3 Takeover Arbitrator

**Files:**
- Create: `sg_ttrans/cognitive/epistemic_gap.py`
- Create: `sg_ttrans/cognitive/takeover_arbitrator.py`
- Create: `tests/test_eag_arbitration.py`

**Interfaces:**
- Consumes: `hazard_weights` (N,), `accumulated_cognition` (N,), `ttc_list` (N,), `STHGSTConfig`.
- Produces: `ArbitrationDecision` (action, eag, causal_hazard_id).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eag_arbitration.py
import pytest
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import TakeoverAction
from sg_ttrans.cognitive.epistemic_gap import compute_eag
from sg_ttrans.cognitive.takeover_arbitrator import TakeoverArbitrator

def test_scenario_a_aligned_attention():
    # Hazard is high (H=0.9), driver is fully comprehending it (C=0.85)
    eag, top_hazard = compute_eag(
        hazard_weights=np.array([0.9, 0.1]),
        accumulated_cognition=np.array([0.85, 0.05]),
        theta_comp=0.60,
    )
    assert eag < 0.20
    arbitrator = TakeoverArbitrator()
    decision = arbitrator.arbitrate(eag=eag, min_ttc=2.2, top_hazard_idx=top_hazard)
    assert decision.action == TakeoverAction.SAFE_HANDOVER

def test_scenario_b_inattentional_blindness():
    # Hazard is high (H=0.9), driver looking elsewhere (C=0.05 on hazard, 0.95 on safe road)
    eag, top_hazard = compute_eag(
        hazard_weights=np.array([0.9, 0.0]),
        accumulated_cognition=np.array([0.05, 0.95]),
        theta_comp=0.60,
    )
    assert 0.20 <= eag < 0.65
    assert top_hazard == 0
    arbitrator = TakeoverArbitrator()
    decision = arbitrator.arbitrate(eag=eag, min_ttc=2.0, top_hazard_idx=top_hazard)
    assert decision.action == TakeoverAction.TARGETED_SPATIAL_CUE
    assert decision.causal_hazard_id == 0

def test_scenario_c_imminent_collision_override():
    # Even if EAG is moderate, imminent collision (TTC = 1.0s < 1.2s) must force Level 2 MRM
    arbitrator = TakeoverArbitrator()
    decision = arbitrator.arbitrate(eag=0.15, min_ttc=1.0, top_hazard_idx=0)
    assert decision.action == TakeoverAction.MINIMUM_RISK_MANEUVER
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_eag_arbitration.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sg_ttrans.cognitive.epistemic_gap'`

- [ ] **Step 3: Write minimal implementation**

```python
# sg_ttrans/cognitive/epistemic_gap.py
from typing import Optional, Tuple
import numpy as np

def compute_eag(
    hazard_weights: np.ndarray,
    accumulated_cognition: np.ndarray,
    theta_comp: float = 0.60,
) -> Tuple[float, Optional[int]]:
    """
    Computes Epistemic Attention Gap (EAG) in [0, 1] and identifies causal hazard index.
    EAG = sum(H_i * (1 - min(1, C_i / theta_comp))) / max(sum(H_i), 1e-4)
    """
    h = np.asarray(hazard_weights, dtype=np.float32)
    c = np.asarray(accumulated_cognition, dtype=np.float32)
    if len(h) == 0:
        return 0.0, None

    comprehension_ratio = np.clip(c / max(theta_comp, 1e-4), 0.0, 1.0)
    uncomprehended_gap = 1.0 - comprehension_ratio
    hazard_gaps = h * uncomprehended_gap
    
    total_hazard = float(h.sum())
    if total_hazard < 1e-4:
        return 0.0, None

    eag = float(np.clip(hazard_gaps.sum() / max(total_hazard, 1.0), 0.0, 1.0))
    top_hazard_idx = int(np.argmax(hazard_gaps))
    return eag, top_hazard_idx
```

```python
# sg_ttrans/cognitive/takeover_arbitrator.py
from typing import Optional
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.types import ArbitrationDecision, TakeoverAction

class TakeoverArbitrator:
    """Euro NCAP 2026 Level-3 Context-Aware Takeover Safety State Machine."""
    def __init__(self, cfg: Optional[STHGSTConfig] = None):
        self.cfg = cfg or STHGSTConfig()

    def arbitrate(
        self,
        eag: float,
        min_ttc: float,
        top_hazard_idx: Optional[int],
    ) -> ArbitrationDecision:
        # Override Rule: Imminent collision mandates emergency MRM
        if min_ttc <= self.cfg.ttc_imminent_threshold:
            return ArbitrationDecision(
                action=TakeoverAction.MINIMUM_RISK_MANEUVER,
                eag=eag,
                causal_hazard_id=top_hazard_idx,
                action_description="Emergency MRM: Imminent collision window detected",
            )

        if eag < self.cfg.eag_handover_thresh:
            return ArbitrationDecision(
                action=TakeoverAction.SAFE_HANDOVER,
                eag=eag,
                causal_hazard_id=None,
                action_description="Safe Handover: Driver attention aligned with hazards",
            )
        elif eag < self.cfg.eag_cue_thresh:
            return ArbitrationDecision(
                action=TakeoverAction.TARGETED_SPATIAL_CUE,
                eag=eag,
                causal_hazard_id=top_hazard_idx,
                action_description=f"Targeted Spatial HUD Cue: Driver missed causal node {top_hazard_idx}",
            )
        else:
            return ArbitrationDecision(
                action=TakeoverAction.MINIMUM_RISK_MANEUVER,
                eag=eag,
                causal_hazard_id=top_hazard_idx,
                action_description="Emergency MRM: Severe epistemic cognitive misalignment",
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_eag_arbitration.py -v`  
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add sg_ttrans/cognitive/ tests/test_eag_arbitration.py
git commit -m "feat(cognitive): implement EAG calculation and Takeover Arbitration engine"
```

---

### Task 7: Synthetic Dual-Stream Data Engine for CI & Benchmarking

**Files:**
- Create: `sg_ttrans/data/synthetic_dual_stream.py`
- Create: `tests/test_synthetic_dual_stream.py`

**Interfaces:**
- Consumes: `batch_size`, `num_frames`, `scenario_type`.
- Produces: PyTorch dataset yielding synchronized `(gaze_tokens, scene_nodes, mask, ground_truth_action)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthetic_dual_stream.py
import pytest
import torch
from sg_ttrans.data.synthetic_dual_stream import SyntheticDualStreamDataset

def test_synthetic_dataset_batching():
    dataset = SyntheticDualStreamDataset(num_samples=10, num_objects=5)
    sample = dataset[0]
    assert "gaze" in sample
    assert "nodes" in sample
    assert "mask" in sample
    assert "target_action" in sample
    assert sample["gaze"].shape == (1, 5)
    assert sample["nodes"].shape == (5, 8)
    assert sample["mask"].shape == (5,)

def test_synthetic_scenario_types():
    # Scenario A (Handover): Gaze directed at hazard
    ds_a = SyntheticDualStreamDataset(num_samples=5, scenario="aligned")
    sample_a = ds_a[0]
    assert sample_a["target_action"] == 0

    # Scenario B (Spatial Cue): Gaze away from hazard
    ds_b = SyntheticDualStreamDataset(num_samples=5, scenario="blindness")
    sample_b = ds_b[0]
    assert sample_b["target_action"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_synthetic_dual_stream.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'sg_ttrans.data.synthetic_dual_stream'`

- [ ] **Step 3: Write minimal implementation**

```python
# sg_ttrans/data/synthetic_dual_stream.py
import torch
from torch.utils.data import Dataset
import numpy as np

class SyntheticDualStreamDataset(Dataset):
    """Generates synthetic synchronized dual-stream data for CI/CD and unit testing."""
    def __init__(self, num_samples: int = 100, num_objects: int = 5, scenario: str = "random"):
        self.num_samples = num_samples
        self.num_objects = num_objects
        self.scenario = scenario

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.RandomState(idx)
        n = self.num_objects
        nodes = np.zeros((n, 8), dtype=np.float32)
        
        # Node 0 is the critical hazard
        hazard_x, hazard_y = 0.5, 0.4
        nodes[0] = [0.0, hazard_x, hazard_y, 0.15, 0.25, 0.0, -5.0, 1.8]
        for i in range(1, n):
            nodes[i] = [1.0, rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.8), 0.1, 0.1, 0.0, 0.0, 8.0]

        if self.scenario == "aligned" or (self.scenario == "random" and idx % 3 == 0):
            # Gaze aligned on hazard
            gaze = np.array([hazard_x - 0.5, hazard_y - 0.5, 1.0, hazard_x, hazard_y], dtype=np.float32)
            target = 0
        elif self.scenario == "blindness" or (self.scenario == "random" and idx % 3 == 1):
            # Gaze away on right side
            gaze = np.array([0.4, 0.0, 1.0, 0.9, 0.5], dtype=np.float32)
            target = 1
        else:
            # Distracted/sleep gaze looking down inside cabin
            gaze = np.array([0.0, -0.8, 0.2, 0.5, 0.95], dtype=np.float32)
            nodes[0, 7] = 1.0  # Imminent TTC
            target = 2

        norm = np.linalg.norm(gaze[:3])
        gaze[:3] /= max(norm, 1e-4)

        return {
            "gaze": torch.from_numpy(gaze).unsqueeze(0),
            "nodes": torch.from_numpy(nodes),
            "mask": torch.ones(n, dtype=torch.bool),
            "target_action": target,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_synthetic_dual_stream.py -v`  
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add sg_ttrans/data/synthetic_dual_stream.py tests/test_synthetic_dual_stream.py
git commit -m "feat(data): implement synthetic dual-stream dataset generator"
```

---

### Task 8: End-to-End Pipeline Verification, Hardware Profiling & Interactive Demo

**Files:**
- Create: `tests/test_end_to_end_st_hgst.py`
- Modify: `benchmark.py`
- Create: `demo_dual_stream.py`

**Interfaces:**
- Consumes: All components from Tasks 1–7.
- Produces: Validated end-to-end forward pass, GPU latency profiling report, live interactive visual stream.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_end_to_end_st_hgst.py
import pytest
import torch
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.models.st_hgst_net import STHGSTNetwork
from sg_ttrans.cognitive.leaky_accumulator import LeakyCognitiveAccumulator
from sg_ttrans.cognitive.epistemic_gap import compute_eag
from sg_ttrans.cognitive.takeover_arbitrator import TakeoverArbitrator
from sg_ttrans.types import TakeoverAction

def test_full_pipeline_flow():
    cfg = STHGSTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STHGSTNetwork(embed_dim=128, num_heads=4).to(device)
    accumulator = LeakyCognitiveAccumulator(cfg)
    arbitrator = TakeoverArbitrator(cfg)

    # 1. Inputs
    gaze = torch.tensor([[[0.0, 0.0, 1.0, 0.5, 0.5]]], device=device)
    nodes = torch.tensor([[[0.0, 0.5, 0.5, 0.1, 0.2, 0.0, -4.0, 2.0],
                           [1.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 10.0]]], device=device)
    
    # 2. Model inference
    with torch.no_grad():
        out = model(gaze, nodes)
    attn_weights = out["attention_weights"].squeeze(0).cpu().numpy()
    
    # 3. Cognitive accumulation
    cog_state = accumulator.step(attn_weights)
    
    # 4. EAG & Takeover Decision
    hazard_weights = np.array([0.8, 0.05])
    eag, top_hazard = compute_eag(hazard_weights, cog_state.accumulated_cognition)
    decision = arbitrator.arbitrate(eag, min_ttc=2.0, top_hazard_idx=top_hazard)
    
    assert decision.action in (TakeoverAction.SAFE_HANDOVER, TakeoverAction.TARGETED_SPATIAL_CUE, TakeoverAction.MINIMUM_RISK_MANEUVER)
    assert 0.0 <= decision.eag <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_end_to_end_st_hgst.py -v`  
Expected: Must fail or pass depending on device handling.

- [ ] **Step 3: Implement benchmark profiler & visual demo script**

Update `benchmark.py` to profile per-module latency on GPU:
```python
# benchmark.py
import time
import torch
import numpy as np
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.models.st_hgst_net import STHGSTNetwork
from sg_ttrans.cognitive.leaky_accumulator import LeakyCognitiveAccumulator
from sg_ttrans.cognitive.epistemic_gap import compute_eag
from sg_ttrans.cognitive.takeover_arbitrator import TakeoverArbitrator

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Benchmarking ST-HGST on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) ===")
    
    cfg = STHGSTConfig()
    model = STHGSTNetwork().to(device)
    model.eval()
    accumulator = LeakyCognitiveAccumulator(cfg)
    arbitrator = TakeoverArbitrator(cfg)

    gaze = torch.randn(1, 1, 5, device=device)
    nodes = torch.randn(1, 10, 8, device=device)
    hazard_weights = np.random.uniform(0, 1, size=(10,)).astype(np.float32)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(gaze, nodes)

    iterations = 500
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(iterations):
        with torch.no_grad():
            out = model(gaze, nodes)
        attn = out["attention_weights"].squeeze(0).cpu().numpy()
        cog = accumulator.step(attn)
        eag, top_h = compute_eag(hazard_weights, cog.accumulated_cognition)
        _ = arbitrator.arbitrate(eag, min_ttc=2.0, top_hazard_idx=top_h)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0
    avg_latency = (total_time / iterations) * 1000.0
    fps = 1000.0 / avg_latency

    print(f"Average Pipeline Latency: {avg_latency:.2f} ms")
    print(f"Effective Throughput:     {fps:.1f} FPS")
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU VRAM Allocated:  {mem:.1f} MB")

if __name__ == "__main__":
    benchmark()
```

- [ ] **Step 4: Run tests and benchmark**

Run:
```bash
/home/roshanbinoj/Documents/BTP/venv/bin/pytest tests/test_end_to_end_st_hgst.py -v
/home/roshanbinoj/Documents/BTP/venv/bin/python benchmark.py
```
Expected: All tests PASS and benchmark reports > 60 FPS on GPU.

- [ ] **Step 5: Commit**

```bash
git add tests/test_end_to_end_st_hgst.py benchmark.py
git commit -m "feat(pipeline): complete end-to-end ST-HGST pipeline and hardware profiler"
```

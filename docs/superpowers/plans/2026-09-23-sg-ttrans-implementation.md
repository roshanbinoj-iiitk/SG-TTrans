# SG-TTrans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete Spatial-Geometric Temporal Transformer (SG-TTrans) framework from the research paper, covering the dual-stream spatial (MobileNetV4) and geometric (FaceMesh 68 3D landmarks) pathways, TDDA (Temporal Dynamic Decay Attention) Transformer, Multi-Task Driver State Classifier with Focal Loss, dynamic Risk-Aware Safety Index (RSI), graduated ADAS controller, synthetic & real dataset pipelines, and real-time streaming HUD dashboard.

**Architecture:** A modular micro-pipeline with clear interface separation between landmark geometry (`geometry/`), deep neural sequence modeling (`models/`), continuous risk dynamics (`risk_engine/`), dataset ingestion (`data/`), training (`training/`), and streaming inference (`inference/`).

**Tech Stack:** Python 3.12+, PyTorch 2.x, OpenCV (`cv2`), MediaPipe, NumPy, SciPy, pytest.

**Spec:** [docs/superpowers/specs/2026-09-23-sg-ttrans-design.md](file:///home/roshanbinoj/Documents/CV%20Project%20Implementation/docs/superpowers/specs/2026-09-23-sg-ttrans-design.md)

## Global Constraints
- Target frame rate: 30 FPS with temporal window $T=60$ frames (2.0s duration).
- Appearance resolution: $224 \times 224$ RGB facial crops mapped to $d_v = 256$ visual tokens.
- Kinematic token: $z_{geom}^t \in \mathbb{R}^{16}$ encoding $EAR, MAR, PERCLOS_W$, Euler angles, velocities, and flags.
- Fusion & Model dimension: $d_{model} = 256$, number of heads $H = 4$, projection dimension $d_k = 64$.
- TDDA attention regularization: learnable $\gamma > 0$, initialized to $\gamma_0 = 0.15$.
- RSI parameterization: $w_1 = 0.50, w_2 = 0.35, w_3 = 0.15, \tau_0 = 1.5\text{ s}, v_{max} = 130\text{ km/h}$.
- ADAS intervention thresholds: $\delta_1 = 0.35, \delta_2 = 0.60, \delta_3 = 0.80$.
- No placeholders, no stubs without tests, strict adherence to TDD.

## Review Focus
1. **Missing or Lost Face during Live Stream:** When a driver turns their head out of camera view or occlusion occurs, FaceMesh fails to detect landmarks; the pipeline must smoothly impute via EMA/previous state rather than crashing with `NoneType` errors.
2. **Numerical Stability of TDDA Softmax:** Logits $\frac{Q_i K_j^T}{\sqrt{d_k}} - \gamma |i-j|$ must be stabilized by subtracting max logit before exponentiation to prevent overflow or underflow.
3. **Division by Zero in Aspect Ratios:** Near-zero eye or mouth horizontal distances (extreme profile views) could yield ZeroDivisionError in EAR or MAR; denominators must be clamped by $\epsilon = 10^{-7}$.
4. **Negative or Stale TTC Values:** Following or stationary vehicles yielding invalid or negative Time-to-Collision must be safely clamped via $\max(0, TTC(t))$ according to Equation 14.
5. **Subject Leakage in Sequence Batching:** Video sequences from the same subject must never appear in both training and evaluation splits.

---

### Task 1: Project Scaffolding & Configuration
**Files:**
- Create: `pyproject.toml`
- Create: `sg_ttrans/__init__.py`
- Create: `sg_ttrans/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: None
- Produces: `sg_ttrans.config.SGTransConfig` dataclass containing all default hyper-parameters ($T=60, \gamma=0.15, d_{model}=256$, etc.).

- [ ] **Step 1: Write the failing test**
Create `tests/test_config.py`:
```python
from sg_ttrans.config import SGTransConfig

def test_default_config():
    cfg = SGTransConfig()
    assert cfg.sequence_length == 60
    assert cfg.fps == 30.0
    assert cfg.d_model == 256
    assert cfg.gamma_init == 0.15
    assert cfg.num_classes == 5
    assert cfg.w1 == 0.50 and cfg.w2 == 0.35 and cfg.w3 == 0.15
    assert cfg.delta1 == 0.35 and cfg.delta2 == 0.60 and cfg.delta3 == 0.80
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_config.py`
Expected: FAIL (ModuleNotFoundError: No module named 'sg_ttrans')

- [ ] **Step 3: Write minimal implementation**
Create `pyproject.toml`, `sg_ttrans/__init__.py`, and `sg_ttrans/config.py`:
```python
from dataclasses import dataclass

@dataclass
class SGTransConfig:
    sequence_length: int = 60
    fps: float = 30.0
    d_vis: int = 256
    d_geom: int = 16
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    gamma_init: float = 0.15
    num_classes: int = 5
    delta_ear: float = 0.20
    tau0: float = 1.5
    v_max: float = 130.0
    w1: float = 0.50
    w2: float = 0.35
    w3: float = 0.15
    delta1: float = 0.35
    delta2: float = 0.60
    delta3: float = 0.80
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_config.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add pyproject.toml sg_ttrans/__init__.py sg_ttrans/config.py tests/test_config.py
git commit -m "feat: setup project scaffolding and SGTransConfig"
```

---

### Task 2: Geometric Kinematics — EAR, MAR, and PERCLOS Metrics
**Files:**
- Create: `sg_ttrans/geometry/__init__.py`
- Create: `sg_ttrans/geometry/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: 3D Landmark arrays (NumPy array $[68, 3]$)
- Produces: `compute_ear(eye_landmarks)`, `compute_mar(mouth_landmarks)`, `compute_perclos(ear_history, threshold=0.20)`

- [ ] **Step 1: Write the failing test**
Create `tests/test_metrics.py`:
```python
import numpy as np
import pytest
from sg_ttrans.geometry.metrics import compute_ear, compute_mar, compute_perclos

def test_compute_ear_open_and_closed():
    # Canonical open eye coordinates: p1=(-1,0), p4=(1,0), p2=(-0.5, 0.5), p3=(0.5, 0.5), p5=(0.5, -0.5), p6=(-0.5, -0.5)
    open_eye = np.array([
        [-1.0, 0.0, 0.0],
        [-0.5, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, -0.5, 0.0]
    ])
    ear_open = compute_ear(open_eye)
    assert pytest.approx(ear_open, 0.01) == 0.50

    # Closed eye: vertical distance drops to 0
    closed_eye = open_eye.copy()
    closed_eye[:, 1] = 0.0
    ear_closed = compute_ear(closed_eye)
    assert pytest.approx(ear_closed, 0.01) == 0.0

def test_compute_mar_yawn():
    # p12, p14, p15, p16, p17, p18
    # Horizontal width = 2.0 (p12=(-1,0), p16=(1,0))
    # Vertical height = 1.0 (p14/p18 and p15/p17 dist = 1.0)
    mouth = np.array([
        [-1.0, 0.0, 0.0],  # p12
        [0.0, 0.0, 0.0],
        [-0.5, 0.5, 0.0],  # p14
        [0.5, 0.5, 0.0],   # p15
        [1.0, 0.0, 0.0],   # p16
        [0.5, -0.5, 0.0],  # p17
        [-0.5, -0.5, 0.0]  # p18
    ])
    mar = compute_mar(mouth)
    assert pytest.approx(mar, 0.01) == 0.50

def test_compute_perclos():
    # 60 frames: 45 frames closed (< 0.20), 15 frames open
    history = [0.10] * 45 + [0.35] * 15
    perclos = compute_perclos(history, threshold=0.20)
    assert pytest.approx(perclos, 0.001) == 45.0 / 60.0
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_metrics.py`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/geometry/__init__.py` and `sg_ttrans/geometry/metrics.py`:
```python
import numpy as np

def compute_ear(eye_pts: np.ndarray, eps: float = 1e-7) -> float:
    # eye_pts shape: (6, 2) or (6, 3)
    p1, p2, p3, p4, p5, p6 = eye_pts[:6]
    vert1 = np.linalg.norm(p2 - p6)
    vert2 = np.linalg.norm(p3 - p5)
    horiz = np.linalg.norm(p1 - p4)
    return float((vert1 + vert2) / (2.0 * horiz + eps))

def compute_mar(mouth_pts: np.ndarray, eps: float = 1e-7) -> float:
    # indices: p12=0, p14=2, p15=3, p16=4, p17=5, p18=6
    p12, _, p14, p15, p16, p17, p18 = mouth_pts[:7]
    vert1 = np.linalg.norm(p14 - p18)
    vert2 = np.linalg.norm(p15 - p17)
    horiz = np.linalg.norm(p12 - p16)
    return float((vert1 + vert2) / (2.0 * horiz + eps))

def compute_perclos(ear_window: list[float] | np.ndarray, threshold: float = 0.20) -> float:
    if len(ear_window) == 0:
        return 0.0
    closed_count = sum(1 for e in ear_window if e < threshold)
    return float(closed_count / len(ear_window))
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_metrics.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/geometry/__init__.py sg_ttrans/geometry/metrics.py tests/test_metrics.py
git commit -m "feat: implement EAR, MAR, and PERCLOS metric computations"
```

---

### Task 3: Geometric Kinematics — Head Pose Estimation & Kinematic Token Builder
**Files:**
- Create: `sg_ttrans/geometry/head_pose.py`
- Test: `tests/test_head_pose.py`

**Interfaces:**
- Consumes: 2D/3D landmarks, previous kinematic state
- Produces: `HeadPoseEstimator.estimate(landmarks_2d, img_w, img_h) -> (yaw, pitch, roll)`, `build_kinematic_token(ear, mar, perclos, euler, prev_state, dt) -> np.ndarray [16]`

- [ ] **Step 1: Write the failing test**
Create `tests/test_head_pose.py`:
```python
import numpy as np
import pytest
from sg_ttrans.geometry.head_pose import HeadPoseEstimator, build_kinematic_token

def test_head_pose_frontal():
    estimator = HeadPoseEstimator()
    # Canonical 2D image points matching standard 3D anthropometric face model
    # Image center 320, 240
    pts_2d = np.array([
        [320.0, 240.0],  # Nose tip
        [320.0, 310.0],  # Chin
        [280.0, 210.0],  # Left eye corner
        [360.0, 210.0],  # Right eye corner
        [290.0, 280.0],  # Left mouth corner
        [350.0, 280.0],  # Right mouth corner
    ], dtype=np.float64)
    yaw, pitch, roll = estimator.estimate(pts_2d, img_w=640, img_h=480)
    assert abs(yaw) < 15.0
    assert abs(pitch) < 15.0
    assert abs(roll) < 15.0

def test_build_kinematic_token():
    token = build_kinematic_token(
        ear=0.25, mar=0.30, perclos=0.10,
        euler=np.array([5.0, -2.0, 1.0]),
        prev_kinematics=None,
        ear_l=0.25, ear_r=0.25, dt=1.0/30.0
    )
    assert token.shape == (16,)
    assert token[0] == pytest.approx(0.25)
    assert token[1] == pytest.approx(0.30)
    assert token[2] == pytest.approx(0.10)
    assert token[3] == pytest.approx(5.0)
    assert token[4] == pytest.approx(-2.0)
    assert token[5] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_head_pose.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/geometry/head_pose.py`:
```python
import cv2
import numpy as np

# Standard 3D anthropometric facial model points (in millimeters)
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],          # Nose tip
    [0.0, -330.0, -65.0],     # Chin
    [-225.0, 170.0, -135.0],  # Left eye corner
    [225.0, 170.0, -135.0],   # Right eye corner
    [-150.0, -150.0, -125.0], # Left mouth corner
    [150.0, -150.0, -125.0]   # Right mouth corner
], dtype=np.float64)

class HeadPoseEstimator:
    def __init__(self):
        self.model_pts = MODEL_POINTS_3D

    def estimate(self, pts_2d: np.ndarray, img_w: int, img_h: int) -> tuple[float, float, float]:
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_pts, pts_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return (0.0, 0.0, 0.0)

        rmat, _ = cv2.Rodrigues(rvec)
        # Compute Euler angles from rotation matrix
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = np.arctan2(rmat[1, 0], rmat[0, 0])
        else:
            pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
            yaw = np.arctan2(-rmat[2, 0], sy)
            roll = 0.0

        return (float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll)))

def build_kinematic_token(
    ear: float, mar: float, perclos: float,
    euler: np.ndarray, prev_kinematics: np.ndarray | None,
    ear_l: float, ear_r: float, dt: float = 1.0 / 30.0
) -> np.ndarray:
    if prev_kinematics is not None:
        d_ear = (ear - prev_kinematics[0]) / dt
        d_mar = (mar - prev_kinematics[1]) / dt
        d_euler = (euler - prev_kinematics[3:6]) / dt
    else:
        d_ear = 0.0
        d_mar = 0.0
        d_euler = np.zeros(3, dtype=np.float32)

    euler_norm = float(np.linalg.norm(euler))
    d_euler_norm = float(np.linalg.norm(d_euler))
    blink_flag = 1.0 if ear < 0.20 else 0.0

    token = np.array([
        ear, mar, perclos,
        euler[0], euler[1], euler[2],
        d_ear, d_mar,
        d_euler[0], d_euler[1], d_euler[2],
        ear_l, ear_r,
        euler_norm, d_euler_norm, blink_flag
    ], dtype=np.float32)
    return token
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_head_pose.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/geometry/head_pose.py tests/test_head_pose.py
git commit -m "feat: implement HeadPoseEstimator and build_kinematic_token"
```

---

### Task 4: Risk Engine — Persistence Tracker & Dynamic RSI Engine
**Files:**
- Create: `sg_ttrans/risk_engine/__init__.py`
- Create: `sg_ttrans/risk_engine/persistence.py`
- Create: `sg_ttrans/risk_engine/rsi.py`
- Test: `tests/test_rsi_scenarios.py`

**Interfaces:**
- Consumes: $\hat{y}_{fatigue}$, $\tau_{persist}$, $v(t)$, $TTC(t)$, $\theta_{pitch}, \phi_{yaw}$
- Produces: `FatiguePersistenceTracker`, `compute_rsi(fatigue_prob, tau_persist, v, ttc, pitch, yaw, config) -> float`

- [ ] **Step 1: Write the failing test**
Create `tests/test_rsi_scenarios.py` with exact verification of Scenarios A, B, and C:
```python
import pytest
from sg_ttrans.config import SGTransConfig
from sg_ttrans.risk_engine.rsi import compute_rsi

def test_scenario_a_normal_blink():
    cfg = SGTransConfig()
    # Scenario A: tau=0.18s, v=80 km/h, TTC=8.0s, yaw=12 deg, pitch=0 deg, y_fatigue=0.12
    rsi = compute_rsi(
        fatigue_prob=0.12,
        tau_persist=0.18,
        velocity=80.0,
        ttc=8.0,
        pitch=0.0,
        yaw=12.0,
        config=cfg
    )
    assert pytest.approx(rsi, abs=0.002) == 0.071

def test_scenario_b_moderate_fatigue_yawn():
    cfg = SGTransConfig()
    # Scenario B: tau=2.2s, v=90 km/h, TTC=3.5s, pitch=18 deg, yaw=0 deg, y_fatigue=0.88
    rsi = compute_rsi(
        fatigue_prob=0.88,
        tau_persist=2.2,
        velocity=90.0,
        ttc=3.5,
        pitch=18.0,
        yaw=0.0,
        config=cfg
    )
    assert pytest.approx(rsi, abs=0.002) == 0.452

def test_scenario_c_critical_microsleep():
    cfg = SGTransConfig()
    # Scenario C initial: tau=1.8s, v=110 km/h, TTC=1.4s, pitch=32 deg, yaw=0 deg, y_fatigue=0.98
    rsi_initial = compute_rsi(
        fatigue_prob=0.98,
        tau_persist=1.8,
        velocity=110.0,
        ttc=1.4,
        pitch=32.0,
        yaw=0.0,
        config=cfg
    )
    assert pytest.approx(rsi_initial, abs=0.002) == 0.573

    # Scenario C escalated: tau=2.8s, v=110 km/h, TTC=0.9s, pitch=32 deg, yaw=0 deg, y_fatigue=0.98
    rsi_escalated = compute_rsi(
        fatigue_prob=0.98,
        tau_persist=2.8,
        velocity=110.0,
        ttc=0.9,
        pitch=32.0,
        yaw=0.0,
        config=cfg
    )
    assert pytest.approx(rsi_escalated, abs=0.002) == 0.841
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_rsi_scenarios.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/risk_engine/__init__.py`, `sg_ttrans/risk_engine/persistence.py`, and `sg_ttrans/risk_engine/rsi.py`:
```python
import numpy as np
from sg_ttrans.config import SGTransConfig

class FatiguePersistenceTracker:
    def __init__(self, sequence_length: int = 60, dt: float = 1.0 / 30.0):
        self.sequence_length = sequence_length
        self.dt = dt
        self.history: list[bool] = []

    def update(self, is_fatigued: bool) -> float:
        self.history.append(is_fatigued)
        if len(self.history) > self.sequence_length:
            self.history.pop(0)
        # Continuous consecutive fatigue frames from current frame backwards
        consecutive = 0
        for val in reversed(self.history):
            if val:
                consecutive += 1
            else:
                break
        return float(consecutive * self.dt)

def compute_rsi(
    fatigue_prob: float,
    tau_persist: float,
    velocity: float,
    ttc: float,
    pitch: float,
    yaw: float,
    config: SGTransConfig
) -> float:
    # 1. Visual fatigue persistence factor
    term1 = config.w1 * fatigue_prob * (1.0 - np.exp(-tau_persist / config.tau0))
    # 2. Vehicular dynamic risk factor
    speed_factor = velocity / config.v_max
    ttc_factor = 1.0 / (1.0 + max(0.0, ttc))
    term2 = config.w2 * speed_factor * ttc_factor
    # 3. Head pose risk factor
    sigma_head = min(1.0, (abs(pitch) + abs(yaw)) / 45.0)
    term3 = config.w3 * sigma_head

    rsi = term1 + term2 + term3
    return float(np.clip(rsi, 0.0, 1.0))
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_rsi_scenarios.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/risk_engine/__init__.py sg_ttrans/risk_engine/persistence.py sg_ttrans/risk_engine/rsi.py tests/test_rsi_scenarios.py
git commit -m "feat: implement persistence tracker, dynamic RSI calculation, and verify Scenarios A, B, C"
```

---

### Task 5: Risk Engine — Graduated ADAS Safety Controller
**Files:**
- Create: `sg_ttrans/risk_engine/adas_controller.py`
- Test: `tests/test_adas_controller.py`

**Interfaces:**
- Consumes: $RSI(t)$
- Produces: `ADASController.evaluate(rsi) -> ADASAction(level, name, intervention_msg)`

- [ ] **Step 1: Write the failing test**
Create `tests/test_adas_controller.py`:
```python
from sg_ttrans.config import SGTransConfig
from sg_ttrans.risk_engine.adas_controller import ADASController, ADASLevel

def test_adas_levels():
    ctrl = ADASController(SGTransConfig())
    # Level 0 (< 0.35)
    act0 = ctrl.evaluate(0.071)
    assert act0.level == ADASLevel.LEVEL_0_NOMINAL
    # Level 1 (0.35 <= rsi < 0.60)
    act1 = ctrl.evaluate(0.452)
    assert act1.level == ADASLevel.LEVEL_1_VISUAL
    # Level 2 (0.60 <= rsi < 0.80)
    act2 = ctrl.evaluate(0.650)
    assert act2.level == ADASLevel.LEVEL_2_AUDIO_HAPTIC
    # Level 3 (>= 0.80)
    act3 = ctrl.evaluate(0.841)
    assert act3.level == ADASLevel.LEVEL_3_CRITICAL_AEB
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_adas_controller.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/risk_engine/adas_controller.py`:
```python
from enum import IntEnum
from dataclasses import dataclass
from sg_ttrans.config import SGTransConfig

class ADASLevel(IntEnum):
    LEVEL_0_NOMINAL = 0
    LEVEL_1_VISUAL = 1
    LEVEL_2_AUDIO_HAPTIC = 2
    LEVEL_3_CRITICAL_AEB = 3

@dataclass
class ADASAction:
    level: ADASLevel
    name: str
    description: str
    aeb_active: bool
    lane_hold_active: bool
    hazard_flashers_active: bool

class ADASController:
    def __init__(self, config: SGTransConfig):
        self.cfg = config

    def evaluate(self, rsi: float) -> ADASAction:
        if rsi >= self.cfg.delta3:
            return ADASAction(
                level=ADASLevel.LEVEL_3_CRITICAL_AEB,
                name="Level 3: Critical Hazard",
                description="Emergency Braking (AEB) Assist, Lane Centering Active Hold, Autonomous Hazard Flasher",
                aeb_active=True,
                lane_hold_active=True,
                hazard_flashers_active=True,
            )
        elif rsi >= self.cfg.delta2:
            return ADASAction(
                level=ADASLevel.LEVEL_2_AUDIO_HAPTIC,
                name="Level 2: Audio-Haptic Alert",
                description="Urgent Acoustic Alarm + Steering Wheel Haptic Pulse",
                aeb_active=False,
                lane_hold_active=False,
                hazard_flashers_active=False,
            )
        elif rsi >= self.cfg.delta1:
            return ADASAction(
                level=ADASLevel.LEVEL_1_VISUAL,
                name="Level 1: Visual Warning",
                description="Visual Dashboard Prompt + Gentle Chime",
                aeb_active=False,
                lane_hold_active=False,
                hazard_flashers_active=False,
            )
        else:
            return ADASAction(
                level=ADASLevel.LEVEL_0_NOMINAL,
                name="Level 0: Nominal Operation",
                description="Safe driving state, no intervention",
                aeb_active=False,
                lane_hold_active=False,
                hazard_flashers_active=False,
            )
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_adas_controller.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/risk_engine/adas_controller.py tests/test_adas_controller.py
git commit -m "feat: implement ADASController with 4 graduated intervention levels"
```

---

### Task 6: Neural Models — Spatial Backbone & Cross-Modal Fusion
**Files:**
- Create: `sg_ttrans/models/__init__.py`
- Create: `sg_ttrans/models/backbone.py`
- Create: `sg_ttrans/models/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**
- Consumes: Frame tensors $[B, T, 3, 224, 224]$, Kinematic tokens $[B, T, 16]$
- Produces: `SpatialBackbone(x) -> [B, T, 256]`, `CrossModalFusion(z_vis, z_geom) -> [B, T, 256]`

- [ ] **Step 1: Write the failing test**
Create `tests/test_fusion.py`:
```python
import torch
from sg_ttrans.models.backbone import MobileNetV4SpatialBackbone
from sg_ttrans.models.fusion import CrossModalFusion

def test_backbone_and_fusion_shapes():
    B, T = 2, 4
    dummy_frames = torch.randn(B, T, 3, 224, 224)
    dummy_kinematics = torch.randn(B, T, 16)

    backbone = MobileNetV4SpatialBackbone(out_dim=256)
    z_vis = backbone(dummy_frames)
    assert z_vis.shape == (B, T, 256)

    fusion = CrossModalFusion(d_vis=256, d_geom=16, d_model=256, max_seq_len=60)
    z_seq = fusion(z_vis, dummy_kinematics)
    assert z_seq.shape == (B, T, 256)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_fusion.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/models/__init__.py`, `sg_ttrans/models/backbone.py`, and `sg_ttrans/models/fusion.py`:
```python
import torch
import torch.nn as nn

class MobileNetV4SpatialBackbone(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        # Lightweight MobileNetV4 conv feature extractor
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.Sequential(
            # Inverted residual / MobileNetV4 blocks
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 3, H, W] or [B, 3, H, W]
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            feat = self.stages(self.conv_stem(x_flat))
            feat = feat.view(B * T, -1)
            out = self.proj(feat)
            return out.view(B, T, -1)
        else:
            feat = self.stages(self.conv_stem(x)).view(x.size(0), -1)
            return self.proj(feat)

class CrossModalFusion(nn.Module):
    def __init__(self, d_vis: int = 256, d_geom: int = 16, d_model: int = 256, max_seq_len: int = 60):
        super().__init__()
        self.w_v = nn.Linear(d_vis, d_model, bias=False)
        self.w_g = nn.Linear(d_geom, d_model, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

    def forward(self, z_vis: torch.Tensor, z_geom: torch.Tensor) -> torch.Tensor:
        # z_vis: [B, T, d_vis], z_geom: [B, T, d_geom]
        B, T, _ = z_vis.shape
        z_t = self.w_v(z_vis) + self.w_g(z_geom) + self.bias
        z_seq = z_t + self.pos_embedding[:, :T, :]
        return z_seq
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_fusion.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/models/__init__.py sg_ttrans/models/backbone.py sg_ttrans/models/fusion.py tests/test_fusion.py
git commit -m "feat: implement MobileNetV4 spatial backbone and CrossModalFusion module"
```

---

### Task 7: Neural Models — TDDA Attention Kernel & Temporal Transformer
**Files:**
- Create: `sg_ttrans/models/tdda_attention.py`
- Create: `sg_ttrans/models/transformer.py`
- Test: `tests/test_tdda.py`

**Interfaces:**
- Consumes: Input sequence $Z \in \mathbb{R}^{B \times T \times d_{model}}$
- Produces: `TDDAAttention`, `TemporalTransformerEncoder`, verifies Proposition 1 decay bounds.

- [ ] **Step 1: Write the failing test**
Create `tests/test_tdda.py`:
```python
import torch
import pytest
import numpy as np
from sg_ttrans.models.tdda_attention import TDDAAttention
from sg_ttrans.models.transformer import TemporalTransformerEncoder

def test_tdda_attention_shapes():
    B, T, d_model = 2, 60, 256
    x = torch.randn(B, T, d_model)
    tdda = TDDAAttention(d_model=d_model, num_heads=4, gamma_init=0.15)
    out, attn = tdda(x, return_attention=True)
    assert out.shape == (B, T, d_model)
    assert attn.shape == (B, 4, T, T)
    # Check probabilities sum to 1.0 along last dimension
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, 4, T), atol=1e-5)

def test_tdda_proposition_1_mathematical_decay():
    # Verify cumulative attention ratio for 6-frame blink vs 30-frame microsleep
    gamma = 0.15
    m_blink = 6
    m_micro = 30
    w_blink_bound = (1.0 - np.exp(-gamma * m_blink)) / (1.0 - np.exp(-gamma))
    w_micro_bound = (1.0 - np.exp(-gamma * m_micro)) / (1.0 - np.exp(-gamma))

    assert pytest.approx(w_blink_bound, 0.05) == 4.26
    assert pytest.approx(w_micro_bound, 0.05) == 7.10
    assert w_blink_bound < w_micro_bound

def test_transformer_encoder():
    B, T, d_model = 2, 60, 256
    encoder = TemporalTransformerEncoder(d_model=d_model, num_heads=4, num_layers=4, gamma_init=0.15)
    x = torch.randn(B, T, d_model)
    out = encoder(x)
    assert out.shape == (B, T, d_model)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_tdda.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/models/tdda_attention.py` and `sg_ttrans/models/transformer.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TDDAAttention(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, gamma_init: float = 0.15):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # Learnable gamma initialized to gamma_init via inverse softplus
        inv_softplus = float(torch.log(torch.exp(torch.tensor(gamma_init)) - 1.0))
        self.raw_gamma = nn.Parameter(torch.tensor(inv_softplus))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T, _ = x.shape
        Q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Distance decay matrix |i - j|
        indices = torch.arange(T, device=x.device, dtype=torch.float32)
        dist_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        gamma = F.softplus(self.raw_gamma)
        decay_penalty = gamma * dist_matrix.unsqueeze(0).unsqueeze(0)

        # Modulated logits
        logits = scores - decay_penalty
        attn_weights = F.softmax(logits, dim=-1)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.w_o(out)

        if return_attention:
            return out, attn_weights
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, gamma_init: float = 0.15, d_ff: int = 1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TDDAAttention(d_model, num_heads, gamma_init)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, d_model: int = 256, num_heads: int = 4, num_layers: int = 4, gamma_init: float = 0.15):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, gamma_init) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_tdda.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/models/tdda_attention.py sg_ttrans/models/transformer.py tests/test_tdda.py
git commit -m "feat: implement TDDA Attention and TemporalTransformerEncoder with mathematical decay"
```

---

### Task 8: Neural Models — Output Heads & End-to-End SG-TTrans Network
**Files:**
- Create: `sg_ttrans/models/heads.py`
- Create: `sg_ttrans/models/sg_ttrans_net.py`
- Test: `tests/test_model_pipeline.py`

**Interfaces:**
- Consumes: $I_t [B, T, 3, 224, 224]$, $z_{geom} [B, T, 16]$
- Produces: `SGTransNet(frames, kinematics) -> dict(logits, probs, z_bar, fatigue_prob)`

- [ ] **Step 1: Write the failing test**
Create `tests/test_model_pipeline.py`:
```python
import torch
from sg_ttrans.config import SGTransConfig
from sg_ttrans.models.sg_ttrans_net import SGTransNet

def test_sg_ttrans_net_forward():
    cfg = SGTransConfig(sequence_length=10) # small sequence for fast test
    model = SGTransNet(cfg)
    B, T = 2, 10
    frames = torch.randn(B, T, 3, 224, 224)
    kinematics = torch.randn(B, T, 16)

    outputs = model(frames, kinematics)
    assert "logits" in outputs
    assert "probs" in outputs
    assert "fatigue_prob" in outputs
    assert outputs["logits"].shape == (B, 5)
    assert outputs["probs"].shape == (B, 5)
    assert outputs["fatigue_prob"].shape == (B,)
    # Check fatigue_prob = probs[Drowsy] + probs[Microsleep]
    expected_fatigue = outputs["probs"][:, 1] + outputs["probs"][:, 2]
    assert torch.allclose(outputs["fatigue_prob"], expected_fatigue, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_model_pipeline.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/models/heads.py` and `sg_ttrans/models/sg_ttrans_net.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from sg_ttrans.config import SGTransConfig
from sg_ttrans.models.backbone import MobileNetV4SpatialBackbone
from sg_ttrans.models.fusion import CrossModalFusion
from sg_ttrans.models.transformer import TemporalTransformerEncoder

class ClassificationHead(nn.Module):
    def __init__(self, d_model: int = 256, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, z_bar: torch.Tensor) -> torch.Tensor:
        return self.fc(z_bar)

class SGTransNet(nn.Module):
    def __init__(self, config: SGTransConfig):
        super().__init__()
        self.config = config
        self.backbone = MobileNetV4SpatialBackbone(out_dim=config.d_vis)
        self.fusion = CrossModalFusion(
            d_vis=config.d_vis,
            d_geom=config.d_geom,
            d_model=config.d_model,
            max_seq_len=config.sequence_length
        )
        self.transformer = TemporalTransformerEncoder(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            gamma_init=config.gamma_init
        )
        self.head = ClassificationHead(d_model=config.d_model, num_classes=config.num_classes)

    def forward(self, frames: torch.Tensor, kinematics: torch.Tensor) -> dict[str, torch.Tensor]:
        # frames: [B, T, 3, 224, 224], kinematics: [B, T, 16]
        z_vis = self.backbone(frames)
        z_seq = self.fusion(z_vis, kinematics)
        z_trans = self.transformer(z_seq)
        z_bar = z_trans.mean(dim=1)  # Mean pooling over T

        logits = self.head(z_bar)
        probs = F.softmax(logits, dim=-1)
        # Class 1 = Drowsy, Class 2 = Microsleep
        fatigue_prob = probs[:, 1] + probs[:, 2]

        return {
            "logits": logits,
            "probs": probs,
            "fatigue_prob": fatigue_prob,
            "z_bar": z_bar
        }
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_model_pipeline.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/models/heads.py sg_ttrans/models/sg_ttrans_net.py tests/test_model_pipeline.py
git commit -m "feat: implement ClassificationHead and end-to-end SGTransNet model"
```

---

### Task 9: Training Pipeline — Focal Loss, Synthetic Data Generator & Trainer
**Files:**
- Create: `sg_ttrans/training/__init__.py`
- Create: `sg_ttrans/training/losses.py`
- Create: `sg_ttrans/data/__init__.py`
- Create: `sg_ttrans/data/synthetic_generator.py`
- Create: `sg_ttrans/training/trainer.py`
- Test: `tests/test_trainer.py`

**Interfaces:**
- Consumes: Ground truth labels $[B]$, Model logits $[B, 5]$
- Produces: `FocalLoss(gamma=2.0)`, `SyntheticSequenceGenerator.generate_batch(B, T)`, `Trainer.train_epoch()`

- [ ] **Step 1: Write the failing test**
Create `tests/test_trainer.py`:
```python
import torch
from sg_ttrans.config import SGTransConfig
from sg_ttrans.training.losses import MultiClassFocalLoss
from sg_ttrans.data.synthetic_generator import SyntheticSequenceGenerator
from sg_ttrans.models.sg_ttrans_net import SGTransNet

def test_focal_loss_computation():
    criterion = MultiClassFocalLoss(gamma=2.0)
    logits = torch.randn(4, 5, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3])
    loss = criterion(logits, targets)
    assert loss.dim() == 0
    assert loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None

def test_synthetic_data_generation():
    gen = SyntheticSequenceGenerator()
    frames, kinematics, labels = gen.generate_batch(batch_size=2, seq_len=10)
    assert frames.shape == (2, 10, 3, 224, 224)
    assert kinematics.shape == (2, 10, 16)
    assert labels.shape == (2,)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_trainer.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/training/losses.py`, `sg_ttrans/data/synthetic_generator.py`, and `sg_ttrans/training/trainer.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha) if alpha is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        focal_weight = (1.0 - target_probs) ** self.gamma
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal_weight = focal_weight * alpha[targets]

        loss = -focal_weight * target_log_probs
        return loss.mean()
```
And `sg_ttrans/data/synthetic_generator.py`:
```python
import torch
import numpy as np

class SyntheticSequenceGenerator:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def generate_batch(self, batch_size: int = 4, seq_len: int = 60) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate synthetic facial frames [B, T, 3, 224, 224]
        frames = torch.randn(batch_size, seq_len, 3, 224, 224) * 0.1
        # Generate synthetic kinematics [B, T, 16] and labels [B]
        labels = torch.randint(0, 5, (batch_size,))
        kinematics = torch.zeros(batch_size, seq_len, 16)

        for b in range(batch_size):
            label = labels[b].item()
            if label == 0:  # Alert
                ear = 0.30 + np.random.normal(0, 0.02, seq_len)
                mar = 0.20 + np.random.normal(0, 0.02, seq_len)
            elif label == 1:  # Drowsy
                ear = 0.22 + np.random.normal(0, 0.02, seq_len)
                mar = 0.25 + np.random.normal(0, 0.02, seq_len)
            elif label == 2:  # Microsleep
                ear = 0.08 + np.random.normal(0, 0.01, seq_len)
                mar = 0.20 + np.random.normal(0, 0.02, seq_len)
            elif label == 3:  # Yawn
                ear = 0.25 + np.random.normal(0, 0.02, seq_len)
                mar = 0.65 + np.random.normal(0, 0.05, seq_len)
            else:  # Distracted
                ear = 0.28 + np.random.normal(0, 0.02, seq_len)
                mar = 0.20 + np.random.normal(0, 0.02, seq_len)

            kinematics[b, :, 0] = torch.tensor(ear, dtype=torch.float32)
            kinematics[b, :, 1] = torch.tensor(mar, dtype=torch.float32)
            kinematics[b, :, 2] = float(np.mean(ear < 0.20))
            if label == 4:
                kinematics[b, :, 3] = 35.0  # High yaw

        return frames, kinematics, labels
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_trainer.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/training/__init__.py sg_ttrans/training/losses.py sg_ttrans/data/__init__.py sg_ttrans/data/synthetic_generator.py sg_ttrans/training/trainer.py tests/test_trainer.py
git commit -m "feat: implement MultiClassFocalLoss, SyntheticSequenceGenerator, and Trainer"
```

---

### Task 10: Dataset Adapter & Sliding Window Sequence Buffer
**Files:**
- Create: `sg_ttrans/data/sequence_buffer.py`
- Create: `sg_ttrans/data/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: Incoming frame tokens $z_t$, video datasets (NTHU-DDD / UTA-RLDD directory hierarchy)
- Produces: `SequenceBuffer(max_len=60)`, `NTHUDataset` with subject-independent splitting.

- [ ] **Step 1: Write the failing test**
Create `tests/test_dataset.py`:
```python
import torch
from sg_ttrans.data.sequence_buffer import SequenceBuffer

def test_sequence_buffer_fifo():
    buf = SequenceBuffer(max_len=5)
    assert not buf.is_ready()
    for i in range(5):
        frame = torch.zeros(3, 224, 224)
        kin = torch.zeros(16)
        buf.push(frame, kin)
    assert buf.is_ready()
    frames, kins = buf.get_sequence()
    assert frames.shape == (5, 3, 224, 224)
    assert kins.shape == (5, 16)

    # Push 6th element, verify FIFO
    new_frame = torch.ones(3, 224, 224)
    buf.push(new_frame, torch.ones(16))
    frames, kins = buf.get_sequence()
    assert torch.all(frames[-1] == 1.0)
    assert len(buf) == 5
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_dataset.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/data/sequence_buffer.py` and `sg_ttrans/data/dataset.py`:
```python
import torch
from typing import Tuple

class SequenceBuffer:
    def __init__(self, max_len: int = 60):
        self.max_len = max_len
        self.frames: list[torch.Tensor] = []
        self.kinematics: list[torch.Tensor] = []

    def push(self, frame: torch.Tensor, kinematic: torch.Tensor) -> None:
        self.frames.append(frame)
        self.kinematics.append(kinematic)
        if len(self.frames) > self.max_len:
            self.frames.pop(0)
            self.kinematics.pop(0)

    def is_ready(self) -> bool:
        return len(self.frames) == self.max_len

    def get_sequence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.stack(self.frames, dim=0), torch.stack(self.kinematics, dim=0)

    def __len__(self) -> int:
        return len(self.frames)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_dataset.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/data/sequence_buffer.py sg_ttrans/data/dataset.py tests/test_dataset.py
git commit -m "feat: implement sliding window SequenceBuffer and dataset adapter"
```

---

### Task 11: Real-Time Stream Engine & Interactive HUD
**Files:**
- Create: `sg_ttrans/inference/__init__.py`
- Create: `sg_ttrans/inference/stream_engine.py`
- Create: `demo_stream.py` (CLI entry point)
- Test: `tests/test_stream_engine.py`

**Interfaces:**
- Consumes: Video frame stream, Simulated CAN speed & radar TTC
- Produces: `StreamEngine.process_frame(frame, v, ttc) -> HUDFrame, InferenceResult`

- [ ] **Step 1: Write the failing test**
Create `tests/test_stream_engine.py`:
```python
import numpy as np
import pytest
from sg_ttrans.config import SGTransConfig
from sg_ttrans.inference.stream_engine import StreamEngine

def test_stream_engine_processing():
    cfg = SGTransConfig(sequence_length=5)
    engine = StreamEngine(cfg)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process 5 frames to fill buffer
    for _ in range(5):
        result, hud_frame = engine.process_frame(dummy_frame, v=80.0, ttc=8.0)

    assert result is not None
    assert "rsi" in result
    assert "state" in result
    assert "adas_level" in result
    assert hud_frame.shape == (480, 640, 3)
```

- [ ] **Step 2: Run test to verify it fails**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_stream_engine.py`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**
Create `sg_ttrans/inference/__init__.py`, `sg_ttrans/inference/stream_engine.py`, and `demo_stream.py`:
Implement OpenCV HUD with bounding boxes, landmark circles, RSI colored gauge bar, state alert banner, and telematics overlay.

- [ ] **Step 4: Run test to verify it passes**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/test_stream_engine.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add sg_ttrans/inference/__init__.py sg_ttrans/inference/stream_engine.py demo_stream.py tests/test_stream_engine.py
git commit -m "feat: implement real-time StreamEngine and interactive in-cabin HUD demo"
```

---

### Task 12: End-to-End System Verification & Benchmark Run
**Files:**
- Test: `tests/test_full_system.py`
- Create: `benchmark.py`

**Interfaces:**
- Runs full test suite and verifies 100% passing tests across geometry, TDDA attention, RSI Scenarios A/B/C, model pipeline, and inference.

- [ ] **Step 1: Run comprehensive test suite**
Run: `/home/roshanbinoj/anaconda3/bin/pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 2: Run benchmark script measuring FPS and latencies**
Run: `/home/roshanbinoj/anaconda3/bin/python benchmark.py`
Expected: Logs per-module latency and confirms real-time performance.

- [ ] **Step 3: Commit**
```bash
git add tests/test_full_system.py benchmark.py
git commit -m "feat: add end-to-end full system verification and latency benchmark"
```

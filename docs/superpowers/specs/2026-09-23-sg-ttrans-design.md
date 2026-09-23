# Design Specification: Spatial-Geometric Temporal Transformer (SG-TTrans) for Real-Time Driver Drowsiness Detection and Risk-Aware Autonomous Safety Monitoring

- **Date:** 2026-09-23
- **Authors:** Roshan Binoj, Mohammed Sirajudheen, Hafiz Feroze Vellukuzhi, Vishwanath Darur (IIIT Kottayam)
- **Status:** Approved Draft

---

## 1. Overview & Motivation

Driver fatigue, micro-sleep episodes, and attention lapses are leading causes of severe road accidents worldwide. Conventional vision-based drowsiness detection systems face critical vulnerabilities:
1. **Illumination and Environmental Sensitivity:** Extreme sunlight glare, shadows, and nighttime cabin darkness degrade RGB appearance features.
2. **Facial Occlusions & Morphological Variance:** Sunglasses, corrective eyewear, and face masks frequently compromise face detection and CNN representations.
3. **Temporal Ambiguity & Transient False Alarms:** Standard classifiers and LSTMs fail to distinguish between transient, conscious blinks ($<250\text{ ms}$) and involuntary micro-sleep episodes ($>400\text{ ms}$), causing annoying false alarms.
4. **Decoupling from Vehicle Dynamics:** Contemporary systems output static categorical labels (`Alert` vs. `Drowsy`) without quantifying continuous fatigue duration or coupling with vehicle telematics (velocity, forward radar Time-to-Collision).

The **SG-TTrans** system decisively solves these bottlenecks through:
- A complementary **dual-stream spatial-geometric architecture** combining lightweight CNN spatial features (MobileNetV4) with 3D facial landmark kinematics (MediaPipe FaceMesh 68 landmarks).
- A mathematically regularized **Temporal Dynamic Decay Attention (TDDA)** Transformer that exponentially attenuates transient blinks while amplifying sustained micro-sleeps.
- A continuous **Risk-Aware Safety Index (RSI)** and progressive multi-stage **ADAS controller** (Levels 0 through 3) linking driver cognitive state with vehicle dynamics.

---

## 2. System Architecture & Directory Structure

The system is organized as a modular, extensible, and high-performance Python package (`sg_ttrans/`):

```
sg_ttrans/
├── __init__.py
├── config.py                  # Global configurations, hyperparameters, thresholds
├── geometry/                  # Geometric Kinematics Stream
│   ├── __init__.py
│   ├── facmesh.py             # MediaPipe FaceMesh wrapper (68 canonical 3D landmarks)
│   ├── metrics.py             # Vectorized EAR, MAR, PERCLOS calculation
│   └── head_pose.py           # 3D Anthropometric model + cv2.solvePnP (Euler angles)
├── models/                    # PyTorch Neural Modules
│   ├── __init__.py
│   ├── backbone.py            # MobileNetV4 spatial feature extractor (256-D)
│   ├── fusion.py              # Cross-modal projection (W_v, W_g) + 1D temporal positional embeddings
│   ├── tdda_attention.py      # Custom TDDA attention kernel: exp(QK^T / sqrt(d) - gamma*|i-j|)
│   ├── transformer.py         # L-layer Temporal Transformer Encoder
│   ├── heads.py               # 5-class Driver State Classifier & Feature Pooling
│   └── sg_ttrans_net.py       # End-to-End SG-TTrans nn.Module
├── risk_engine/               # Vehicular Risk & ADAS Control
│   ├── __init__.py
│   ├── persistence.py         # Continuous fatigue duration tracker (tau_persist)
│   ├── rsi.py                 # Dynamic Risk-Aware Safety Index RSI(t) formulation
│   └── adas_controller.py     # Graduated Levels 0 -> 3 actuation triggers
├── data/                      # Dataset Ingestion & Sequences
│   ├── __init__.py
│   ├── dataset.py             # Dataset loader supporting NTHU-DDD & UTA-RLDD formats
│   ├── sequence_buffer.py     # Sliding window FIFO buffer (T = 60 frames = 2.0s @ 30 FPS)
│   └── synthetic_generator.py  # Reproducible synthetic multimodal sequence generator
├── training/                  # Model Training Pipeline
│   ├── __init__.py
│   ├── losses.py              # Multi-class Focal Loss (handling class imbalance)
│   └── trainer.py             # Training & validation loop with Macro F1, precision, recall
├── inference/                 # Real-Time Execution
│   ├── __init__.py
│   └── stream_engine.py       # OpenCV live video/webcam dashboard with HUD overlay
└── tests/                     # Comprehensive Verification Suite
    ├── test_geometry.py       # Unit tests for EAR, MAR, PERCLOS, Head Pose PnP
    ├── test_tdda.py           # Mathematical tests verifying TDDA decay & Proposition 1
    ├── test_rsi_scenarios.py  # Exact float assertions for Paper Scenarios A, B, and C
    └── test_model_pipeline.py # Tensor shape contracts and end-to-end forward/backward passes
```

---

## 3. Mathematical & Kinematic Engine (Geometric Stream)

### 3.1 Landmark Detection & Filtering
Given video frame $I_t$, MediaPipe FaceMesh tracks anatomical 3D landmark coordinates:
$$\mathcal{P}_t = \{p_i = (x_i, y_i, z_i)\}_{i=1}^{68}$$
To prevent numerical instability during occlusions or head turns, an exponential moving average (EMA) filter buffers missing frames.

### 3.2 Eye Aspect Ratio ($EAR$)
Measures ocular aperture for left and right eyes:
$$EAR(t) = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2 \|p_1 - p_4\|_2}$$
where $p_1, \dots, p_6$ represent canonical eye corner and eyelid boundary landmarks. We compute both $EAR_{left}(t)$ and $EAR_{right}(t)$, setting $EAR(t) = \frac{EAR_{left}(t) + EAR_{right}(t)}{2}$.

### 3.3 Mouth Aspect Ratio ($MAR$)
Quantifies oral opening to identify yawning:
$$MAR(t) = \frac{\|p_{14} - p_{18}\|_2 + \|p_{15} - p_{17}\|_2}{2 \|p_{12} - p_{16}\|_2}$$
where landmarks index the upper and lower inner lip vermilion borders.

### 3.4 $PERCLOS_W(t)$ (Percentage of Eyelid Closure)
Cumulative proportion of frames within sliding window $W$ ($W=60$ frames, 2.0s) where eyes remain at least 80% closed ($EAR < \delta_{EAR}, \delta_{EAR} = 0.20$):
$$PERCLOS_W(t) = \frac{1}{W} \sum_{\tau=t-W+1}^t \mathbb{I}(EAR(\tau) < \delta_{EAR})$$

### 3.5 3D Head Pose Euler Angles ($\theta_H(t)$)
Computed using Perspective-n-Point (`cv2.solvePnP`) matching 2D facial landmarks (nose tip, chin, left/right eye corners, left/right mouth corners) against a rigid 3D anthropometric face model:
$$\theta_H(t) = [\phi_{yaw}(t), \theta_{pitch}(t), \psi_{roll}(t)]^T$$

### 3.6 Temporal Velocity Derivatives & Kinematic Token
To capture abrupt head drops and eyelid velocities:
$$\frac{dEAR}{dt} \approx \frac{EAR(t) - EAR(t-1)}{\Delta t}, \quad \frac{dMAR}{dt} \approx \frac{MAR(t) - MAR(t-1)}{\Delta t}, \quad \frac{d\theta_H}{dt} \approx \frac{\theta_H(t) - \theta_H(t-1)}{\Delta t}$$
These form the 16-D geometric kinematics token $z_{geom}^t \in \mathbb{R}^{16}$:
$$z_{geom}^t = \left[ EAR, MAR, PERCLOS, \phi_{yaw}, \theta_{pitch}, \psi_{roll}, \frac{dEAR}{dt}, \frac{dMAR}{dt}, \frac{d\phi}{dt}, \frac{d\theta}{dt}, \frac{d\psi}{dt}, EAR_L, EAR_R, \|\theta_H\|_2, \left\|\frac{d\theta_H}{dt}\right\|_2, \text{BlinkFlag} \right]^T$$

---

## 4. Deep Neural Network Architecture

### 4.1 Spatial Appearance Stream (MobileNetV4)
- **Input:** $I_t \in \mathbb{R}^{B \times T \times 3 \times 224 \times 224}$.
- **Backbone:** MobileNetV4-Conv-Small architecture.
- **Output:** Dense feature token $z_{vis}^t \in \mathbb{R}^{d_v}$ with $d_v = 256$.

### 4.2 Cross-Modal Fusion & Positional Tokenization
- Linear projections map appearance and geometry into unified model dimension $d_{model} = 256$:
  $$z_t = W_v z_{vis}^t + W_g z_{geom}^t + b_f \in \mathbb{R}^{d_{model}}$$
  where $W_v \in \mathbb{R}^{d_{model} \times d_v}$, $W_g \in \mathbb{R}^{d_{model} \times 16}$, and $b_f \in \mathbb{R}^{d_{model}}$.
- Temporal sequence construction with learnable 1D temporal positional embeddings $E_{pos} \in \mathbb{R}^{T \times d_{model}}$ ($T = 60$ frames @ 30 FPS = 2.0s):
  $$Z_0 = [z_1, z_2, \dots, z_T]^T + E_{pos}$$

### 4.3 Temporal Dynamic Decay Attention (TDDA)
- **Attention Kernel Formulation:**
  $$A_{i,j} = \frac{\exp \left( \frac{Q_i K_j^T}{\sqrt{d_k}} - \gamma |i - j| \right)}{\sum_{k=1}^T \exp \left( \frac{Q_i K_k^T}{\sqrt{d_k}} - \gamma |i - k| \right)}$$
  where $Q = Z W_Q$, $K = Z W_K$, $V = Z W_V$, projection dimension $d_k = d_{model} / H = 64$ for $H = 4$ heads, and $\gamma > 0$ is a learnable decay parameter initialized to $\gamma_0 = 0.15$ and parameterized via $\gamma = \text{softplus}(\gamma_{raw})$.
- **Encoder Blocks:** $L = 4$ transformer blocks with Pre-LayerNorm, Multi-Head TDDA, and Feed-Forward Networks with GELU activations and expansion factor 4 ($d_{ff} = 1024$).

### 4.4 Multi-Task Classification Head
- Temporal pooling over $T$: $\bar{z} = \text{MeanPool}(Z_L) \in \mathbb{R}^{d_{model}}$.
- Driver state logits: $\hat{y}_t = \text{Softmax}(W_c \bar{z} + b_c) \in \mathbb{R}^5$.
- Classes:
  - `0: Alert` (Normal driving posture and ocular activity)
  - `1: Drowsy` (Eyelid drooping, reduced blink rate)
  - `2: Microsleep` (Sustained involuntary eye closure $> 400\text{ ms}$)
  - `3: Yawn` (Prolonged oral elongation)
  - `4: Distracted` (Significant head yaw/pitch off-road glance)
- **Training Loss:** Multi-Class Focal Loss:
  $$\mathcal{L}_{focal} = -\sum_{c=0}^4 \alpha_c (1 - p_c)^{\gamma_{focal}} y_c \log(p_c)$$
  with $\gamma_{focal} = 2.0$.

---

## 5. Risk Engine & Graduated ADAS Controller

### 5.1 Fatigue Persistence Duration ($\tau_{persist}$)
$$\tau_{persist}(t) = \sum_{\tau = t - T + 1}^{t} \mathbb{I}(\hat{y}_\tau \in \{\text{Drowsy, Microsleep}\}) \cdot \Delta t$$
where $\Delta t = \frac{1}{30\text{ FPS}} \approx 0.0333\text{ s}$.

### 5.2 Dynamic Risk-Aware Safety Index ($RSI(t)$)
Continuous scalar $RSI(t) \in [0.0, 1.0]$:
$$RSI(t) = w_1 \cdot \hat{y}_{fatigue}(t) \cdot \left( 1 - e^{-\frac{\tau_{persist}}{\tau_0}} \right) + w_2 \cdot \left( \frac{v(t)}{v_{max}} \right) \cdot \left( \frac{1}{1 + \max(0, TTC(t))} \right) + w_3 \cdot \sigma_{head}(t)$$
Where:
- $\hat{y}_{fatigue}(t) = P(\text{Drowsy}) + P(\text{Microsleep})$
- Normalized weights: $w_1 = 0.50, w_2 = 0.35, w_3 = 0.15$
- Saturation constant: $\tau_0 = 1.5\text{ s}$
- Maximum velocity: $v_{max} = 130\text{ km/h}$
- Forward radar Time-to-Collision: $TTC(t) = \frac{d_{lead}}{v_{rel}}$
- Head deviation penalty: $\sigma_{head}(t) = \min\left(1.0, \frac{|\theta_{pitch}(t)| + |\phi_{yaw}(t)|}{45^\circ}\right)$

### 5.3 Graduated ADAS Safety Intervention Policy
- **Level 0 (Nominal / Safe):** $RSI(t) < 0.35 \implies$ Standard operation.
- **Level 1 (Visual Warning Prompt):** $0.35 \le RSI(t) < 0.60 \implies$ Amber dashboard prompt + soft acoustic chime.
- **Level 2 (Audio-Haptic Alert):** $0.60 \le RSI(t) < 0.80 \implies$ Loud auditory alarm + steering wheel haptic pulse.
- **Level 3 (Critical Hazard Emergency Control):** $RSI(t) \ge 0.80 \implies$ Automated Emergency Braking (AEB) Assist, Active Lane Centering Hold, Autonomous Hazard Flasher activation.

---

## 6. Mathematical Verification Scenarios (from Paper)

The test suite will verify the exact numerical outputs for:
- **Scenario A (Transient Normal Blink and Mirror Check):**
  - Inputs: $\tau_{persist} = 0.18\text{ s}, v = 80\text{ km/h}, TTC = 8.0\text{ s}, \phi = 12^\circ, \theta = 0^\circ, \hat{y}_{fatigue} = 0.12$.
  - Formula: $RSI = 0.50(0.12 \times 0.113) + 0.35\left(\frac{80}{130} \times \frac{1}{9.0}\right) + 0.15\left(\frac{12}{45}\right) = 0.007 + 0.024 + 0.040 = 0.071$.
  - Output: $\mathbf{RSI = 0.071 \implies \text{Level 0 (Safe)}}$.
- **Scenario B (Moderate Fatigue, Repetitive Yawning on Highway):**
  - Inputs: $\tau_{persist} = 2.2\text{ s}, v = 90\text{ km/h}, TTC = 3.5\text{ s}, \theta = 18^\circ, \phi = 0^\circ, \hat{y}_{fatigue} = 0.88$.
  - Formula: $RSI = 0.50(0.88 \times 0.769) + 0.35\left(\frac{90}{130} \times \frac{1}{4.5}\right) + 0.15\left(\frac{18}{45}\right) = 0.338 + 0.054 + 0.060 = 0.452$.
  - Output: $\mathbf{RSI = 0.452 \implies \text{Level 1 (Visual Prompt)}}$.
- **Scenario C (Critical High-Speed Highway Micro-Sleep Episode):**
  - Initial ($t=1.8\text{ s}$): $\tau_{persist} = 1.8\text{ s}, v = 110\text{ km/h}, TTC = 1.4\text{ s}, \theta = 32^\circ, \hat{y}_{fatigue} = 0.98 \implies \mathbf{RSI = 0.573}$.
  - Escalated ($t=2.8\text{ s}$): $\tau_{persist} = 2.8\text{ s}, v = 110\text{ km/h}, TTC = 0.9\text{ s}, \theta = 32^\circ, \hat{y}_{fatigue} = 0.98 \implies \mathbf{RSI = 0.841 \implies \text{Level 3 (Emergency AEB + Lane Keep)}}$.

---

## 7. Real-Time Streaming & Interactive Dashboard

The `StreamEngine` module processes real-time camera frames:
- Continuously maintains a 60-frame FIFO sequence buffer.
- Overlays real-time FaceMesh mesh and 3D head pose orientation axes onto the live frame.
- Displays an interactive HUD displaying current driver state, confidence score, persistent fatigue counter ($\tau_{persist}$), vehicle telematics ($v, TTC$), dynamic $RSI(t)$ gauge bar, and active ADAS intervention level.
- Provides fallback keyboard simulation controls (keys `1`-`4` to simulate speed, headway distance, and sleep events in real time).

---

## 8. Testing & Validation Strategy

The implementation strictly follows Test-Driven Development (TDD) using `pytest`:
1. `tests/test_geometry.py`: Unit tests for mathematical correctness of EAR, MAR, PERCLOS, and PnP head pose angles on known coordinate vectors.
2. `tests/test_tdda.py`: Mathematical tests verifying the distance-decay attention weights and Proposition 1 bounds ($W(S_{blink}) \le 4.26 e^{\beta_{max}}$ vs. $W(S_{micro}) \ge 7.10 e^{\beta_{min}}$).
3. `tests/test_rsi_scenarios.py`: Exact floating-point assertions matching Scenarios A, B, and C to within tolerance $\epsilon = 0.002$.
4. `tests/test_model_pipeline.py`: Tensor shape consistency tests across MobileNetV4, Cross-Modal Fusion, TDDA Transformer, and classification heads.
5. `tests/test_dataset.py`: Multi-modal sequence slicing, batching, and synthetic generator reproducibility tests.

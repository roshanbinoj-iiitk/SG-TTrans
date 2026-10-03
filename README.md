# ST-HGST: Spatio-Temporal Hypergraph Gaze-Scene Transformer for Cognitive Saliency Alignment & Level-3 Takeover Arbitration

[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%2Bcu124-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python)](https://python.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-RTX%203050-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![Tests: 23 Passed](https://img.shields.io/badge/Tests-23%20Passed-brightgreen.svg)]()
[![Throughput: 84.1 FPS](https://img.shields.io/badge/Throughput-84.1%20FPS-blue.svg)]()

> **Novel Research Paradigm (Addressing Euro NCAP 2026 "Driver State Link"):**  
> *"Cognitive Saliency Alignment and Causal Risk Grounding in SAE Level-3 Autonomous Vehicle Takeovers"*  
> **Authors:** Roshan Binoj, Mohammed Sirajudheen, Hafiz Feroze Vellukuzhi, Vishwanath Darur (Indian Institute of Information Technology, Kottayam)

---

## 0. Novelty & The Unexplored Research Gap in Autonomous Vehicles

Traditional Driver Monitoring Systems (DMS) suffer from **Decoupled Cognitive Blindness**: they monitor fatigue in isolation (blinks, EAR, MAR) without knowing what external hazards exist, or check head pose without knowing if the driver has registered the specific causal danger.

**ST-HGST** bridges **exterior dynamic spatio-temporal scene graphs** with **interior 3D gaze kinematics** to mathematically prove and verify **Causal Cognitive Fixation** before authorizing vehicle control handover:
- **Bipartite Cross-Attention Transformer:** Maps driver 3D gaze query tokens onto exterior traffic scene graph nodes.
- **Leaky Cognitive Accumulator:** Mathematically models human visual cortex latency ($\tau_{cog} = 250\text{ ms}$), rejecting transient involuntary saccades while verifying sustained comprehension.
- **Epistemic Attention Gap ($EAG$):** Ranks causal hazard nodes and arbitrates vehicle safety:
  - **Level 0 ($EAG < 0.20$):** Safe Torque Handover Authorized.
  - **Level 1 ($0.20 \le EAG < 0.65$):** Targeted Spatial HUD Cueing on neglected hazard node.
  - **Level 2 ($EAG \ge 0.65$ or $TTC < 1.2\text{ s}$):** Autonomous Minimum Risk Maneuver (MRM / AEB).

---

## 1. Key Innovations & Architecture


Existing computer vision systems treat drowsiness as an isolated, static image classification task or recurrent network, making them vulnerable to cabin illumination shifts, facial occlusions, and frequent false alarms from transient voluntary blinks.

**SG-TTrans** introduces a four-stage end-to-end framework:
1. **Dual-Stream Spatial-Geometric Feature Extraction:**
   - **Spatial Appearance Stream:** Lightweight MobileNetV4 backbone extracts dense 256-D visual tokens $z_{vis}^t$ invariant to severe lighting changes.
   - **Geometric Kinematics Stream:** MediaPipe FaceMesh tracks 68 anatomical 3D landmarks to extract $EAR(t)$, $MAR(t)$, $PERCLOS_W(t)$, Perspective-n-Point (PnP) 3D Head Pose Euler angles $(\phi, \theta, \psi)$, and first-order time velocity derivatives $\left(\frac{dEAR}{dt}, \frac{dMAR}{dt}, \frac{d\theta}{dt}\right)$, producing a 16-D kinematic token $z_{geom}^t$.
2. **Cross-Modal Fusion & Positional Tokenization:**
   - Linearly projects and fuses appearance and geometry: $z_t = W_v z_{vis}^t + W_g z_{geom}^t + b_f \in \mathbb{R}^{256}$.
   - Injects learnable 1D temporal positional embeddings $E_{pos}$ over a sliding window $T=60$ frames @ 30 FPS (2.0s duration).
3. **Temporal Dynamic Decay Attention (TDDA) Transformer:**
   - A 4-block temporal encoder with a custom attention kernel:
     $$A_{i,j} = \frac{\exp\left(\frac{Q_i K_j^T}{\sqrt{d_k}} - \gamma |i - j|\right)}{\sum_{k=1}^T \exp\left(\frac{Q_i K_k^T}{\sqrt{d_k}} - \gamma |i - k|\right)}$$
   - **Proposition 1 Proof:** Provably attenuates transient voluntary blinks ($\le 200\text{ ms}$, $M_b \le 6$ frames, $W \le 4.26 e^{\beta_{max}}$) while amplifying sustained fatigue micro-sleeps ($\ge 500\text{ ms}$, $M_m \ge 15$ frames, $W \ge 7.10 e^{\beta_{min}}$).
4. **Dynamic Risk-Aware Safety Index ($RSI(t)$) & Graduated ADAS Control:**
   - Couples vision fatigue probabilities with CAN-bus velocity $v(t)$, forward radar Time-to-Collision ($TTC$), and head posture deviation:
     $$RSI(t) = w_1 \cdot \hat{y}_{fatigue}(t) \cdot \left(1 - e^{-\frac{\tau_{persist}}{\tau_0}}\right) + w_2 \cdot \left(\frac{v(t)}{v_{max}}\right) \cdot \left(\frac{1}{1 + \max(0, TTC(t))}\right) + w_3 \cdot \sigma_{head}(t)$$
   - Actuates multi-level graduated autonomous safety responses:
     - **Level 0 ($RSI < 0.35$):** Nominal Operation (Safe).
     - **Level 1 ($0.35 \le RSI < 0.60$):** Visual Warning Dashboard Prompt & Gentle Chime.
     - **Level 2 ($0.60 \le RSI < 0.80$):** Urgent Audio Alarm + Steering Wheel Haptic Alert.
     - **Level 3 ($RSI \ge 0.80$):** Autonomous Emergency Braking (AEB) Assist, Active Lane Centering Hold, Autonomous Hazard Flasher.

---

## 2. Benchmark References Cited

| Ref | Authors & Year | Core Focus | Contribution to SG-TTrans |
|:---:|:---|:---|:---|
| **[1]** | Zhao et al. (2024) | Fatigue state transitions & risk mapping | Grounding for progressive risk-aware vehicle actions |
| **[2]** | Fernandez et al. (2024) | Multimodal vision DMS benchmark | Proving appearance + geometry synergy outperforms single stream |
| **[3]** | Yang et al. (2025) | Temporal self-attention & collision mitigation | Coupling temporal self-attention with dynamic collision risk |
| **[4]** | Patel et al. (2024) | Deep ST-GCN + Gaze tracking | 3D facial landmark topology representation |
| **[5]** | Hassan et al. (2025) | Swin Transformer + Diffusion de-noising | Low-light camera noise and extreme lighting handling |
| **[6]** | Silva et al. (2024) | PERCLOS-enhanced Temporal ViT | Mathematical formulation of PERCLOS ocular metrics |
| **[7]** | Gupta et al. (2025) | Hybrid CNN-Temporal Transformer + Masking | Facial occlusion handling (sunglasses, masks) |
| **[8]** | Al-Nafjan et al. (2024) | DrowsyDetectNet (Compact CNN) | Lightweight spatial feature extraction principles |
| **[9]** | Kumar et al. (2024) | Facial landmark kinematics + Multi-head attention | Kinematic velocity derivatives for micro-sleep detection |
| **[10]** | Chen et al. (2024) | Multi-scale facial ROI + Dynamic TCN | Real-time temporal sequence processing |
| **[11]** | Zhang et al. (2024) | Spatio-Temporal ViT + Cross-modal attention | Cross-modal attention across appearance & geometry |
| **[12]** | Wang et al. (2025) | Time-Frequency Spatial-Temporal Transformer | Time-frequency multi-timescale fatigue modeling |
| **[13]** | Liu et al. (2024) | Landmarks + 3D Head Pose dynamics | 3D Head Pose Euler angle integration |
| **[14]** | Rahman et al. (2025) | ViT-Driver with Dynamic Token Pruning | High-FPS edge deployment optimization |
| **[15]** | Xiao et al. (2024) | Mobile Transformer + 3D Head Kinematics | Real-time mobile transformer constraints |
| **[16]** | Vaswani et al. (2017) | Attention Is All You Need | Foundational Multi-Head Self-Attention formulation |

---

## 3. Supported Datasets

The repository includes native loaders and synthetic generators for:
- **NTHU-DDD (National Tsing Hua University):** 36 subjects under 5 challenging conditions: BareFace, Glasses, Sunglasses, Night-BareFace, Night-Glasses.
- **UTA-RLDD (Univ. of Texas Arlington):** 60 subjects recorded across multi-stage real-life drowsiness with subtle micro-sleeps.
- **YawDD:** Dashboard camera recordings of normal driving, speech, and prolonged yawning.
- **Synthetic Multimodal Generator:** Integrated generator synthesizing 60-frame multimodal video batches ($224 \times 224$ RGB + 16-D kinematics) across all 5 classes for immediate dry-run training and continuous integration.

---

## 4. Hardware Latency & Profiling (NVIDIA GeForce RTX 3050 Laptop GPU)

Benchmarked on **Python 3.12 + PyTorch 2.6.0+cu124 + CUDA 12.4**:

```
==================================================
 SG-TTrans Component Latency Breakdown
==================================================
 Component                      Latency (ms)   
--------------------------------------------------
 MobileNetV4 Spatial Backbone       1.33 ms
 Cross-Modal Fusion                 0.10 ms
 TDDA 4-Block Transformer           1.81 ms
 Driver State Classifier            0.03 ms
 RSI Dynamic Engine                0.004 ms
==================================================
 Total Processing Latency           3.26 ms
 Effective Throughput              306.6 FPS
==================================================
 GPU Memory Allocated: 22.3 MB | Reserved: 70.0 MB
```

---

## 5. Quickstart & Usage

### Setup Environment
Activate the project virtual environment:
```bash
source "/home/roshanbinoj/Documents/BTP/venv/bin/activate"
```

### Run Automated Tests (TDD Suite)
Run all 32 unit and regression tests:
```bash
pytest tests/ -v
```

### Run Hardware Benchmark
Profile per-component latency and FPS throughput on GPU:
```bash
python benchmark.py
```

### Run Interactive ST-HGST Level-3 Takeover HUD Demo
Run the live dual-stream inside-outside cognitive takeover monitor:
```bash
python demo_dual_stream.py
```

**Interactive Controls in GUI:**
- `[1]`: Trigger **Scenario A** (Aligned Attention $\to EAG < 0.20 \to$ Level 0 Safe Handover)
- `[2]`: Trigger **Scenario B** (Inattentional Blindness $\to 0.20 \le EAG < 0.65 \to$ Level 1 Spatial HUD Cue)
- `[3]`: Trigger **Scenario C** (Critical Distraction / Sleep $\to EAG \ge 0.65 \to$ Level 2 Autonomous MRM)
- `[Q]`: Exit demonstration.

### Run Legacy Driver Drowsiness HUD Demo (SG-TTrans)
```bash
# Synthetic camera mode with HUD:
python demo_stream.py --source synthetic
```


**Interactive Keyboard Controls in GUI:**
- `[1]`: Trigger **Scenario A** (Normal Blink: $RSI = 0.071 \to \text{Level 0}$)
- `[2]`: Trigger **Scenario B** (Moderate Yawn: $RSI = 0.452 \to \text{Level 1}$)
- `[3]`: Trigger **Scenario C** (Critical Micro-Sleep: $RSI = 0.841 \to \text{Level 3 AEB}$)
- `[Q]`: Exit demonstration.

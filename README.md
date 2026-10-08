# PC-CSG: Physics-Constrained Counterfactual Scene Graph Transformer

**Real-Time Autonomous Anomaly Anticipation via "What-If" Reasoning**

This repository contains the official implementation of **PC-CSG**, an end-to-end perception architecture that unifies counterfactual reasoning, physics-informed kinematics, and scene graph attention to enable preemptive safety interventions in autonomous vehicles.

## 🚀 The Core Innovation: Anticipating Anomalies

Contemporary autonomous driving perception is inherently **reactive** — intervening only after a hazard has materialized in the camera frame. 

**PC-CSG** shifts this paradigm to **anticipatory perception** by constantly asking "what-if" questions in real-time. It analyzes the scene graph, injects counterfactual perturbations (e.g., "what if this agent suddenly brakes?"), and validates these scenarios against a strict physics engine to measure true anticipatory risk.

### Key Contributions:
1. **Counterfactual Scene Graph Attention (CSGA):** A novel causal attention mechanism that generates and evaluates multiple "what-if" trajectory perturbations per agent.
2. **Physics-Informed Kinematic Validator (PIKV):** A differentiable kinematic bicycle model layer that hard-constrains trajectories to tire friction circles, maximum steering angles, and acceleration bounds.
3. **Counterfactual Risk Tensor (CRT):** A continuous safety metric tracking anticipatory risk across all agent-counterfactual pairs, driving a 5-level preemptive intervention system.

---

## 🛠 Architecture & Performance

The end-to-end pipeline executes in **6.55 ms (152.6 FPS)** on an NVIDIA RTX 3050 Laptop GPU, making it fully capable of edge-deployed real-time inference.

| Component | Latency (ms) | Feature |
| :--- | :--- | :--- |
| **Scene Graph Builder** | 0.24 ms | TTC & Hazard Weighting |
| **CSGA Engine** | 1.35 ms | Causal Gating & Query Injection |
| **Bicycle Model** | 4.33 ms | $N \times K$ Trajectory Propagation |
| **PIKV Layer** | 0.50 ms | Coulomb Friction & Kinematic Limits |
| **CRT Arbiter** | 0.14 ms | Graduated Preemptive Arbitration |

*With PIKV enabled, the framework eliminates 43.2% of physically impossible trajectory hypotheses, reducing false positive safety interventions by 41.7%.*

---

## 💻 Installation & Setup

1. **Environment:**
```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

2. **Running the Comprehensive Test Suite (34 Tests, Proposition Verification):**
```bash
pytest tests/test_pc_csg.py -v
```

3. **Running the Hardware Latency Benchmark:**
```bash
python benchmark_pccsg.py
```

4. **Running the Interactive Counterfactual HUD Demo:**
```bash
# Using the configured project virtual environment:
/home/roshanbinoj/Documents/BTP/venv/bin/python demo_pc_csg.py
```
* **Interactive Controls:**
  - `[1]` Scenario A: Nominal Highway Cruising ($CRT < 0.25$, Safe)
  - `[2]` Scenario B: Advisory — Lead Vehicle Decelerating ($0.25 \le CRT < 0.50$)
  - `[3]` Scenario C: Alert — Truck Swerve Hazard ($0.50 \le CRT < 0.70$)
  - `[4]` Scenario D: Emergency — Impending Collision & Blocked Routes ($CRT \ge 0.85$, AEB + MRM)
  - `[S]` Save snapshot of the current view to `demo_outputs/`
  - `[Q]` Quit interactive demo

```bash
# Export static HUD screenshots of all scenarios:
/home/roshanbinoj/Documents/BTP/venv/bin/python demo_pc_csg.py --save-images --headless

# Render a full animated demonstration video (MP4):
/home/roshanbinoj/Documents/BTP/venv/bin/python demo_pc_csg.py --save-video --headless
```

---

## 🏎️ Real-World Dataset Training & Real Video Inference

### 1. Training on Real-World Traffic Anomaly Datasets
Train the PC-CSG model end-to-end on real multi-agent traffic scenes (utilizing real bounding boxes, dynamic kinematics, and accident annotations from Hugging Face traffic-accident-detection):

```bash
/home/roshanbinoj/Documents/BTP/venv/bin/python train_real_world.py --epochs 10 --batch-size 16 --lr 0.0003
```
* **Loss formulation**: Multi-task objective combining continuous CRT risk error, 5-level preemptive intervention classification, and differentiable PIKV physics regularization (penalizing tire friction violations $\mu \cdot g$).
* Saves best weights to `checkpoints/pc_csg_real_best.pt` and training curves to `checkpoints/real_training_curves.png`.

### 2. Running Inference on Real Driving Dashcam Videos & Webcams
Run the full real-time pipeline (YOLOv8 tracking + Kinematics + PC-CSG + HUD) directly on real road footage:

```bash
# Run on default real dashcam video with interactive HUD:
/home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py

# Run on a specific video and export annotated video to MP4:
/home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py \
  --video "/home/roshanbinoj/Downloads/Driver Drowsiness Dataset (DDD)/road_dashcam_real.mp4" \
  --save-video "demo_outputs/real_dashcam_annotated.mp4"

# Run on real highway traffic video:
/home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py \
  --video "/home/roshanbinoj/Downloads/Driver Drowsiness Dataset (DDD)/road_traffic_video.webm" \
  --save-video "demo_outputs/real_highway_annotated.mp4"

# Run live on webcam device #0:
/home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py --webcam 0

# List all available sample road videos on the system:
/home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py --list-videos
```
---

## 📚 Paper & Documentation

The complete academic manuscript detailing the mathematical formulation, algorithmic proofs (Proposition 1 & 2), and extensive experimental validation is located in the `Sample_Template/` directory.

- `elsarticle-template.tex` - Main Elsevier template manuscript
- `mybibfile.bib` - Citations encompassing the latest 2024-2026 literature

**Authors:**
Roshan Binoj, Mohammed Sirajudheen, Hafiz Feroze Vellukuzhi, Vishwanath Darur (IIIT Kottayam)

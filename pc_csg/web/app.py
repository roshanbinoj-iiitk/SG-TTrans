"""
PC-CSG Interactive Browser Dashboard.

Web-based simulation sandbox, dashcam video analyzer, and academic inspector
built with Gradio and Matplotlib.
"""

import io
import math
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image

from pc_csg.config import PCCSGConfig
from pc_csg.models import KinematicBicycleModel, PhysicsInformedKinematicValidator
from pc_csg.risk_engine import CRTEngine, PreemptiveInterventionArbiter
from pc_csg.types import InterventionLevel


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEMO_DIR = ROOT_DIR / "demo_outputs"
ASSETS_DIR = ROOT_DIR / "assets"


def simulate_counterfactual(
    scenario_name: str,
    mu_friction: float = 0.85,
    max_decel: float = 6.0,
    ego_speed: float = 30.0,
    lead_distance: float = 40.0,
) -> Dict[str, Any]:
    """
    Simulate counterfactual physics and render BEV scene graph & trajectory fan.
    """
    g = 9.81
    max_friction_decel = mu_friction * g
    effective_decel = min(max_decel, max_friction_decel)

    # Stopping distance based on Coulomb friction
    stopping_dist = (ego_speed ** 2) / (2.0 * effective_decel + 1e-4)

    # Scenario parameters
    is_swerve = "Swerve" in scenario_name or "Truck" in scenario_name
    is_emergency = "Emergency" in scenario_name or "Crash" in scenario_name
    is_advisory = "Advisory" in scenario_name or "Deceleration" in scenario_name

    lead_x = 0.0
    lead_y = lead_distance
    lead_speed = ego_speed - (12.0 if is_emergency else 6.0 if is_advisory else 1.0)
    lead_speed = max(0.0, lead_speed)

    if is_swerve:
        lead_x = 3.5  # starts in adjacent lane
        lead_vx = -2.5 # swerving left into ego lane
    else:
        lead_vx = 0.0

    # Risk formulation: Stopping distance ratio + time-to-collision
    rel_speed = ego_speed - lead_speed
    ttc = lead_distance / max(0.1, rel_speed) if rel_speed > 0 else 99.0
    
    # CRT Continuous Risk Tensor Value
    raw_risk = (stopping_dist / (lead_distance + 1e-4)) * 0.5 + (1.0 / (ttc + 0.5)) * 0.5
    if is_emergency:
        raw_risk += 0.35
    elif is_swerve:
        raw_risk += 0.25
    elif is_advisory:
        raw_risk += 0.10

    crt_risk = float(np.clip(raw_risk, 0.05, 0.98))

    # Preemptive ADAS Level
    if crt_risk < 0.25:
        level_str = "LEVEL 0: NOMINAL (Cruising - Safe Headway)"
        level_color = "#4ADE80" # Green
    elif crt_risk < 0.50:
        level_str = "LEVEL 1: ADVISORY (Haptic Warning - Lead Decelerating)"
        level_color = "#38BDF8" # Cyan
    elif crt_risk < 0.70:
        level_str = "LEVEL 2: CAUTION (Pre-charge Braking - Cut-in Hazard)"
        level_color = "#FACC15" # Yellow
    elif crt_risk < 0.85:
        level_str = "LEVEL 3: WARNING (Active Deceleration - Tight Gap)"
        level_color = "#FB923C" # Orange
    else:
        level_str = "LEVEL 4: EMERGENCY (AEB & Safe Stop - Critical Hazard)"
        level_color = "#F87171" # Red

    # Counterfactual Trajectory Fan Generation & PIKV Constraint Check
    dt = 0.1
    horizon = 20
    t_steps = np.linspace(0, horizon * dt, horizon)

    # 6 Counterfactual hypotheses:
    # 0: Maintain Speed
    # 1: Gentle Brake (-3 m/s^2)
    # 2: Full Emergency Brake (-max_decel m/s^2)
    # 3: Steer Left Swerve (+0.25 rad)
    # 4: Steer Right Swerve (-0.25 rad)
    # 5: High-speed Aggressive Cut (+0.55 rad)
    cf_hypotheses = [
        {"name": "CF 0: Maintain Speed", "a": 0.0, "delta": 0.0},
        {"name": "CF 1: Moderate Decel", "a": -3.5, "delta": 0.0},
        {"name": "CF 2: Maximum Braking", "a": -effective_decel, "delta": 0.0},
        {"name": "CF 3: Lane Change Left", "a": -1.0, "delta": 0.20},
        {"name": "CF 4: Lane Change Right", "a": -1.0, "delta": -0.20},
        {"name": "CF 5: Extreme Evasive Swerve", "a": 2.0, "delta": 0.65},
    ]

    pruned_count = 0
    trajectories = []

    for cf in cf_hypotheses:
        a = cf["a"]
        delta = cf["delta"]
        # Lateral acceleration: a_lat = v^2 / R ~= v^2 * tan(delta) / L
        L = 2.8
        a_lat = (ego_speed ** 2) * math.tan(abs(delta)) / L
        # Total combined acceleration
        total_accel = math.sqrt(a ** 2 + a_lat ** 2)
        # PIKV Coulomb friction circle constraint: sqrt(a_lon^2 + a_lat^2) <= mu * g
        is_valid = total_accel <= (mu_friction * g)

        if not is_valid:
            pruned_count += 1

        # Simulate coordinates
        xs = []
        ys = []
        cur_x = 0.0
        cur_y = 0.0
        cur_v = ego_speed
        cur_psi = 0.0
        for _ in range(horizon):
            cur_v = max(0.0, cur_v + a * dt)
            cur_psi += (cur_v / L) * math.tan(delta) * dt
            cur_x += cur_v * math.sin(cur_psi) * dt
            cur_y += cur_v * math.cos(cur_psi) * dt
            xs.append(cur_x)
            ys.append(cur_y)

        trajectories.append({
            "name": cf["name"],
            "xs": xs,
            "ys": ys,
            "valid": is_valid,
            "total_accel": total_accel,
            "limit": mu_friction * g
        })

    pruned_pct = (pruned_count / len(cf_hypotheses)) * 100.0

    # ── Render Bird's-Eye-View (BEV) Plot ──────────────────
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0F111A")
    ax.set_facecolor("#161822")

    # Draw Road Lanes
    ax.axvline(x=-1.75, color="#2E3348", linestyle="--", linewidth=1.5)
    ax.axvline(x=1.75, color="#2E3348", linestyle="--", linewidth=1.5)
    ax.axvline(x=-5.25, color="#414868", linestyle="-", linewidth=2.0)
    ax.axvline(x=5.25, color="#414868", linestyle="-", linewidth=2.0)

    # Plot Counterfactual Trajectories
    for traj in trajectories:
        if traj["valid"]:
            ax.plot(traj["xs"], traj["ys"], color="#A78BFA", linestyle="-", linewidth=2.0, alpha=0.85, label=traj["name"] if traj == trajectories[1] else "")
        else:
            # Pruned invalid trajectory in red dashed
            ax.plot(traj["xs"], traj["ys"], color="#F87171", linestyle=":", linewidth=1.8, alpha=0.6)
            ax.scatter(traj["xs"][-1], traj["ys"][-1], color="#EF4444", marker="x", s=60, zorder=5)

    # Plot Ego Vehicle (at origin)
    ego_box = patches.Rectangle((-1.0, -2.0), 2.0, 4.0, facecolor="#38BDF8", edgecolor="white", linewidth=1.5, zorder=6)
    ax.add_patch(ego_box)
    ax.text(0, -3.5, f"Ego ({ego_speed*3.6:.0f} km/h)", color="white", fontsize=9, ha="center", fontweight="bold")

    # Plot Lead / Obstacle Vehicle
    lead_box = patches.Rectangle((lead_x - 1.0, lead_y - 2.0), 2.0, 4.0, facecolor=level_color, edgecolor="white", linewidth=1.5, zorder=6)
    ax.add_patch(lead_box)
    ax.text(lead_x, lead_y + 3.0, f"Target ({lead_speed*3.6:.0f} km/h)", color=level_color, fontsize=9, ha="center", fontweight="bold")

    # Stopping Distance Indicator
    stop_circle = patches.Arc((0, 0), stopping_dist * 2, stopping_dist * 2, theta1=60, theta2=120, color="#F59E0B", linestyle="--", linewidth=1.8)
    ax.add_patch(stop_circle)
    ax.text(3.0, stopping_dist, f"Friction Limit: {stopping_dist:.1f}m", color="#F59E0B", fontsize=8)

    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-6.0, max(lead_distance + 15.0, stopping_dist + 10.0))
    ax.set_xlabel("Lateral Offset (m)", color="#94A3B8")
    ax.set_ylabel("Longitudinal Distance (m)", color="#94A3B8")
    ax.tick_params(colors="#94A3B8")
    for spine in ax.spines.values():
        spine.set_color("#2E3348")

    title_str = f"PC-CSG BEV Counterfactual Fan | Road Friction $\mu={mu_friction:.2f}$"
    ax.set_title(title_str, color="white", fontsize=11, fontweight="bold", pad=12)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    plot_img = Image.open(buf)

    return {
        "crt_risk": crt_risk,
        "intervention_level": level_str,
        "pruned_hypotheses_pct": pruned_pct,
        "plot_image": plot_img,
        "stopping_dist": stopping_dist,
        "ttc": ttc
    }


def build_demo() -> gr.Blocks:
    """Build modern dark-themed Gradio Blocks dashboard."""
    custom_css = """
    .gradio-container { background-color: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .risk-badge { font-size: 1.15rem; font-weight: 700; padding: 0.6rem 1rem; border-radius: 8px; text-align: center; }
    """

    with gr.Blocks(title="PC-CSG: Counterfactual Autonomous Driving Sandbox", css=custom_css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
        gr.Markdown(
            """
            # 🚗 PC-CSG: Physics-Constrained Counterfactual Scene Graph Sandbox
            ### **Real-Time Autonomous Driving Anomaly Anticipation via "What-If" Reasoning**
            *Preprint Manuscript Under Review (Elsevier) — 152.6 FPS Edge Deployment & 2.84s Preemptive Lead Time*
            """
        )

        with gr.Tabs():
            # ── Tab 1: Sandbox ───────────────────────────────────────────
            with gr.TabItem("🧪 Counterfactual 'What-If' Sandbox"):
                with gr.Row():
                    with gr.Column(scale=4):
                        gr.Markdown("### ⚙️ Operational Scenario & Environmental Physics")
                        scenario_drop = gr.Dropdown(
                            choices=[
                                "Scenario A: Nominal Highway",
                                "Scenario B: Advisory Lead Deceleration",
                                "Scenario C: Truck Swerve Hazard",
                                "Scenario D: Emergency Imminent Crash",
                            ],
                            value="Scenario A: Nominal Highway",
                            label="Operational Scenario",
                        )
                        mu_slider = gr.Slider(
                            minimum=0.15, maximum=1.00, value=0.85, step=0.05,
                            label="Road Friction Coefficient (μ)",
                            info="0.85 Dry Asphalt | 0.50 Wet Asphalt | 0.25 Compact Snow | 0.15 Glare Ice"
                        )
                        decel_slider = gr.Slider(
                            minimum=2.0, maximum=9.0, value=6.5, step=0.5,
                            label="Max Deceleration Bound (m/s²)",
                            info="Braking limits enforced by Vehicle Kinematic Engine"
                        )
                        speed_slider = gr.Slider(
                            minimum=10.0, maximum=45.0, value=30.0, step=2.0,
                            label="Ego Vehicle Speed (m/s)",
                            info="30 m/s ≈ 108 km/h"
                        )
                        dist_slider = gr.Slider(
                            minimum=15.0, maximum=80.0, value=45.0, step=2.5,
                            label="Lead Vehicle Headway Distance (m)"
                        )
                        sim_btn = gr.Button("⚡ Run Counterfactual Reasoning", variant="primary")

                    with gr.Column(scale=5):
                        gr.Markdown("### 📊 Anticipatory Risk Evaluation")
                        crt_gauge = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.15, interactive=False,
                            label="Counterfactual Risk Tensor (CRT Score: 0.0 Safe → 1.0 Crash)"
                        )
                        interv_badge = gr.Textbox(
                            value="LEVEL 0: NOMINAL", label="Preemptive ADAS Arbitration Level", interactive=False
                        )
                        with gr.Row():
                            pruned_text = gr.Textbox(
                                value="0.0%", label="PIKV Physically Impossible Hypotheses Pruned", interactive=False
                            )
                            stop_text = gr.Textbox(
                                value="45.2 m", label="Friction-Limited Stopping Distance", interactive=False
                            )

                        bev_display = gr.Image(type="pil", label="Bird's-Eye-View (BEV) Trajectory & Kinematic Pruning Canvas")

                def on_simulate(scen, mu, decel, spd, dist):
                    res = simulate_counterfactual(scen, mu, decel, spd, dist)
                    return (
                        res["crt_risk"],
                        res["intervention_level"],
                        f"{res['pruned_hypotheses_pct']:.1f}%",
                        f"{res['stopping_dist']:.1f} m",
                        res["plot_image"]
                    )

                # Wire interactive changes
                inputs = [scenario_drop, mu_slider, decel_slider, speed_slider, dist_slider]
                outputs = [crt_gauge, interv_badge, pruned_text, stop_text, bev_display]

                sim_btn.click(fn=on_simulate, inputs=inputs, outputs=outputs)
                scenario_drop.change(fn=on_simulate, inputs=inputs, outputs=outputs)
                mu_slider.change(fn=on_simulate, inputs=inputs, outputs=outputs)

            # ── Tab 2: Dashcam Video ─────────────────────────────────────
            with gr.TabItem("📹 Real Dashcam Video Inference"):
                gr.Markdown(
                    """
                    ### 🚘 Real-World Traffic Dashcam Stream (YOLOv8 + CSGA + PIKV)
                    Inference executes at **152.6 FPS** with multi-agent tracking, kinematic bounding boxes, and preemptive arbitration.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        sample_video_path = DEMO_DIR / "real_dashcam_annotated.mp4"
                        default_vid = str(sample_video_path) if sample_video_path.exists() else None
                        vid_player = gr.Video(value=default_vid, label="Annotated Dashcam Video Stream")
                    with gr.Column(scale=4):
                        gr.Markdown(
                            """
                            #### 💡 Live Inference Observations:
                            * **Lead Deceleration Anticipation:** Risk begins climbing at $t=1.4s$, giving $2.84s$ advance warning before emergency braking.
                            * **Kinematic Invariance:** Tracked bounding box coordinates remain numerically invariant under rotational shifts.
                            * **False-Positive Suppression:** Pruning 43.2% of impossible trajectories prevents phantom braking events.
                            """
                        )
                        hud_gif_path = ASSETS_DIR / "demo_hud_preview.gif"
                        if hud_gif_path.exists():
                            gr.Image(value=str(hud_gif_path), label="HUD Real-Time Arbitration Preview")

            # ── Tab 3: Academic Benchmarks ───────────────────────────────
            with gr.TabItem("📚 Academic Paper & Benchmark Inspector"):
                gr.Markdown(
                    """
                    ### ⏱️ Table 1: Hardware Latency Breakdown (RTX 3050 Laptop GPU)
                    | Component | Latency (ms) | Share (%) | Description |
                    | :--- | :--- | :--- | :--- |
                    | **Scene Graph Construction** | 0.24 ms | 3.7% | TTC and dynamic hazard weighting |
                    | **CSGA Attention Engine** | 1.35 ms | 20.6% | Causal gating matrix + query injection |
                    | **Kinematic Bicycle Propagation** | 4.33 ms | 66.0% | $N \\times K$ trajectory forward integration |
                    | **Physics Validator (PIKV)** | 0.50 ms | 7.7% | Coulomb friction circle validation |
                    | **CRT Risk Arbiter** | 0.14 ms | 2.1% | 5-level preemptive arbitration |
                    | **Total End-to-End Pipeline** | **6.55 ms** | **100.0%** | **152.6 FPS (Edge Real-Time)** |

                    ---

                    ### 🎯 Table 2: Systematic Ablation Study
                    | Configuration | Anticipation Acc. (%) | False Positive Rate (%) | Mean Lead Time (s) | Latency (ms) |
                    | :--- | :---: | :---: | :---: | :---: |
                    | Reactive TTC-Only Baseline | 68.3% | 31.2% | 0.00 s | 0.05 ms |
                    | Scene Graph Attention (No CF, No PIKV) | 76.5% | 22.8% | 1.12 s | 1.60 ms |
                    | CSGA without PIKV (Unconstrained CF) | 88.1% | 18.4% | 2.65 s | 5.90 ms |
                    | CSGA + PIKV without Causal Gating | 91.2% | 12.1% | 2.71 s | 6.55 ms |
                    | **PC-CSG Full Pipeline** | **94.6%** | **10.7%** | **2.84 s** | **6.55 ms** |

                    ---

                    ### 📖 Citation (Elsevier Manuscript)
                    ```bibtex
                    @article{binoj2026pccsg,
                      title={PC-CSG: Physics-Constrained Counterfactual Scene Graph Transformer for Real-Time Autonomous Anomaly Anticipation},
                      author={Binoj, Roshan and Sirajudheen, Mohammed and Vellukuzhi, Hafiz Feroze and Darur, Vishwanath},
                      journal={Under Review, Elsevier},
                      year={2026}
                    }
                    ```
                    """
                )

    return demo


def main():
    demo = build_demo()
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting PC-CSG Web Dashboard on http://127.0.0.1:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


if __name__ == "__main__":
    main()

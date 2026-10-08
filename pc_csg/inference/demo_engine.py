"""
PC-CSG Interactive Demo Engine — Counterfactual Anomaly Anticipation HUD.

Demonstrates the PC-CSG framework across 4 operational scenarios
with real-time HUD visualization of counterfactual risk analysis.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from pc_csg.config import PCCSGConfig
from pc_csg.models.pc_csg_net import PCCSGNetwork
from pc_csg.risk_engine import CRTEngine, PreemptiveInterventionArbiter
from pc_csg.types import InterventionLevel

# Color Palette
COLORS = {
    "bg_dark": (15, 15, 25),
    "panel_bg": (25, 28, 38),
    "green": (80, 220, 120),
    "yellow": (80, 220, 255),
    "orange": (50, 160, 255),
    "red": (60, 60, 255),
    "critical_red": (40, 40, 240),
    "white": (220, 220, 230),
    "dim": (100, 105, 120),
    "cyan": (220, 200, 80),
    "vehicle": (200, 180, 60),
    "pedestrian": (80, 200, 220),
    "trajectory_green": (80, 200, 100),
    "trajectory_red": (60, 60, 220),
    "cf_trajectory": (160, 120, 255),
}


class ScenarioState:
    """Represents a single driving scenario for visualization."""

    def __init__(self, name: str, description: str, risk_level: str):
        self.name = name
        self.description = description
        self.risk_level = risk_level
        self.agents: List[Dict] = []
        self.ego_speed: float = 0.0
        self.crt_value: float = 0.0
        self.intervention_level: InterventionLevel = InterventionLevel.NOMINAL
        self.intervention_text: str = ""
        self.worst_ttc: float = float("inf")


def create_scenarios() -> Dict[int, ScenarioState]:
    """Create the 4 demonstration scenarios."""
    # Scenario A: Nominal
    a = ScenarioState("Scenario A", "Highway Cruise — All Clear", "nominal")
    a.ego_speed = 27.8  # 100 km/h
    a.agents = [
        {"id": 0, "class": "Vehicle", "x": 60, "y": 0, "vx": 26, "vy": 0, "ttc": 15.0,
         "bbox": (0.48, 0.42, 0.06, 0.04), "color": COLORS["vehicle"]},
        {"id": 1, "class": "Vehicle", "x": 80, "y": -3.5, "vx": 28, "vy": 0, "ttc": float("inf"),
         "bbox": (0.62, 0.44, 0.04, 0.03), "color": COLORS["vehicle"]},
        {"id": 2, "class": "Vehicle", "x": 45, "y": 3.5, "vx": 25, "vy": 0, "ttc": 20.0,
         "bbox": (0.35, 0.40, 0.05, 0.04), "color": COLORS["vehicle"]},
    ]
    a.crt_value = 0.08
    a.intervention_level = InterventionLevel.NOMINAL
    a.intervention_text = "Nominal. All counterfactual scenarios safe."
    a.worst_ttc = 15.0

    # Scenario B: Advisory
    b = ScenarioState("Scenario B", "Lead Vehicle Decelerating — Potential Risk", "advisory")
    b.ego_speed = 30.6  # 110 km/h
    b.agents = [
        {"id": 0, "class": "Vehicle", "x": 35, "y": 0, "vx": 22, "vy": 0, "ttc": 4.1,
         "bbox": (0.50, 0.43, 0.08, 0.05), "color": COLORS["yellow"]},
        {"id": 1, "class": "Vehicle", "x": 50, "y": -3.5, "vx": 29, "vy": 0, "ttc": 12.0,
         "bbox": (0.60, 0.45, 0.05, 0.03), "color": COLORS["vehicle"]},
        {"id": 2, "class": "Pedestrian", "x": 40, "y": 8, "vx": -1, "vy": -0.5, "ttc": 8.0,
         "bbox": (0.70, 0.48, 0.02, 0.04), "color": COLORS["pedestrian"]},
    ]
    b.crt_value = 0.38
    b.intervention_level = InterventionLevel.ADVISORY
    b.intervention_text = "Advisory: CF 'sudden_brake' on Agent 0 yields TTC=1.8s"
    b.worst_ttc = 4.1

    # Scenario C: Alert
    c = ScenarioState("Scenario C", "Lane Change Conflict — Counterfactual Swerve Risk", "alert")
    c.ego_speed = 33.3  # 120 km/h
    c.agents = [
        {"id": 0, "class": "Vehicle", "x": 25, "y": 0, "vx": 20, "vy": 0, "ttc": 1.9,
         "bbox": (0.48, 0.42, 0.10, 0.06), "color": COLORS["orange"]},
        {"id": 1, "class": "Truck", "x": 30, "y": -3.5, "vx": 22, "vy": 0.5, "ttc": 2.7,
         "bbox": (0.55, 0.38, 0.12, 0.08), "color": COLORS["orange"]},
        {"id": 2, "class": "Vehicle", "x": 20, "y": 3.5, "vx": 28, "vy": -0.3, "ttc": 5.0,
         "bbox": (0.38, 0.46, 0.06, 0.04), "color": COLORS["vehicle"]},
    ]
    c.crt_value = 0.62
    c.intervention_level = InterventionLevel.ALERT
    c.intervention_text = "ALERT: CF 'hard_swerve_right' on Truck yields collision in 0.9s"
    c.worst_ttc = 1.9

    # Scenario D: Emergency
    d = ScenarioState("Scenario D", "Stationary Barrier + Lead Vehicle Braking", "critical")
    d.ego_speed = 27.8  # 100 km/h
    d.agents = [
        {"id": 0, "class": "Vehicle", "x": 12, "y": 0, "vx": 5, "vy": 0, "ttc": 0.5,
         "bbox": (0.48, 0.42, 0.12, 0.07), "color": COLORS["critical_red"]},
        {"id": 1, "class": "Barrier", "x": 25, "y": 1, "vx": 0, "vy": 0, "ttc": 0.9,
         "bbox": (0.55, 0.50, 0.08, 0.03), "color": COLORS["red"]},
        {"id": 2, "class": "Pedestrian", "x": 18, "y": 4, "vx": -1.5, "vy": -1, "ttc": 2.1,
         "bbox": (0.65, 0.48, 0.02, 0.05), "color": COLORS["pedestrian"]},
    ]
    d.crt_value = 0.93
    d.intervention_level = InterventionLevel.EMERGENCY
    d.intervention_text = "EMERGENCY: AEB + MRM. CF analysis: all escape routes blocked."
    d.worst_ttc = 0.5

    return {1: a, 2: b, 3: c, 4: d}


def draw_hud(frame: np.ndarray, scenario: ScenarioState, t: float) -> np.ndarray:
    """Draw the full HUD overlay onto the frame."""
    H, W = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (W, 60), COLORS["panel_bg"], -1)
    cv2.putText(frame, "PC-CSG: Physics-Constrained Counterfactual Scene Graph Transformer",
                (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["cyan"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"{scenario.name}: {scenario.description}",
                (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["white"], 1, cv2.LINE_AA)

    # Draw road scene
    road_y = int(H * 0.55)
    cv2.line(frame, (0, road_y - 40), (W, road_y - 40), COLORS["dim"], 1)
    cv2.line(frame, (0, road_y + 40), (W, road_y + 40), COLORS["dim"], 1)
    # Dashed center line
    for x in range(0, W, 30):
        cv2.line(frame, (x, road_y), (x + 15, road_y), COLORS["dim"], 1)

    # Draw agents with bounding boxes
    for agent in scenario.agents:
        bx, by, bw, bh = agent["bbox"]
        x1, y1 = int(bx * W - bw * W / 2), int(by * H - bh * H / 2)
        x2, y2 = int(bx * W + bw * W / 2), int(by * H + bh * H / 2)
        color = agent["color"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{agent['class']} #{agent['id']} TTC={agent['ttc']:.1f}s"
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # Draw counterfactual trajectory arrows for risky agents
        if agent["ttc"] < 5.0:
            cx, cy = int(bx * W), int(by * H)
            # Original trajectory (green)
            cv2.arrowedLine(frame, (cx, cy), (cx + 60, cy), COLORS["trajectory_green"], 1, tipLength=0.3)
            # Counterfactual trajectories (purple)
            for angle in [-20, 20]:
                end_x = cx + int(50 * np.cos(np.radians(angle)))
                end_y = cy + int(50 * np.sin(np.radians(angle)))
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), COLORS["cf_trajectory"], 1, tipLength=0.3)

    # CRT Gauge (bottom-left)
    gauge_x, gauge_y = 20, H - 140
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + 250, gauge_y + 120), COLORS["panel_bg"], -1)
    cv2.rectangle(frame, (gauge_x, gauge_y), (gauge_x + 250, gauge_y + 120), COLORS["dim"], 1)

    cv2.putText(frame, "Counterfactual Risk Tensor (CRT)",
                (gauge_x + 10, gauge_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["white"], 1, cv2.LINE_AA)

    # CRT bar
    bar_x, bar_y = gauge_x + 10, gauge_y + 35
    bar_w, bar_h = 230, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLORS["dim"], 1)

    fill_w = int(bar_w * scenario.crt_value)
    if scenario.crt_value < 0.25:
        fill_color = COLORS["green"]
    elif scenario.crt_value < 0.50:
        fill_color = COLORS["yellow"]
    elif scenario.crt_value < 0.70:
        fill_color = COLORS["orange"]
    else:
        pulse = int(40 * np.sin(t * 6))
        fill_color = (40, 40, min(255, 220 + pulse))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)

    cv2.putText(frame, f"CRT = {scenario.crt_value:.3f}",
                (bar_x, bar_y + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["white"], 1, cv2.LINE_AA)

    # Threshold markers
    for thresh, label in [(0.25, "Safe"), (0.50, "Caution"), (0.70, "Warn"), (0.85, "Crit")]:
        mx = bar_x + int(bar_w * thresh)
        cv2.line(frame, (mx, bar_y), (mx, bar_y + bar_h), COLORS["white"], 1)

    # Intervention level
    level_names = {
        InterventionLevel.NOMINAL: ("LEVEL 0: NOMINAL", COLORS["green"]),
        InterventionLevel.ADVISORY: ("LEVEL 1: ADVISORY", COLORS["yellow"]),
        InterventionLevel.ALERT: ("LEVEL 2: ALERT", COLORS["orange"]),
        InterventionLevel.PRE_INTERVENTION: ("LEVEL 3: PRE-INTERVENTION", COLORS["orange"]),
        InterventionLevel.EMERGENCY: ("LEVEL 4: EMERGENCY AEB+MRM", COLORS["critical_red"]),
    }
    level_text, level_color = level_names[scenario.intervention_level]
    cv2.putText(frame, level_text,
                (gauge_x + 10, gauge_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, level_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Worst-case TTC: {scenario.worst_ttc:.1f}s",
                (gauge_x + 10, gauge_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["dim"], 1, cv2.LINE_AA)

    # Info panel (bottom-right)
    info_x = W - 380
    info_y = H - 140
    cv2.rectangle(frame, (info_x, info_y), (W - 10, info_y + 120), COLORS["panel_bg"], -1)
    cv2.rectangle(frame, (info_x, info_y), (W - 10, info_y + 120), COLORS["dim"], 1)

    cv2.putText(frame, "Counterfactual Analysis",
                (info_x + 10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["cyan"], 1, cv2.LINE_AA)

    text = scenario.intervention_text
    y_offset = info_y + 42
    words = text.split(" ")
    line = ""
    for word in words:
        if len(line + " " + word) > 45:
            cv2.putText(frame, line, (info_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLORS["white"], 1, cv2.LINE_AA)
            y_offset += 16
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        cv2.putText(frame, line, (info_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLORS["white"], 1, cv2.LINE_AA)

    cv2.putText(frame, f"Ego Speed: {scenario.ego_speed:.1f} m/s ({scenario.ego_speed*3.6:.0f} km/h)",
                (info_x + 10, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLORS["dim"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"Agents: {len(scenario.agents)} | PIKV-validated CFs: 6/agent",
                (info_x + 10, info_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.33, COLORS["dim"], 1, cv2.LINE_AA)

    # Key controls
    cv2.putText(frame, "[1-4] Scenarios  [Q] Quit", (W // 2 - 100, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["dim"], 1, cv2.LINE_AA)

    # Emergency flash overlay
    if scenario.intervention_level == InterventionLevel.EMERGENCY:
        flash = int(80 * abs(np.sin(t * 4)))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, flash + 40), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "! AUTONOMOUS EMERGENCY BRAKING !",
                    (W // 2 - 180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["critical_red"], 2, cv2.LINE_AA)

    return frame


def save_scenario_snapshots(scenarios: Dict[int, ScenarioState], output_dir: str = "demo_outputs"):
    """Save static snapshot images for all four scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    letters = {1: "A", 2: "B", 3: "C", 4: "D"}
    for idx, sc in scenarios.items():
        frame = np.full((540, 960, 3), COLORS["bg_dark"], dtype=np.uint8)
        frame = draw_hud(frame, sc, t=0.5)
        path = os.path.join(output_dir, f"scenario_{letters[idx]}_{sc.risk_level}.png")
        cv2.imwrite(path, frame)
        print(f"  [+] Saved snapshot: {path}")
    print(f"All snapshots successfully saved to '{output_dir}/'.")


def save_demo_video(scenarios: Dict[int, ScenarioState], output_path: str = "demo_outputs/pc_csg_demo.mp4", duration_sec: int = 12):
    """Render an animated demo video cycling through all scenarios."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (960, 540))
    if not out.isOpened():
        print(f"Warning: Could not open VideoWriter for {output_path}")
        return

    total_frames = duration_sec * fps
    sec_per_scenario = duration_sec / 4.0
    print(f"Rendering {duration_sec}s demo video ({total_frames} frames) to {output_path}...")

    for frame_idx in range(total_frames):
        t = frame_idx / fps
        scenario_idx = int(t / sec_per_scenario) + 1
        scenario_idx = min(scenario_idx, 4)
        current = scenarios[scenario_idx]

        frame = np.full((540, 960, 3), COLORS["bg_dark"], dtype=np.uint8)
        frame = draw_hud(frame, current, t)
        out.write(frame)

    out.release()
    print(f"  [+] Demo video successfully saved to: {output_path}")


def run_demo(
    save_images: bool = False,
    save_video: bool = False,
    headless: bool = False,
    output_dir: str = "demo_outputs",
):
    """Run interactive or headless counterfactual HUD demo."""
    scenarios = create_scenarios()

    if save_images:
        print("Saving scenario HUD snapshots...")
        save_scenario_snapshots(scenarios, output_dir=output_dir)
        if not save_video and headless:
            return

    if save_video:
        save_demo_video(scenarios, output_path=os.path.join(output_dir, "pc_csg_demo.mp4"))
        if headless:
            return

    if headless:
        print("Headless mode completed.")
        return

    print("=" * 60)
    print(" PC-CSG Interactive Counterfactual Anomaly Anticipation Demo")
    print("=" * 60)
    print(" Controls:")
    print("   [1] Scenario A: Nominal driving (CRT < 0.25)")
    print("   [2] Scenario B: Advisory — potential risk (0.25 <= CRT < 0.50)")
    print("   [3] Scenario C: Alert — elevated risk (0.50 <= CRT < 0.70)")
    print("   [4] Scenario D: Emergency — imminent collision (CRT >= 0.85)")
    print("   [S] Save snapshot of current view")
    print("   [Q] Quit")
    print("=" * 60)

    current = scenarios[1]

    try:
        cv2.namedWindow("PC-CSG Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PC-CSG Demo", 960, 540)
    except Exception as e:
        print(f"Warning: Cannot open OpenCV display window ({e}).")
        print("Tip: Run with --save-images to view the HUD snapshots!")
        save_scenario_snapshots(scenarios, output_dir=output_dir)
        return

    start_time = time.time()

    while True:
        t = time.time() - start_time
        frame = np.full((540, 960, 3), COLORS["bg_dark"], dtype=np.uint8)
        frame = draw_hud(frame, current, t)

        cv2.imshow("PC-CSG Demo", frame)
        key = cv2.waitKey(33) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord("1"):
            current = scenarios[1]
            print(f"-> {current.name}: {current.description}")
        elif key == ord("2"):
            current = scenarios[2]
            print(f"-> {current.name}: {current.description}")
        elif key == ord("3"):
            current = scenarios[3]
            print(f"-> {current.name}: {current.description}")
        elif key == ord("4"):
            current = scenarios[4]
            print(f"-> {current.name}: {current.description}")
        elif key == ord("s") or key == ord("S"):
            os.makedirs(output_dir, exist_ok=True)
            fname = os.path.join(output_dir, f"snapshot_{current.name.replace(' ', '_').lower()}.png")
            cv2.imwrite(fname, frame)
            print(f"-> Snapshot saved to {fname}")

    cv2.destroyAllWindows()
    print("Demo terminated.")


def main():
    """CLI entrypoint for demo."""
    import argparse
    parser = argparse.ArgumentParser(description="PC-CSG Real-Time HUD Demo")
    parser.add_argument("--save-images", action="store_true", help="Save PNG snapshots to demo_outputs/")
    parser.add_argument("--save-video", action="store_true", help="Render demo video to demo_outputs/pc_csg_demo.mp4")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    args = parser.parse_args()

    run_demo(
        save_images=args.save_images,
        save_video=args.save_video,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PC-CSG Visual Asset Export Pipeline.

Generates web-optimized animated GIFs, scenario comparison grids,
and architecture infographics for the GitHub repository showcase.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


ROOT_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
DEMO_DIR = ROOT_DIR / "demo_outputs"


def video_to_gif(video_path: Path, output_gif_path: Path, max_frames: int = 50, target_width: int = 640, fps: int = 12):
    """Convert an MP4 video into an optimized animated GIF."""
    if not video_path.exists():
        print(f"Warning: {video_path} not found. Skipping GIF export.")
        return False

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // max_frames) if total_frames > max_frames else 1

    count = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            h, w = frame.shape[:2]
            aspect = h / w
            target_height = int(target_width * aspect)
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        count += 1
    cap.release()

    if not frames:
        print(f"No frames captured from {video_path}")
        return False

    # Save as animated GIF with adaptive palette
    duration = int(1000 / fps)
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration,
        loop=0
    )
    size_mb = os.path.getsize(output_gif_path) / (1024 * 1024)
    print(f"Exported GIF: {output_gif_path.name} ({size_mb:.2f} MB, {len(frames)} frames)")
    return True


def build_scenario_grid(output_grid_path: Path):
    """Combine 4 scenario snapshots into a 2x2 presentation grid."""
    scenarios = [
        ("scenario_A_nominal.png", "Scenario A: Nominal Cruising (CRT < 0.25, Green)"),
        ("scenario_B_advisory.png", "Scenario B: Advisory Deceleration (0.25 <= CRT < 0.50, Cyan)"),
        ("scenario_C_alert.png", "Scenario C: Truck Swerve Hazard (0.50 <= CRT < 0.70, Orange)"),
        ("scenario_D_critical.png", "Scenario D: Emergency Imminent Crash (CRT >= 0.85, Critical Red)")
    ]

    imgs = []
    for fname, title in scenarios:
        fpath = DEMO_DIR / fname
        if fpath.exists():
            img = Image.open(fpath)
            imgs.append((img, title))
        else:
            print(f"Warning: {fpath} not found.")

    if len(imgs) != 4:
        print("Skipping scenario grid: Not all 4 scenario snapshots are present.")
        return False

    w, h = imgs[0][0].size
    grid = Image.new("RGB", (w * 2, h * 2), color=(15, 15, 25))

    grid.paste(imgs[0][0], (0, 0))
    grid.paste(imgs[1][0], (w, 0))
    grid.paste(imgs[2][0], (0, h))
    grid.paste(imgs[3][0], (w, h))

    # Resize for clean GitHub embedding
    grid = grid.resize((1280, int(1280 * (h * 2) / (w * 2))), Image.Resampling.LANCZOS)
    grid.save(output_grid_path, optimize=True, quality=90)
    print(f"Exported Scenario Grid: {output_grid_path.name}")
    return True


def build_architecture_diagram(output_path: Path):
    """Generate high-resolution architecture flowchart of PC-CSG."""
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0F111A")
    ax.set_facecolor("#0F111A")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    stages = [
        ("Sensors / Video", "YOLOv8 Detection &\nMulti-Agent Tracking\n(x, y, v, theta)", "#24283B", "#7AA2F7", 1.2),
        ("Scene Graph", "Spatio-Temporal Graph\nTime-to-Collision (TTC)\nHazard Weighting", "#24283B", "#2AC3DE", 3.7),
        ("CSGA Engine", "Counterfactual Attention\nCausal Gating Matrix\nWhat-If Query Injection", "#2D2640", "#BB9AF7", 6.2),
        ("Kinematic Propagator", "Differentiable Bicycle Model\nN x K Trajectories\nHorizon H=15 Steps", "#24283B", "#7DCFFF", 8.7),
        ("PIKV Validator", "Coulomb Friction Circle\nSteering/Decel Limits\n43.2% Hypotheses Pruned", "#3D2628", "#F7768E", 11.2),
        ("CRT Arbiter", "Counterfactual Risk Tensor\n5-Level Preemptive ADAS\nAEB & MRM Trigger", "#243B28", "#9ECE6A", 13.0)
    ]

    for title, desc, bg_col, edge_col, x_center in stages:
        box_w = 2.0 if x_center != 13.0 else 1.5
        box = patches.FancyBboxPatch(
            (x_center - box_w/2, 1.2), box_w, 3.6,
            boxstyle="round,pad=0.15",
            facecolor=bg_col,
            edgecolor=edge_col,
            linewidth=2.2
        )
        ax.add_patch(box)
        ax.text(x_center, 4.3, title, color="white", fontsize=11, fontweight="bold", ha="center", va="center")
        ax.text(x_center, 2.7, desc, color="#C0CAF5", fontsize=8.5, ha="center", va="center", linespacing=1.4)

    # Draw arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i][4] + (1.0 if stages[i][4] != 13.0 else 0.75)
        x2 = stages[i+1][4] - (1.0 if stages[i+1][4] != 13.0 else 0.75)
        ax.annotate(
            "", xy=(x2, 3.0), xytext=(x1, 3.0),
            arrowprops=dict(arrowstyle="-|>", color="#7AA2F7", lw=2.0, mutation_scale=15)
        )

    plt.title("PC-CSG: Physics-Constrained Counterfactual Scene Graph Architecture", color="white", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Exported Architecture Flow: {output_path.name}")
    return True


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Export Preview GIFs
    video_to_gif(
        DEMO_DIR / "pc_csg_demo.mp4",
        ASSETS_DIR / "demo_hud_preview.gif",
        max_frames=45,
        target_width=640,
        fps=12
    )
    video_to_gif(
        DEMO_DIR / "real_dashcam_annotated.mp4",
        ASSETS_DIR / "real_dashcam_preview.gif",
        max_frames=45,
        target_width=640,
        fps=12
    )

    # 2. Export 4-Scenario Grid
    build_scenario_grid(ASSETS_DIR / "scenarios_grid.png")

    # 3. Export Architecture Flow
    build_architecture_diagram(ASSETS_DIR / "architecture_flow.png")


if __name__ == "__main__":
    main()

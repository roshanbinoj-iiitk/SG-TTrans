#!/usr/bin/env python3
"""
ST-HGST Interactive Dual-Stream Visual Dashboard & HUD.
Displays synchronized in-cabin gaze tracking and exterior scene graph with real-time Level-3 Takeover Arbitration.
"""

import argparse
import time
import cv2
import numpy as np
import torch
from sg_ttrans.config import STHGSTConfig
from sg_ttrans.models.st_hgst_net import STHGSTNetwork
from sg_ttrans.cognitive.leaky_accumulator import LeakyCognitiveAccumulator
from sg_ttrans.cognitive.epistemic_gap import compute_eag
from sg_ttrans.cognitive.takeover_arbitrator import TakeoverArbitrator
from sg_ttrans.types import TakeoverAction

def draw_hud(frame, scenario_name, eag, decision, fps):
    h, w, _ = frame.shape
    # Top overlay bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & FPS
    cv2.putText(frame, "ST-HGST: Level-3 Cognitive Takeover Monitor", (15, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Throughput: {fps:.1f} FPS", (w - 220, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Scenario: {scenario_name} (Keys: [1] Aligned, [2] Blindness, [3] Critical MRM, [Q] Exit)",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # EAG Progress Bar
    bar_x, bar_y, bar_w, bar_h = 15, h - 50, 300, 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(np.clip(eag, 0.0, 1.0) * bar_w)
    
    # Color based on action
    if decision.action == TakeoverAction.SAFE_HANDOVER:
        color = (0, 220, 0)
        action_text = "LEVEL 0: SAFE TORQUE HANDOVER AUTHORIZED"
    elif decision.action == TakeoverAction.TARGETED_SPATIAL_CUE:
        color = (0, 180, 255)
        action_text = f"LEVEL 1: SPATIAL HUD CUE ON HAZARD #{decision.causal_hazard_id}"
    else:
        color = (0, 0, 255)
        action_text = "LEVEL 2: AUTONOMOUS MINIMUM RISK MANEUVER (MRM)"

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    cv2.putText(frame, f"EAG: {eag:.3f}", (bar_x + 10, bar_y + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Action Banner
    cv2.putText(frame, action_text, (bar_x + bar_w + 30, bar_y + 18),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2, cv2.LINE_AA)

def run_demo():
    print("\nStarting ST-HGST Dual-Stream Real-Time Stream Demo...")
    cfg = STHGSTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STHGSTNetwork().to(device).eval()
    accumulator = LeakyCognitiveAccumulator(cfg)
    arbitrator = TakeoverArbitrator(cfg)

    scenario_mode = "aligned"
    scenario_names = {
        "aligned": "A: Aligned Attention (Smooth Handover)",
        "blindness": "B: Inattentional Blindness (Spatial Cue)",
        "distracted": "C: Critical Distraction / Sleep (AEB MRM)",
    }

    width, height = 1280, 600
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    frame_count = 0
    t0 = time.perf_counter()
    fps = 60.0

    print("Controls: Press [1] for Aligned, [2] for Inattentional Blindness, [3] for Critical Distraction, [Q] to quit.")

    while True:
        frame_count += 1
        t_now = time.perf_counter()
        if t_now - t0 >= 1.0:
            fps = frame_count / (t_now - t0)
            frame_count = 0
            t0 = t_now

        # 1. Synthesize exterior road view (Right) & cabin view (Left)
        canvas[:] = (30, 30, 30)
        cabin_view = canvas[:, :640]
        road_view = canvas[:, 640:]

        # Road divider
        cv2.line(canvas, (640, 80), (640, height), (70, 70, 70), 2)
        cv2.putText(canvas, "[IN-CABIN DRIVER STREAM]", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        cv2.putText(canvas, "[EXTERIOR ROAD SCENE GRAPH]", (660, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Hazard obstacle on road (car ahead at x=960, y=350)
        hz_x, hz_y = 960, 350
        cv2.rectangle(canvas, (hz_x - 60, hz_y - 40), (hz_x + 60, hz_y + 40), (0, 0, 200), 2)
        cv2.putText(canvas, "Hazard #0: Cut-In Car (TTC=1.8s)", (hz_x - 90, hz_y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 150, 255), 1)

        # Ambient vehicle (far right)
        cv2.rectangle(canvas, (1150, 280), (1230, 340), (150, 150, 150), 1)
        cv2.putText(canvas, "Vehicle #1 (TTC=9.0s)", (1140, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Gaze setting based on scenario
        if scenario_mode == "aligned":
            gaze_road_pt = (hz_x, hz_y)
            gaze_token = torch.tensor([[[0.0, 0.0, 1.0, 0.5, 0.58]]], device=device)
            status_text = "Driver Focus: Fixated on Causal Hazard #0"
        elif scenario_mode == "blindness":
            gaze_road_pt = (1190, 310)  # Looking at harmless ambient car
            gaze_token = torch.tensor([[[0.5, 0.0, 0.86, 0.85, 0.51]]], device=device)
            status_text = "Driver Focus: Fixated on Harmless Vehicle #1 (Missed Hazard #0)"
        else:
            gaze_road_pt = (320, 520)   # Looking down inside cabin
            gaze_token = torch.tensor([[[0.0, -0.9, 0.4, 0.5, 0.95]]], device=device)
            status_text = "Driver Focus: Down at Phone / Eyes Closed"

        # Draw Gaze in Cabin View
        face_center = (320, 320)
        cv2.circle(canvas, face_center, 80, (80, 80, 80), -1)
        cv2.circle(canvas, (290, 300), 12, (200, 200, 200), -1)
        cv2.circle(canvas, (350, 300), 12, (200, 200, 200), -1)
        
        # Pupil gaze direction
        if scenario_mode == "aligned":
            pupil_offset = (0, 0)
        elif scenario_mode == "blindness":
            pupil_offset = (6, 0)
        else:
            pupil_offset = (0, 8)
        cv2.circle(canvas, (290 + pupil_offset[0], 300 + pupil_offset[1]), 5, (0, 0, 0), -1)
        cv2.circle(canvas, (350 + pupil_offset[0], 300 + pupil_offset[1]), 5, (0, 0, 0), -1)

        # Draw gaze fixation target on road
        if scenario_mode in ("aligned", "blindness"):
            cv2.circle(canvas, gaze_road_pt, 18, (0, 255, 255), 2)
            cv2.line(canvas, (gaze_road_pt[0] - 25, gaze_road_pt[1]), (gaze_road_pt[0] + 25, gaze_road_pt[1]), (0, 255, 255), 1)
            cv2.line(canvas, (gaze_road_pt[0], gaze_road_pt[1] - 25), (gaze_road_pt[0], gaze_road_pt[1] + 25), (0, 255, 255), 1)

        cv2.putText(canvas, status_text, (20, height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # 2. Run ST-HGST inference
        nodes_token = torch.tensor([[[0.0, 0.5, 0.58, 0.15, 0.25, 0.0, -5.0, 1.8],
                                     [1.0, 0.85, 0.51, 0.10, 0.15, 0.0, -0.5, 9.0]]], device=device)
        with torch.no_grad():
            out = model(gaze_token, nodes_token)
        attn = out["attention_weights"].squeeze(0).cpu().numpy()
        cog = accumulator.step(attn)

        # If spatial cue active, draw directional spatial pulse
        hazard_weights = np.array([0.88, 0.05], dtype=np.float32)
        min_ttc = 1.8 if scenario_mode != "distracted" else 1.0
        eag, top_h = compute_eag(hazard_weights, cog.accumulated_cognition)
        decision = arbitrator.arbitrate(eag, min_ttc=min_ttc, top_hazard_idx=top_h)

        if decision.action == TakeoverAction.TARGETED_SPATIAL_CUE:
            cv2.circle(canvas, (hz_x, hz_y), 75 + int(10 * np.sin(time.time() * 10)), (0, 180, 255), 3)
            cv2.putText(canvas, ">> LOOK HERE <<", (hz_x - 65, hz_y + 65),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 180, 255), 1)

        # Draw HUD
        draw_hud(canvas, scenario_names[scenario_mode], eag, decision, fps)

        cv2.imshow("ST-HGST Takeover Monitor", canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            scenario_mode = "aligned"
            accumulator.reset()
        elif key == ord('2'):
            scenario_mode = "blindness"
            accumulator.reset()
        elif key == ord('3'):
            scenario_mode = "distracted"
            accumulator.reset()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()

#!/usr/bin/env python3
"""
ST-HGST Real-Visuals Dual-Stream Dashboard & HUD.
Integrates real in-cabin driver video with real exterior road traffic dashcam footage,
demonstrating live 3D gaze estimation, real-time object scene graph, and Level-3 Takeover Arbitration.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from sg_ttrans.config import STHGSTConfig
from sg_ttrans.geometry.facemesh import FaceMeshExtractor
from sg_ttrans.geometry.head_pose import HeadPoseEstimator
from sg_ttrans.models.st_hgst_net import STHGSTNetwork
from sg_ttrans.cognitive.leaky_accumulator import LeakyCognitiveAccumulator
from sg_ttrans.cognitive.epistemic_gap import compute_eag
from sg_ttrans.cognitive.takeover_arbitrator import TakeoverArbitrator
from sg_ttrans.types import TakeoverAction, GazeVector

def draw_hud_header(canvas, fps, driver_label, road_label):
    h, w, _ = canvas.shape
    # Top overlay banner
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

    cv2.putText(canvas, "ST-HGST: Level-3 Cognitive Saliency & Takeover Monitor", (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Throughput: {fps:.1f} FPS (GPU: RTX 3050)", (w - 340, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (140, 255, 140), 1, cv2.LINE_AA)

    # Subtitles
    cv2.putText(canvas, f"Cabin: {driver_label}  |  Road: {road_label}", (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Controls: [1] Attentive Driver | [2] Drowsy Driver | [SPACE] Pause | [Q] Quit",
                (w - 580, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

def draw_arbitration_bar(canvas, eag, decision):
    h, w, _ = canvas.shape
    bar_x, bar_y, bar_w, bar_h = 25, h - 55, 340, 28
    
    # Progress background
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (35, 35, 40), -1)
    fill_w = int(np.clip(eag, 0.0, 1.0) * bar_w)

    if decision.action == TakeoverAction.SAFE_HANDOVER:
        color = (0, 225, 0)
        action_title = "LEVEL 0: SAFE TORQUE HANDOVER AUTHORIZED"
        sub_text = "Driver attention aligned with critical road hazards"
    elif decision.action == TakeoverAction.TARGETED_SPATIAL_CUE:
        color = (0, 185, 255)
        action_title = f"LEVEL 1: SPATIAL HUD CUE ON HAZARD #{decision.causal_hazard_id}"
        sub_text = "Inattentional blindness: Attention mismatch on lead hazard"
    else:
        color = (40, 40, 255)
        action_title = "LEVEL 2: AUTONOMOUS MINIMUM RISK MANEUVER (MRM)"
        sub_text = "Severe cognitive distraction / Imminent collision threshold"

    # Gradient fill bar
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    cv2.putText(canvas, f"EAG: {eag:.3f}", (bar_x + 12, bar_y + 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Decision Banner Text
    cv2.putText(canvas, action_title, (bar_x + bar_w + 25, bar_y + 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, sub_text, (bar_x + bar_w + 25, bar_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)

def run_real_visuals_demo(
    road_path: str,
    cabin_paths: list,
    headless: bool = False,
    max_frames: int = 0,
    output_path: str = None
):
    print("=" * 65)
    print(" ST-HGST Dual-Stream Real-Visuals Takeover Dashboard")
    print("=" * 65)
    print(f" Loading Road Video:   {road_path}")
    print(f" Loading Cabin Videos: {cabin_paths}")

    cfg = STHGSTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Neural Inference Engine running on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load Deep Learning Models
    st_hgst_net = STHGSTNetwork(embed_dim=cfg.embed_dim, num_heads=cfg.num_heads).to(device).eval()
    facemesh = FaceMeshExtractor()
    pose_estimator = HeadPoseEstimator()
    yolo_model = YOLO("yolov8n.pt")
    accumulator = LeakyCognitiveAccumulator(cfg)
    arbitrator = TakeoverArbitrator(cfg)

    # Video Captures
    cap_road = cv2.VideoCapture(road_path)
    if not cap_road.isOpened():
        print(f"Error: Could not open road video at {road_path}")
        sys.exit(1)

    cabin_idx = 0
    cap_cabin = cv2.VideoCapture(cabin_paths[cabin_idx])
    driver_labels = [
        "Driver 1 (Active / Attentive)",
        "Driver 2 (Drowsy / Inattentive)",
    ]

    out_w, out_h = 1360, 680
    pane_w = out_w // 2
    pane_h = out_h - 150  # Height between header and bottom bar

    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, 25.0, (out_w, out_h))
        print(f" Recording dashboard output to: {output_path}")

    paused = False
    prev_time = time.perf_counter()
    fps = 30.0

    # Smoothing states
    smooth_gaze_x, smooth_gaze_y = pane_w + pane_w // 2, 80 + pane_h // 2
    prev_distances = {}
    frame_count = 0

    while True:
        if max_frames > 0 and frame_count >= max_frames:
            print(f" Reached maximum requested frames ({max_frames}). Stopping.")
            break

        if not paused:
            ret_road, frame_road = cap_road.read()
            if not ret_road:
                cap_road.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_road, frame_road = cap_road.read()

            ret_cabin, frame_cabin = cap_cabin.read()
            if not ret_cabin:
                cap_cabin.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_cabin, frame_cabin = cap_cabin.read()

            frame_count += 1

            t_now = time.perf_counter()
            dt = t_now - prev_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            prev_time = t_now

        # Prepare composite canvas
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        # -------------------------------------------------------------
        # 1. PROCESS IN-CABIN DRIVER STREAM
        # -------------------------------------------------------------
        disp_cabin = cv2.resize(frame_cabin, (pane_w, pane_h))
        lm_68, face_found = facemesh.extract(disp_cabin)
        
        gaze_yaw, gaze_pitch, gaze_roll = 0.0, -50.0, 0.0
        ear_val = 0.32

        if face_found and lm_68 is not None:
            cw, ch = pane_w, pane_h
            pts_6 = lm_68[[30, 8, 36, 45, 48, 54], :2] * np.array([cw, ch])
            gaze_yaw, gaze_pitch, gaze_roll = pose_estimator.estimate(pts_6, cw, ch)

            # Draw facial landmarks
            for pt in lm_68:
                px, py = int(pt[0] * cw), int(pt[1] * ch)
                cv2.circle(disp_cabin, (px, py), 1, (0, 255, 200), -1)

            # Draw gaze direction ray from nose tip (idx 30)
            nose_px = int(lm_68[30, 0] * cw)
            nose_py = int(lm_68[30, 1] * ch)
            ray_dx = int(np.clip(gaze_yaw * 3.0, -80, 80))
            ray_dy = int(np.clip((gaze_pitch - (-50.0)) * 2.5, -60, 60))
            cv2.arrowedLine(disp_cabin, (nose_px, nose_py),
                            (nose_px + ray_dx, nose_py + ray_dy), (0, 140, 255), 3, tipLength=0.3)

            # Draw bottom telemetry badge
            cv2.rectangle(disp_cabin, (10, ch - 35), (320, ch - 10), (18, 18, 24), -1)
            cv2.rectangle(disp_cabin, (10, ch - 35), (320, ch - 10), (60, 60, 75), 1)
            cv2.putText(disp_cabin, f"Head Pose: Yaw {gaze_yaw:+.1f} deg | Pitch {gaze_pitch:+.1f} deg",
                        (18, ch - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 240), 1, cv2.LINE_AA)

        # Map 3D gaze to Road Frame coordinates
        norm_yaw = np.clip(gaze_yaw / 35.0, -1.0, 1.0)
        norm_pitch = np.clip((gaze_pitch - (-50.0)) / 25.0, -1.0, 1.0)
        target_gaze_x = pane_w + int((0.5 + 0.45 * norm_yaw) * pane_w)
        target_gaze_y = 80 + int((0.5 + 0.40 * norm_pitch) * pane_h)

        # Smooth gaze point
        smooth_gaze_x = int(0.75 * smooth_gaze_x + 0.25 * target_gaze_x)
        smooth_gaze_y = int(0.75 * smooth_gaze_y + 0.25 * target_gaze_y)

        # -------------------------------------------------------------
        # 2. PROCESS EXTERIOR ROAD SCENE WITH YOLO
        # -------------------------------------------------------------
        disp_road = cv2.resize(frame_road, (pane_w, pane_h))
        yolo_res = yolo_model(disp_road, verbose=False)
        boxes = yolo_res[0].boxes

        # Filter road vehicles & pedestrians (classes: 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck)
        valid_classes = {0, 1, 2, 3, 5, 7}
        detected_objects = []

        for b_idx, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            if cls_id in valid_classes:
                conf = float(box.conf[0])
                if conf >= 0.25:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    norm_x, norm_y, norm_w, norm_h = box.xywhn[0].cpu().numpy()
                    
                    # Estimated monocular distance from bounding box height
                    est_dist = max(4.0, float(18.0 / max(norm_h, 0.05)))
                    
                    # Compute relative velocity & TTC
                    prev_d = prev_distances.get(b_idx, est_dist)
                    rel_v = (est_dist - prev_d) * 30.0  # m/s
                    prev_distances[b_idx] = est_dist

                    ttc = float(est_dist / max(-rel_v, 0.5)) if rel_v < -0.2 else float(est_dist / 4.5)
                    
                    # Intrinsic Hazard Weight H(v_i)
                    sig_val = np.clip((ttc - cfg.tau_crit) / cfg.lambda_scale, -30.0, 30.0)
                    h_weight = float(1.0 / (1.0 + np.exp(sig_val)))

                    detected_objects.append({
                        "id": b_idx,
                        "class_id": cls_id,
                        "name": yolo_model.names[cls_id],
                        "bbox_px": (int(x1), int(y1), int(x2), int(y2)),
                        "bbox_norm": (norm_x, norm_y, norm_w, norm_h),
                        "distance": est_dist,
                        "ttc": ttc,
                        "hazard_weight": h_weight,
                    })

        # -------------------------------------------------------------
        # 3. ST-HGST NEURAL CROSS-ATTENTION & ARBITRATION
        # -------------------------------------------------------------
        N = max(1, len(detected_objects))
        node_tensor = torch.zeros(1, N, 8, device=device)
        hazard_weights = np.zeros(N, dtype=np.float32)
        ttc_list = np.zeros(N, dtype=np.float32)

        gaze_u = float(np.clip((smooth_gaze_x - pane_w) / pane_w, 0.0, 1.0))
        gaze_v = float(np.clip((smooth_gaze_y - 80) / pane_h, 0.0, 1.0))
        gaze_tensor = torch.tensor([[[norm_yaw, norm_pitch, 1.0, gaze_u, gaze_v]]],
                                   device=device, dtype=torch.float32)

        for i, obj in enumerate(detected_objects):
            nx, ny, nw, nh = obj["bbox_norm"]
            node_tensor[0, i] = torch.tensor([obj["class_id"], nx, ny, nw, nh, 0.0, -1.0, obj["ttc"]])
            hazard_weights[i] = obj["hazard_weight"]
            ttc_list[i] = obj["ttc"]

        with torch.no_grad():
            out = st_hgst_net(gaze_tensor, node_tensor)
        attn_weights = out["attention_weights"].squeeze(0).cpu().numpy()

        # Leaky cognitive accumulator
        cog_state = accumulator.step(attn_weights)
        eag, top_hazard = compute_eag(hazard_weights, cog_state.accumulated_cognition)
        min_ttc = float(np.min(ttc_list)) if len(detected_objects) > 0 else 10.0
        decision = arbitrator.arbitrate(eag, min_ttc=min_ttc, top_hazard_idx=top_hazard)

        # -------------------------------------------------------------
        # 4. RENDER VISUAL OVERLAYS
        # -------------------------------------------------------------
        # Place Cabin & Road streams
        canvas[80:80 + pane_h, :pane_w] = disp_cabin
        canvas[80:80 + pane_h, pane_w:] = disp_road

        # Divider line
        cv2.line(canvas, (pane_w, 80), (pane_w, 80 + pane_h), (80, 80, 90), 2)

        # Left pane title badge
        cv2.rectangle(canvas, (16, 92), (370, 122), (20, 20, 26), -1)
        cv2.rectangle(canvas, (16, 92), (370, 122), (65, 65, 78), 1)
        cv2.putText(canvas, "CABIN STREAM: 3D GAZE & MESH", (26, 113),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (220, 240, 255), 1, cv2.LINE_AA)

        # Right pane title badge
        cv2.rectangle(canvas, (pane_w + 16, 92), (pane_w + 430, 122), (20, 20, 26), -1)
        cv2.rectangle(canvas, (pane_w + 16, 92), (pane_w + 430, 122), (65, 65, 78), 1)
        cv2.putText(canvas, "ROAD SCENE GRAPH: YOLOv8 + ATTENTION", (pane_w + 26, 113),
                    cv2.FONT_HERSHEY_DUPLEX, 0.48, (220, 240, 255), 1, cv2.LINE_AA)

        # Right pane hazard count badge
        cv2.rectangle(canvas, (pane_w + 16, 80 + pane_h - 35), (pane_w + 210, 80 + pane_h - 10), (20, 20, 26), -1)
        cv2.rectangle(canvas, (pane_w + 16, 80 + pane_h - 35), (pane_w + 210, 80 + pane_h - 10), (65, 65, 78), 1)
        cv2.putText(canvas, f"Hazards Tracked: {len(detected_objects)}", (pane_w + 24, 80 + pane_h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)

        # Draw detected objects with attention & hazard styling
        for i, obj in enumerate(detected_objects):
            x1, y1, x2, y2 = obj["bbox_px"]
            rx1, ry1 = pane_w + x1, 80 + y1
            rx2, ry2 = pane_w + x2, 80 + y2

            attn_val = float(attn_weights[i]) if i < len(attn_weights) else 0.0
            cog_val = float(cog_state.accumulated_cognition[i]) if i < len(cog_state.accumulated_cognition) else 0.0
            is_top_hazard = (i == top_hazard and obj["hazard_weight"] > 0.4)

            # Border color based on hazard & attention
            if is_top_hazard:
                box_color = (0, 60, 255)  # Red for primary threat
                thickness = 3
            elif obj["hazard_weight"] > 0.5:
                box_color = (0, 160, 255)  # Orange for moderate hazard
                thickness = 2
            else:
                box_color = (200, 200, 200) # Neutral
                thickness = 1

            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), box_color, thickness)
            
            # Label banner
            lbl = f"{obj['name']} | TTC:{obj['ttc']:.1f}s | Attn:{attn_val:.2f}"
            cv2.putText(canvas, lbl, (rx1, max(95, ry1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)

            # If Level 1 Spatial Cue is active on this object, draw targeted pulsing beacon
            if decision.action == TakeoverAction.TARGETED_SPATIAL_CUE and is_top_hazard:
                pulse_r = int(35 + 8 * np.sin(time.time() * 12))
                cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
                cv2.circle(canvas, (cx, cy), pulse_r, (0, 200, 255), 2)
                cv2.putText(canvas, ">> TARGETED HUD CUE: LOOK HERE <<", (cx - 130, ry1 - 25),
                            cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA)

        # Draw projected driver gaze reticle on road
        cv2.circle(canvas, (smooth_gaze_x, smooth_gaze_y), 16, (0, 255, 255), 2)
        cv2.circle(canvas, (smooth_gaze_x, smooth_gaze_y), 4, (0, 255, 255), -1)
        cv2.line(canvas, (smooth_gaze_x - 22, smooth_gaze_y), (smooth_gaze_x + 22, smooth_gaze_y), (0, 255, 255), 1)
        cv2.line(canvas, (smooth_gaze_x, smooth_gaze_y - 22), (smooth_gaze_x, smooth_gaze_y + 22), (0, 255, 255), 1)
        cv2.putText(canvas, "Gaze Fixation", (smooth_gaze_x + 12, smooth_gaze_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw HUD Headers & Arbitration Bar
        draw_hud_header(canvas, fps, driver_labels[cabin_idx], Path(road_path).name)
        draw_arbitration_bar(canvas, eag, decision)

        # Save frame to output video if recording
        if video_writer is not None:
            video_writer.write(canvas)

        # Show window if not headless
        if not headless:
            cv2.imshow("ST-HGST Level-3 Takeover Arbitration Dashboard", canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('1'):
                cabin_idx = 0
                cap_cabin.release()
                cap_cabin = cv2.VideoCapture(cabin_paths[cabin_idx])
                accumulator.reset()
            elif key == ord('2'):
                cabin_idx = 1
                cap_cabin.release()
                cap_cabin = cv2.VideoCapture(cabin_paths[cabin_idx])
                accumulator.reset()
        else:
            if frame_count % 25 == 0:
                print(f" Frame {frame_count:04d} | FPS: {fps:.1f} | EAG: {eag:.3f} | Decision: {decision.action.name}")

    if video_writer is not None:
        video_writer.release()
        print(f" Saved output video to: {output_path}")

    cap_road.release()
    cap_cabin.release()
    if not headless:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-HGST Real-Visuals Takeover Demo")
    parser.add_argument("--road", type=str, default="data/road_dashcam_real.mp4",
                        help="Path to real road dashcam video")
    parser.add_argument("--cabin1", type=str, default="data/driver_cabin_video.mp4",
                        help="Path to real driver 1 cabin video")
    parser.add_argument("--cabin2", type=str, default="data/driver_cabin_video2.mp4",
                        help="Path to real driver 2 cabin video")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode without opening GUI window")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 for continuous)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save output video (e.g. demo_output.mp4)")
    args = parser.parse_args()

    run_real_visuals_demo(
        road_path=args.road,
        cabin_paths=[args.cabin1, args.cabin2],
        headless=args.headless,
        max_frames=args.max_frames,
        output_path=args.output
    )

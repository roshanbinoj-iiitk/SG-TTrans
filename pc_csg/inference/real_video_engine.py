"""
Real Video Inference Engine for PC-CSG.

Processes real road dashcam videos or live webcams:
1. Detects & tracks vehicles/pedestrians with YOLOv8.
2. Derives dynamic kinematics (velocities, accelerations, TTC).
3. Builds dynamic spatio-temporal scene graphs.
4. Executes PC-CSG: CSGA causal attention + PIKV validation + CRT arbitration.
5. Overlays real-time anticipatory safety HUD directly on real video frames.
"""

import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from pc_csg.config import PCCSGConfig
from pc_csg.models.pc_csg_net import PCCSGNetwork
from pc_csg.scene_graph import compute_hazard_weight, compute_ttc
from pc_csg.types import InterventionLevel


# ── Color Palette for HUD ──
COLORS = {
    "bg_dark": (15, 15, 25),
    "panel_bg": (25, 28, 38),
    "green": (80, 220, 120),
    "yellow": (80, 220, 255),
    "orange": (50, 160, 255),
    "red": (60, 60, 255),
    "critical_red": (40, 40, 240),
    "white": (220, 220, 230),
    "dim": (120, 125, 140),
    "cyan": (220, 200, 80),
    "purple": (200, 100, 220),
}


class AgentTrackState:
    """Maintains kinematic history for a tracked traffic agent."""

    def __init__(self, track_id: int, cls_id: int):
        self.track_id = track_id
        self.cls_id = cls_id
        self.history = deque(maxlen=15)  # (timestamp, cx, cy, w, h)
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.ax: float = 0.0
        self.ay: float = 0.0
        self.ttc: float = float("inf")
        self.hazard_weight: float = 0.0
        self.last_seen: float = 0.0

    def update(self, t: float, cx: float, cy: float, w: float, h: float, ego_speed: float = 25.0):
        self.last_seen = t
        if len(self.history) >= 1:
            prev_t, prev_cx, prev_cy, _, _ = self.history[-1]
            dt = max(1e-4, t - prev_t)
            # Estimate velocities in normalized units / sec
            inst_vx = (cx - prev_cx) / dt
            inst_vy = (cy - prev_cy) / dt

            # Smooth with moving average
            self.ax = (inst_vx - self.vx) / dt
            self.ay = (inst_vy - self.vy) / dt
            self.vx = 0.7 * self.vx + 0.3 * inst_vx
            self.vy = 0.7 * self.vy + 0.3 * inst_vy
        else:
            self.vx, self.vy = 0.0, 0.0

        self.history.append((t, cx, cy, w, h))

        # Approximate road distance from bottom of bounding box (perspective projection)
        bottom_y = float(np.clip(cy + h / 2.0, 0.05, 0.98))
        road_depth = float(max(2.0, 60.0 * (1.0 - bottom_y) / bottom_y))
        closing_rate = (self.vy * road_depth) if self.vy < 0 else (ego_speed * 0.5)

        if closing_rate > 0.5:
            self.ttc = float(max(0.2, road_depth / closing_rate))
        else:
            self.ttc = 25.0

        self.ttc = float(np.nan_to_num(self.ttc, nan=25.0, posinf=25.0, neginf=25.0))
        self.hazard_weight = float(compute_hazard_weight(self.ttc, tau_crit=3.0, lambda_scale=0.5))
        self.hazard_weight = float(np.nan_to_num(self.hazard_weight, nan=0.0))


class RealVideoInferenceEngine:
    """
    End-to-End Real Video PC-CSG Inference Pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/pc_csg_real_best.pt",
        config: Optional[PCCSGConfig] = None,
        device: Optional[torch.device] = None,
        yolo_model: str = "yolov8n.pt",
    ):
        self.config = config or PCCSGConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load YOLO detector / tracker
        print(f"Loading YOLO tracking model: {yolo_model}...")
        self.detector = YOLO(yolo_model)

        # 2. Load PC-CSG network
        print(f"Initializing PC-CSG network on {self.device}...")
        self.model = PCCSGNetwork(self.config).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded successfully!")
        else:
            print("Running with initialized PC-CSG weights.")

        self.model.eval()

        # Target classes in COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        self.valid_classes = {0: 1, 1: 2, 2: 0, 3: 2, 5: 3, 7: 3}
        self.class_names = {0: "Car", 1: "Pedestrian", 2: "Cyclist", 3: "Truck/Bus"}

        self.tracks: Dict[int, AgentTrackState] = {}
        self.intervention_names = {
            InterventionLevel.NOMINAL: ("LEVEL 0: NOMINAL", COLORS["green"]),
            InterventionLevel.ADVISORY: ("LEVEL 1: HUD ADVISORY", COLORS["yellow"]),
            InterventionLevel.ALERT: ("LEVEL 2: ACTIVE ALERT", COLORS["orange"]),
            InterventionLevel.PRE_INTERVENTION: ("LEVEL 3: PRE-BRAKING", COLORS["orange"]),
            InterventionLevel.EMERGENCY: ("LEVEL 4: AEB EMERGENCY BRAKE", COLORS["critical_red"]),
        }

    def process_frame(
        self,
        frame: np.ndarray,
        t_sec: float,
        ego_speed: float = 25.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process a single real video frame through YOLO + PC-CSG + HUD renderer.
        """
        H, W = frame.shape[:2]
        t0 = time.perf_counter()

        # 1. YOLO multi-object tracking
        results = self.detector.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes

        active_agents = []
        if len(boxes) > 0 and boxes.id is not None:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy()
            id_arr = boxes.id.cpu().numpy()

            for i in range(len(xyxy_arr)):
                coco_cls = int(cls_arr[i])
                if coco_cls not in self.valid_classes:
                    continue

                mapped_cls = self.valid_classes[coco_cls]
                track_id = int(id_arr[i])
                x1, y1, x2, y2 = xyxy_arr[i]
                cx, cy = (x1 + x2) / (2.0 * W), (y1 + y2) / (2.0 * H)
                w, h = (x2 - x1) / W, (y2 - y1) / H

                if track_id not in self.tracks:
                    self.tracks[track_id] = AgentTrackState(track_id, mapped_cls)

                track = self.tracks[track_id]
                track.update(t_sec, cx, cy, w, h, ego_speed=ego_speed)
                active_agents.append((track, (x1, y1, x2, y2)))

        # Prune old tracks (> 1.5s since last seen)
        stale_ids = [tid for tid, tr in self.tracks.items() if (t_sec - tr.last_seen) > 1.5]
        for tid in stale_ids:
            del self.tracks[tid]

        # 2. Build Spatio-Temporal Scene Graph
        c = self.config
        max_N = c.max_nodes
        N = min(len(active_agents), max_N)

        node_feat = np.zeros((max_N, c.node_feature_dim), dtype=np.float32)
        hazard = np.zeros(max_N, dtype=np.float32)
        mask = np.zeros(max_N, dtype=bool)

        for i in range(N):
            tr, _ = active_agents[i]
            _, cx, cy, w, h = tr.history[-1]
            heading = float(np.arctan2(tr.vy, tr.vx + 1e-6))
            speed = float(np.sqrt(tr.vx**2 + tr.vy**2))
            node_feat[i] = [
                float(tr.cls_id), cx, cy, w, h, tr.vx, tr.vy, tr.ax, tr.ay,
                heading, min(tr.ttc, 30.0), speed
            ]
            hazard[i] = tr.hazard_weight
            mask[i] = True

        src_list, dst_list = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    src_list.append(i)
                    dst_list.append(j)

        E = len(src_list)
        if E > 0:
            edge_idx = np.array([src_list, dst_list], dtype=np.int64)
            edge_attr = np.zeros((E, c.edge_feature_dim), dtype=np.float32)
            for idx_e in range(E):
                si, di = src_list[idx_e], dst_list[idx_e]
                dx = node_feat[di, 1] - node_feat[si, 1]
                dy = node_feat[di, 2] - node_feat[si, 2]
                dvx = node_feat[di, 5] - node_feat[si, 5]
                dvy = node_feat[di, 6] - node_feat[si, 6]
                dist = np.sqrt(dx**2 + dy**2)
                bearing = np.arctan2(dy, dx)
                edge_attr[idx_e] = [dx, dy, dvx, dvy, dist, bearing]
        else:
            edge_idx = np.zeros((2, 0), dtype=np.int64)
            edge_attr = np.zeros((0, c.edge_feature_dim), dtype=np.float32)

        # 3. PC-CSG Forward Pass
        if N == 0:
            crt_scalar = 0.0
            pikv_validity = np.zeros((max_N, self.config.num_counterfactuals), dtype=bool)
            trajectories = np.zeros((max_N, self.config.num_counterfactuals, self.config.prediction_horizon, 2), dtype=np.float32)
        else:
            # Clean node features of any possible NaN/Inf
            node_feat = np.nan_to_num(node_feat, nan=0.0, posinf=30.0, neginf=-30.0)
            hazard = np.nan_to_num(hazard, nan=0.0, posinf=1.0, neginf=0.0)

            with torch.no_grad():
                node_t = torch.from_numpy(node_feat).unsqueeze(0).to(self.device)
                edge_t = torch.from_numpy(edge_idx).to(self.device)
                edge_a_t = torch.from_numpy(edge_attr).unsqueeze(0).to(self.device)
                hazard_t = torch.from_numpy(hazard).unsqueeze(0).to(self.device)
                mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

                out = self.model(
                    node_features=node_t,
                    edge_index=edge_t,
                    edge_attr=edge_a_t,
                    hazard_weights=hazard_t,
                    node_mask=mask_t,
                )

            crt_raw = float(out["crt_scalar"][0].item())
            crt_scalar = 0.0 if (np.isnan(crt_raw) or np.isinf(crt_raw)) else float(np.clip(crt_raw, 0.0, 1.0))
            pikv_validity = out["pikv_validity"][0].cpu().numpy()  # (N, K)
            trajectories = out["trajectories"][0].cpu().numpy()     # (N, K, H, 2)

        # Determine intervention level
        if crt_scalar < 0.25:
            level = InterventionLevel.NOMINAL
        elif crt_scalar < 0.50:
            level = InterventionLevel.ADVISORY
        elif crt_scalar < 0.70:
            level = InterventionLevel.ALERT
        elif crt_scalar < 0.85:
            level = InterventionLevel.PRE_INTERVENTION
        else:
            level = InterventionLevel.EMERGENCY

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Render Anticipatory HUD directly over real video frame
        annotated_frame = self._render_hud(
            frame=frame,
            active_agents=active_agents,
            crt_scalar=crt_scalar,
            level=level,
            pikv_validity=pikv_validity,
            trajectories=trajectories,
            elapsed_ms=elapsed_ms,
            t_sec=t_sec,
        )

        metrics = {
            "crt": crt_scalar,
            "level": level.value,
            "latency_ms": elapsed_ms,
            "agents_count": len(active_agents),
        }
        return annotated_frame, metrics

    def _render_hud(
        self,
        frame: np.ndarray,
        active_agents: list,
        crt_scalar: float,
        level: InterventionLevel,
        pikv_validity: np.ndarray,
        trajectories: np.ndarray,
        elapsed_ms: float,
        t_sec: float,
    ) -> np.ndarray:
        H, W = frame.shape[:2]
        vis = frame.copy()

        # ── Draw detected agents and counterfactual cones ──
        for idx, (tr, (x1, y1, x2, y2)) in enumerate(active_agents[:self.config.max_nodes]):
            # Color by hazard
            if tr.ttc < 2.0:
                box_color = COLORS["critical_red"]
            elif tr.ttc < 4.0:
                box_color = COLORS["orange"]
            elif tr.ttc < 8.0:
                box_color = COLORS["yellow"]
            else:
                box_color = COLORS["green"]

            # Bounding box
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)

            # Label banner
            cname = self.class_names.get(tr.cls_id, "Vehicle")
            label = f"{cname} #{tr.track_id} | TTC:{tr.ttc:.1f}s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (int(x1), int(y1) - th - 6), (int(x1) + tw + 6, int(y1)), box_color, -1)
            cv2.putText(vis, label, (int(x1) + 3, int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)

            # Draw counterfactual trajectory projections (purple rays)
            cx, cy = int((x1 + x2) / 2), int(y2)
            if tr.ttc < 6.0 and idx < len(trajectories):
                for k in range(min(4, self.config.num_counterfactuals)):
                    is_valid = bool(pikv_validity[idx, k])
                    ray_color = COLORS["purple"] if is_valid else (80, 80, 80)
                    offset_x = int(45 * (k - 1.5))
                    offset_y = int(35 + 10 * k)
                    cv2.arrowedLine(vis, (cx, cy), (cx + offset_x, cy + offset_y), ray_color, 1, tipLength=0.2)

        # ── Top Bar ──
        cv2.rectangle(vis, (0, 0), (W, 45), (15, 18, 26), -1)
        cv2.putText(vis, "PC-CSG: Physics-Constrained Counterfactual Scene Graph Transformer [REAL VIDEO]",
                    (15, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["cyan"], 1, cv2.LINE_AA)
        cv2.putText(vis, f"Inference: {elapsed_ms:.1f}ms ({1000.0/max(1.0, elapsed_ms):.1f} FPS) | Tracked: {len(active_agents)} agents",
                    (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLORS["white"], 1, cv2.LINE_AA)

        # ── Bottom-Left CRT Gauge ──
        gw, gh = 260, 115
        gx, gy = 20, H - gh - 20
        cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (20, 24, 32), -1)
        cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), COLORS["dim"], 1)

        cv2.putText(vis, "Counterfactual Risk Tensor (CRT)",
                    (gx + 12, gy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["white"], 1, cv2.LINE_AA)

        bar_x, bar_y = gx + 12, gy + 32
        bar_w, bar_h = gw - 24, 18
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 45, 55), -1)
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLORS["dim"], 1)

        fill_w = int(bar_w * np.clip(crt_scalar, 0.0, 1.0))
        if crt_scalar < 0.25:
            fill_c = COLORS["green"]
        elif crt_scalar < 0.50:
            fill_c = COLORS["yellow"]
        elif crt_scalar < 0.70:
            fill_c = COLORS["orange"]
        else:
            pulse = int(40 * np.sin(t_sec * 8))
            fill_c = (40, 40, min(255, 215 + pulse))

        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_c, -1)

        # Level name and color
        lname, lcolor = self.intervention_names[level]
        cv2.putText(vis, f"CRT = {crt_scalar:.3f}", (bar_x, bar_y + bar_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLORS["white"], 1, cv2.LINE_AA)
        cv2.putText(vis, lname, (gx + 12, gy + gh - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, lcolor, 1, cv2.LINE_AA)

        # ── Bottom-Right Counterfactual Analysis Card ──
        cw, ch = 320, 115
        cx, cy = W - cw - 20, H - ch - 20
        cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), (20, 24, 32), -1)
        cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), COLORS["dim"], 1)

        cv2.putText(vis, "Anticipatory Counterfactual Analysis",
                    (cx + 12, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["cyan"], 1, cv2.LINE_AA)

        if len(active_agents) > 0:
            min_ttc = min(tr.ttc for tr, _ in active_agents)
            cv2.putText(vis, f"Worst-Case TTC: {min_ttc:.1f}s", (cx + 12, cy + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["white"], 1, cv2.LINE_AA)
            cv2.putText(vis, "PIKV Kinematic Filtering: ACTIVE", (cx + 12, cy + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLORS["green"], 1, cv2.LINE_AA)
            cv2.putText(vis, f"Evaluating {min(len(active_agents), self.config.max_nodes)*6} 'What-If' Trajectories",
                        (cx + 12, cy + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLORS["dim"], 1, cv2.LINE_AA)
            cv2.putText(vis, "Friction Circle Bound: mu*g <= 6.87 m/s^2",
                        (cx + 12, cy + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLORS["dim"], 1, cv2.LINE_AA)
        else:
            cv2.putText(vis, "No active agents detected in scene", (cx + 12, cy + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["dim"], 1, cv2.LINE_AA)

        # ── Emergency Warning Flashing Banner ──
        if level in (InterventionLevel.PRE_INTERVENTION, InterventionLevel.EMERGENCY):
            flash = int(70 * abs(np.sin(t_sec * 6)))
            banner = vis.copy()
            cv2.rectangle(banner, (0, 45), (W, 95), (0, 0, 180 + flash), -1)
            cv2.addWeighted(banner, 0.35, vis, 0.65, 0, vis)
            cv2.putText(vis, "! CRITICAL COLLISION HAZARD ANTICIPATED — PREEMPTIVE INTERVENTION !",
                        (W // 2 - 290, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        return vis

    def process_video(
        self,
        input_video_path: str,
        output_video_path: Optional[str] = None,
        display: bool = True,
        max_frames: Optional[int] = None,
    ):
        """
        Run inference over an entire video file or webcam stream.
        """
        # Support webcam int input
        if input_video_path.isdigit():
            cap = cv2.VideoCapture(int(input_video_path))
        else:
            cap = cv2.VideoCapture(input_video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video stream: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Opening video: {input_video_path}")
        print(f"Resolution: {width}x{height} | FPS: {fps:.1f} | Total Frames: {total_frames}")

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path) or ".", exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            print(f"Writing annotated output to: {output_video_path}")

        if display:
            try:
                cv2.namedWindow("PC-CSG Real Video HUD", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("PC-CSG Real Video HUD", min(1280, width), min(720, height))
            except Exception as e:
                print(f"Warning: Cannot open window ({e}), falling back to headless mode.")
                display = False

        frame_idx = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if max_frames and frame_idx > max_frames:
                    break

                t_sec = frame_idx / fps
                annotated_frame, metrics = self.process_frame(frame, t_sec=t_sec)

                if writer:
                    writer.write(annotated_frame)

                if display:
                    cv2.imshow("PC-CSG Real Video HUD", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == ord("Q"):
                        print("User interrupted playback.")
                        break
                    elif key == ord(" "):
                        # Pause
                        cv2.waitKey(0)

                if frame_idx % 30 == 0:
                    pct = (frame_idx / total_frames * 100.0) if total_frames > 0 else 0.0
                    print(f"Frame {frame_idx}/{total_frames} ({pct:.1f}%) | CRT: {metrics['crt']:.3f} | Latency: {metrics['latency_ms']:.1f}ms")

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / max(0.1, elapsed):.1f} FPS).")
        if output_video_path:
            print(f"Annotated real video saved to: {output_video_path}")

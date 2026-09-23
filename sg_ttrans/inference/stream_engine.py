"""
Real-time streaming inference engine and interactive HUD dashboard for SG-TTrans.
"""

import cv2
import numpy as np
import torch
from sg_ttrans.config import SGTransConfig
from sg_ttrans.geometry.facemesh import FaceMeshExtractor
from sg_ttrans.geometry.metrics import compute_ear, compute_mar, compute_perclos
from sg_ttrans.geometry.head_pose import HeadPoseEstimator, build_kinematic_token
from sg_ttrans.models.sg_ttrans_net import SGTransNet
from sg_ttrans.data.sequence_buffer import SequenceBuffer
from sg_ttrans.risk_engine.persistence import FatiguePersistenceTracker
from sg_ttrans.risk_engine.rsi import compute_rsi
from sg_ttrans.risk_engine.adas_controller import ADASController, ADASLevel

DRIVER_STATE_NAMES = ["Alert", "Drowsy", "Microsleep", "Yawn", "Distracted"]

class StreamEngine:
    """
    Real-time streaming engine coupling dual-stream feature extraction,
    TDDA sequence inference, RSI risk calculation, and ADAS control.
    """
    def __init__(self, config: SGTransConfig | None = None, device: str = "cpu"):
        self.cfg = config if config is not None else SGTransConfig()
        self.device = torch.device(device)

        # Components
        self.facemesh = FaceMeshExtractor()
        self.head_pose = HeadPoseEstimator()
        self.model = SGTransNet(self.cfg).to(self.device)
        self.model.eval()

        self.buffer = SequenceBuffer(max_len=self.cfg.sequence_length)
        self.persistence_tracker = FatiguePersistenceTracker(
            sequence_length=self.cfg.sequence_length, dt=1.0 / self.cfg.fps
        )
        self.adas_controller = ADASController(self.cfg)

        # State memory
        self.ear_history: list[float] = []
        self.prev_kinematics: np.ndarray | None = None
        self.last_result: dict = {
            "rsi": 0.0,
            "driver_state": "Initializing",
            "state_idx": 0,
            "confidence": 1.0,
            "fatigue_prob": 0.0,
            "tau_persist": 0.0,
            "adas_level": 0,
            "adas_action": "Nominal Operation",
            "ear": 0.30,
            "mar": 0.20,
            "perclos": 0.0,
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        }

    def process_frame(
        self,
        bgr_frame: np.ndarray,
        velocity: float = 80.0,
        ttc: float = 8.0
    ) -> tuple[dict, np.ndarray]:
        """
        Processes a single frame, updates temporal buffer, and returns metrics + HUD visualization.
        """
        h, w, _ = bgr_frame.shape

        # 1. Landmark extraction
        landmarks_68, has_face = self.facemesh.extract(bgr_frame)

        if has_face and landmarks_68 is not None:
            # Left eye: indices 36-41, Right eye: 42-47
            left_eye = landmarks_68[36:42]
            right_eye = landmarks_68[42:48]
            ear_l = compute_ear(left_eye)
            ear_r = compute_ear(right_eye)
            ear = (ear_l + ear_r) / 2.0

            # Inner mouth: indices 60-67
            inner_mouth = landmarks_68[60:68]
            mar = compute_mar(inner_mouth)

            # Head pose keypoints
            key_pts = np.array([
                landmarks_68[30][:2],  # Nose tip
                landmarks_68[8][:2],   # Chin
                landmarks_68[36][:2],  # Left eye corner
                landmarks_68[45][:2],  # Right eye corner
                landmarks_68[48][:2],  # Left mouth corner
                landmarks_68[54][:2]   # Right mouth corner
            ], dtype=np.float64)
            yaw, pitch, roll = self.head_pose.estimate(key_pts, img_w=w, img_h=h)
        else:
            # Fallback to smooth default values when face is occluded
            ear_l, ear_r = 0.25, 0.25
            ear = 0.25
            mar = 0.20
            yaw, pitch, roll = 0.0, 0.0, 0.0

        # Update EAR history for PERCLOS
        self.ear_history.append(ear)
        if len(self.ear_history) > self.cfg.perclos_window:
            self.ear_history.pop(0)
        perclos = compute_perclos(self.ear_history, threshold=self.cfg.delta_ear)

        # Assemble 16-D kinematic token
        euler = np.array([yaw, pitch, roll], dtype=np.float32)
        kin_token = build_kinematic_token(
            ear=ear, mar=mar, perclos=perclos, euler=euler,
            prev_kinematics=self.prev_kinematics,
            ear_l=ear_l, ear_r=ear_r, dt=1.0 / self.cfg.fps
        )
        self.prev_kinematics = kin_token

        # Normalize & resize frame to 224x224 RGB tensor
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        frame_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        # Push to FIFO sliding window buffer
        self.buffer.push(frame_tensor, torch.from_numpy(kin_token))

        # Perform inference once buffer has accumulated sequence_length frames
        if self.buffer.is_ready():
            seq_frames, seq_kin = self.buffer.get_sequence()
            seq_frames = seq_frames.unsqueeze(0).to(self.device)  # [1, T, 3, 224, 224]
            seq_kin = seq_kin.unsqueeze(0).to(self.device)        # [1, T, 16]

            with torch.no_grad():
                outputs = self.model(seq_frames, seq_kin)
                probs = outputs["probs"][0].cpu().numpy()
                state_idx = int(np.argmax(probs))
                driver_state = DRIVER_STATE_NAMES[state_idx]
                confidence = float(probs[state_idx])
                fatigue_prob = float(outputs["fatigue_prob"][0].cpu().item())

            # Update persistence tracker
            is_fatigued = (state_idx in [1, 2]) or (fatigue_prob > 0.5)
            tau_persist = self.persistence_tracker.update(is_fatigued)

            # Compute Dynamic RSI and ADAS Intervention
            rsi = compute_rsi(
                fatigue_prob=fatigue_prob,
                tau_persist=tau_persist,
                velocity=velocity,
                ttc=ttc,
                pitch=pitch,
                yaw=yaw,
                config=self.cfg
            )
            adas_action = self.adas_controller.evaluate(rsi)

            self.last_result = {
                "rsi": rsi,
                "driver_state": driver_state,
                "state_idx": state_idx,
                "confidence": confidence,
                "fatigue_prob": fatigue_prob,
                "tau_persist": tau_persist,
                "adas_level": int(adas_action.level),
                "adas_action": adas_action.name,
                "ear": ear,
                "mar": mar,
                "perclos": perclos,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            }
        else:
            self.last_result["ear"] = ear
            self.last_result["mar"] = mar
            self.last_result["perclos"] = perclos
            self.last_result["yaw"] = yaw
            self.last_result["pitch"] = pitch
            self.last_result["roll"] = roll

        # Draw real-time HUD on copy of current frame
        hud_frame = self.render_hud(bgr_frame, self.last_result, landmarks_68, velocity, ttc)
        return self.last_result, hud_frame

    def render_hud(
        self,
        frame: np.ndarray,
        res: dict,
        landmarks: np.ndarray | None,
        velocity: float,
        ttc: float
    ) -> np.ndarray:
        """
        Renders HUD dashboard overlay with facial markers, gauges, and alerts.
        """
        vis = frame.copy()
        h, w, _ = vis.shape

        # 1. Draw facial landmarks if available
        if landmarks is not None:
            for pt in landmarks:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 1, (0, 255, 255), -1)

        # 2. Semi-transparent top dashboard banner
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
        # Bottom ADAS banner
        cv2.rectangle(overlay, (0, h - 60), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)

        # 3. Status text
        state_str = f"DRIVER STATE: {res['driver_state'].upper()} ({res['confidence']*100:.1f}%)"
        state_color = (0, 255, 0) if res["state_idx"] == 0 else (0, 140, 255)
        if res["state_idx"] in [1, 2]:
            state_color = (0, 0, 255)

        cv2.putText(vis, state_str, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, state_color, 2)

        # Metrics text (EAR, MAR, PERCLOS, tau_persist)
        metrics_str = f"EAR: {res['ear']:.2f} | MAR: {res['mar']:.2f} | PERCLOS: {res['perclos']*100:.0f}% | Tau: {res['tau_persist']:.2f}s"
        cv2.putText(vis, metrics_str, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Telematics text (Speed, TTC)
        telem_str = f"SPEED: {velocity:.0f} km/h | TTC: {ttc:.1f} s | Pose: Y:{res['yaw']:.0f} P:{res['pitch']:.0f}"
        cv2.putText(vis, telem_str, (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 255), 1)

        # 4. RSI Dynamic Gauge Bar
        rsi = res["rsi"]
        bar_x, bar_y, bar_w, bar_h = w - 240, 30, 210, 25
        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 2)

        fill_w = int(rsi * bar_w)
        # Color gradient based on RSI value
        if rsi < self.cfg.delta1:
            bar_color = (0, 220, 0)      # Green
        elif rsi < self.cfg.delta2:
            bar_color = (0, 220, 255)    # Yellow
        elif rsi < self.cfg.delta3:
            bar_color = (0, 140, 255)    # Orange
        else:
            bar_color = (0, 0, 255)      # Red

        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.putText(vis, f"RSI: {rsi:.3f}", (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 5. Bottom ADAS Intervention Bar
        level = res["adas_level"]
        adas_color = (0, 255, 0) if level == 0 else (0, 200, 255) if level == 1 else (0, 140, 255) if level == 2 else (0, 0, 255)
        adas_str = f"ADAS ACTION: {res['adas_action']}"
        cv2.putText(vis, adas_str, (20, h - 22), cv2.FONT_HERSHEY_DUPLEX, 0.7, adas_color, 2)

        return vis

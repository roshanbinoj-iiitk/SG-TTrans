#!/usr/bin/env python3
"""
SG-TTrans: Real-Time Live Video / Webcam / Synthetic Demonstration Runner.
Usage:
    python demo_stream.py --source synthetic
    python demo_stream.py --source 0
    python demo_stream.py --source path/to/video.mp4
"""

import argparse
import time
import cv2
import numpy as np
import torch
from sg_ttrans.config import SGTransConfig
from sg_ttrans.inference.stream_engine import StreamEngine
from sg_ttrans.alerts.audio_alert import AudioAlertManager

def main():
    parser = argparse.ArgumentParser(description="SG-TTrans Real-Time Driver Drowsiness Demo")
    parser.add_argument("--source", type=str, default="synthetic", help="'0' for webcam, file path, or 'synthetic'")
    parser.add_argument("--velocity", type=float, default=90.0, help="Initial vehicle velocity (km/h)")
    parser.add_argument("--ttc", type=float, default=5.0, help="Initial forward radar Time-to-Collision (s)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target processing FPS")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 for continuous until closed)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--no-audio", action="store_true", help="Disable acoustic alert audio playback")
    parser.add_argument("--volume", type=float, default=0.6, help="Alert audio volume (0.0 to 1.0, default: 0.6)")
    args = parser.parse_args()

    print(f"Initializing SG-TTrans on device: {args.device}")
    cfg = SGTransConfig(fps=args.fps)
    engine = StreamEngine(config=cfg, device=args.device, checkpoint_path=args.checkpoint if args.checkpoint else None)
    audio_mgr = AudioAlertManager(enabled=not args.no_audio, volume=args.volume)

    is_synthetic = (args.source.lower() == "synthetic")
    cap = None
    if not is_synthetic:
        src = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Failed to open video source '{args.source}', switching to synthetic.")
            is_synthetic = True

    print("\n" + "="*60)
    print(" SG-TTrans Real-Time ADAS Monitoring Engine Active")
    print(" Keyboard Controls (when GUI active):")
    print("   [1] Scenario A: Normal Driving (v=80, TTC=8.0, Alert)")
    print("   [2] Scenario B: Moderate Yawning (v=90, TTC=3.5, Yawn)")
    print("   [3] Scenario C: Critical Micro-sleep (v=110, TTC=1.4 -> 0.9, AEB)")
    print("   [M] Toggle Audio Alert Mute / Unmute")
    print("   [Q/ESC] Quit Demo (or close display window)")
    print("="*60 + "\n")

    velocity = args.velocity
    ttc = args.ttc
    frame_idx = 0
    consecutive_fails = 0
    start_time = time.time()

    # Pre-generate synthetic sequence if synthetic mode
    if is_synthetic:
        rng = np.random.default_rng(42)

    try:
        while True:
            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            if is_synthetic:
                # Generate synthetic test frame (640x480)
                frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                # Draw synthetic face circle
                cv2.circle(frame, (320, 240), 120, (180, 190, 200), -1)
                time.sleep(1.0 / args.fps)
            else:
                ret, frame = cap.read()
                if not ret:
                    consecutive_fails += 1
                    if consecutive_fails > 10:
                        print("End of video stream reached.")
                        break
                    time.sleep(0.01)
                    continue
                consecutive_fails = 0

            # Process frame through SG-TTrans
            res, hud_frame = engine.process_frame(frame, velocity=velocity, ttc=ttc)

            # Update audio alert playback (Level 2 & Level 3 trigger real acoustic alarms)
            audio_mgr.update(res["adas_level"])

            # Print console telemetry every 15 frames
            if frame_idx % 15 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                print(f"[Frame {frame_idx:04d} | {fps_actual:.1f} FPS] State: {res['driver_state']:<11} | "
                      f"EAR: {res['ear']:.2f} | MAR: {res['mar']:.2f} | RSI: {res['rsi']:.3f} | ADAS: {res['adas_action']}")

            if not args.headless:
                # Render Audio Status badge in bottom ADAS banner
                h_hud, w_hud = hud_frame.shape[:2]
                if args.no_audio:
                    audio_label = "[AUDIO: OFF]"
                    audio_col = (130, 130, 130)
                elif audio_mgr.is_muted:
                    audio_label = "[AUDIO: MUTED]"
                    audio_col = (0, 140, 255)
                else:
                    audio_label = "[AUDIO: ACTIVE]"
                    audio_col = (0, 255, 0)
                cv2.putText(hud_frame, audio_label, (w_hud - 185, h_hud - 22), cv2.FONT_HERSHEY_DUPLEX, 0.55, audio_col, 1)

                cv2.imshow("SG-TTrans ADAS In-Cabin Monitor", hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                try:
                    if cv2.getWindowProperty("SG-TTrans ADAS In-Cabin Monitor", cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    pass

                if key == ord('1'):  # Scenario A
                    velocity, ttc = 80.0, 8.0
                    print(">> Triggered Scenario A: Normal Blink (Level 0)")
                elif key == ord('2'):  # Scenario B
                    velocity, ttc = 90.0, 3.5
                    print(">> Triggered Scenario B: Moderate Fatigue (Level 1)")
                elif key == ord('3'):  # Scenario C
                    velocity, ttc = 110.0, 1.4
                    print(">> Triggered Scenario C: Critical Micro-sleep (Level 3 AEB)")
                elif key == ord('m') or key == ord('M'):
                    is_muted = audio_mgr.toggle_mute()
                    status = "MUTED" if is_muted else "ACTIVE"
                    print(f">> Audio alert {status}")

    finally:
        audio_mgr.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession terminated successfully. Processed {frame_idx} frames.")

if __name__ == "__main__":
    main()

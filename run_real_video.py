"""
PC-CSG Real Video Inference Execution Script.

Runs the Physics-Constrained Counterfactual Scene Graph Transformer
directly on real-world driving dashcam videos or live webcams.

Usage:
    # Run interactive HUD on default real road dashcam video:
    /home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py

    # Run on a specific video file:
    /home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py --video "path/to/video.mp4"

    # Export annotated HUD video to file:
    /home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py --save-video "demo_outputs/real_road_annotated.mp4"

    # Run on live webcam:
    /home/roshanbinoj/Documents/BTP/venv/bin/python run_real_video.py --webcam 0
"""

import argparse
import glob
import os
import sys

from pc_csg.inference import RealVideoInferenceEngine


DEFAULT_REAL_VIDEOS = [
    "/home/roshanbinoj/Downloads/Driver Drowsiness Dataset (DDD)/road_dashcam_real.mp4",
    "/home/roshanbinoj/Downloads/Driver Drowsiness Dataset (DDD)/car_detection.mp4",
    "/home/roshanbinoj/Downloads/Driver Drowsiness Dataset (DDD)/road_traffic_video.webm",
    "/home/roshanbinoj/Downloads/Driver Drowsiness Dataset (DDD)/road_bridge_traffic.webm",
    "/home/roshanbinoj/Downloads/Driver Drowsiness Dataset (DDD)/road_dashcam_daylight.ogv",
    "/home/roshanbinoj/Downloads/Driver Drowsiness Dataset (DDD)/road_urban_traffic.ogv",
]


def find_default_video() -> str:
    """Find the best available real road video on the system."""
    for vp in DEFAULT_REAL_VIDEOS:
        if os.path.exists(vp):
            return vp
    # Search downloads
    vids = glob.glob("/home/roshanbinoj/Downloads/**/*.mp4", recursive=True)
    if vids:
        return vids[0]
    return "0"  # fallback to webcam


def main():
    parser = argparse.ArgumentParser(description="PC-CSG Real Video Anomaly Anticipation Inference")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (mp4, webm, ogv)")
    parser.add_argument("--webcam", type=int, default=None, help="Webcam device index (e.g., 0)")
    parser.add_argument("--save-video", type=str, default=None, help="Path to save annotated output video (e.g. output.mp4)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/pc_csg_real_best.pt", help="Path to trained PC-CSG checkpoint")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI display window")
    parser.add_argument("--list-videos", action="store_true", help="List available sample road videos on this machine")
    args = parser.parse_args()

    if args.list_videos:
        print("Available real driving videos detected on this system:")
        for vp in DEFAULT_REAL_VIDEOS:
            exists = "✓ EXISTS" if os.path.exists(vp) else "✗ NOT FOUND"
            print(f"  {exists}: {vp}")
        return

    # Determine input stream
    if args.webcam is not None:
        input_source = str(args.webcam)
        print(f"Using live webcam device #{args.webcam}")
    elif args.video:
        input_source = args.video
    else:
        input_source = find_default_video()
        print(f"Auto-selected real driving video: {input_source}")

    if not input_source.isdigit() and not os.path.exists(input_source):
        print(f"Error: Input video does not exist: {input_source}")
        sys.exit(1)

    print("=" * 68)
    print(" PC-CSG Real-World Video Anomaly Anticipation Engine")
    print("=" * 68)
    print(f" Input:      {input_source}")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Output:     {args.save_video or 'Display Only'}")
    print(f" Display:    {'Disabled (Headless)' if args.headless else 'Interactive GUI'}")
    print("=" * 68)

    # Initialize Engine
    engine = RealVideoInferenceEngine(checkpoint_path=args.checkpoint)

    # Execute
    engine.process_video(
        input_video_path=input_source,
        output_video_path=args.save_video,
        display=not args.headless,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

"""
PC-CSG Command Line Interface (CLI).

Unified entry point for:
- Interactive HUD demonstration (demo)
- Real-world dashcam video inference (video)
- Hardware latency and FPS benchmark (bench)
- Interactive Gradio browser dashboard (web)
"""

import argparse
import sys


def main_demo():
    """Launch interactive counterfactual HUD simulation."""
    from pc_csg.inference.demo_engine import main as demo_main
    demo_main()


def main_video():
    """Run real video / dashcam inference pipeline."""
    from pc_csg.inference.real_video_engine import main as video_main
    video_main()


def main_bench():
    """Run hardware latency and throughput benchmark."""
    from pc_csg.benchmark import main as bench_main
    bench_main()


def main_web():
    """Launch Gradio web dashboard."""
    try:
        from pc_csg.web.app import main as web_main
        web_main()
    except ImportError as e:
        print(f"Error launching web dashboard: {e}")
        print("Please install web dependencies: pip install -e .[web]")
        sys.exit(1)


def main():
    """Primary PC-CSG CLI router."""
    parser = argparse.ArgumentParser(
        prog="pc-csg",
        description="PC-CSG: Physics-Constrained Counterfactual Scene Graph Transformer for Autonomous Anomaly Anticipation"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Demo subcommand
    demo_parser = subparsers.add_parser("demo", help="Launch interactive counterfactual HUD simulation")
    demo_parser.add_argument("--headless", action="store_true", help="Run without opening GUI window")
    demo_parser.add_argument("--save-images", action="store_true", help="Export static HUD screenshots")
    demo_parser.add_argument("--save-video", action="store_true", help="Render full MP4 demonstration video")

    # Video subcommand
    video_parser = subparsers.add_parser("video", help="Run real-world dashcam video inference")
    video_parser.add_argument("video_pos", nargs="?", default=None, help="Path to video file (optional positional)")
    video_parser.add_argument("--video", type=str, default=None, help="Path to video file")
    video_parser.add_argument("--webcam", type=int, default=None, help="Webcam device index (0, 1)")
    video_parser.add_argument("--save-video", type=str, default=None, help="Export annotated video path")
    video_parser.add_argument("--checkpoint", type=str, default="checkpoints/pc_csg_real_best.pt", help="Path to PC-CSG checkpoint")
    video_parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    video_parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("bench", help="Run latency and FPS hardware benchmarks")
    bench_parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Compute device")

    # Web dashboard subcommand
    web_parser = subparsers.add_parser("web", help="Launch interactive browser dashboard (Gradio)")
    web_parser.add_argument("--port", type=int, default=7860, help="Web server port")
    web_parser.add_argument("--share", action="store_true", help="Create public shareable link")

    args, unknown = parser.parse_known_args()

    if args.command == "demo":
        from pc_csg.inference.demo_engine import run_demo
        run_demo(
            save_images=args.save_images,
            save_video=args.save_video,
            headless=args.headless,
        )
    elif args.command == "video":
        from pc_csg.inference.real_video_engine import run_video_pipeline
        target_video = args.video or args.video_pos
        run_video_pipeline(
            video_path=target_video,
            webcam=args.webcam,
            save_video=args.save_video,
            checkpoint=args.checkpoint,
            max_frames=args.max_frames,
            headless=args.headless,
        )
    elif args.command == "bench":
        from pc_csg.benchmark import run_benchmark
        run_benchmark(device_str=args.device)
    elif args.command == "web":
        main_web()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

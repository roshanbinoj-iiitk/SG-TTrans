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
    import demo_pc_csg
    demo_pc_csg.main()


def main_video():
    """Run real video / dashcam inference pipeline."""
    import run_real_video
    run_real_video.main()


def main_bench():
    """Run hardware latency and throughput benchmark."""
    import benchmark_pccsg
    benchmark_pccsg.main()


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
    video_parser.add_argument("--video", type=str, default=None, help="Path to video file")
    video_parser.add_argument("--webcam", type=int, default=None, help="Webcam device index (0, 1)")
    video_parser.add_argument("--save-video", type=str, default=None, help="Export annotated video path")
    video_parser.add_argument("--headless", action="store_true", help="Run in headless mode")

    # Benchmark subcommand
    subparsers.add_parser("bench", help="Run latency and FPS hardware benchmarks")

    # Web dashboard subcommand
    web_parser = subparsers.add_parser("web", help="Launch interactive browser dashboard (Gradio)")
    web_parser.add_argument("--port", type=int, default=7860, help="Web server port")
    web_parser.add_argument("--share", action="store_true", help="Create public shareable link")

    args, unknown = parser.parse_known_args()

    if args.command == "demo":
        sys.argv = [sys.argv[0]] + unknown
        if args.headless:
            sys.argv.append("--headless")
        if args.save_images:
            sys.argv.append("--save-images")
        if args.save_video:
            sys.argv.append("--save-video")
        main_demo()
    elif args.command == "video":
        sys.argv = [sys.argv[0]] + unknown
        if args.video:
            sys.argv.extend(["--video", args.video])
        if args.webcam is not None:
            sys.argv.extend(["--webcam", str(args.webcam)])
        if args.save_video:
            sys.argv.extend(["--save-video", args.save_video])
        if args.headless:
            sys.argv.append("--headless")
        main_video()
    elif args.command == "bench":
        main_bench()
    elif args.command == "web":
        main_web()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

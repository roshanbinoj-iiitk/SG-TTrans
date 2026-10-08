import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "pc_csg.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "PC-CSG" in result.stdout
    assert "demo" in result.stdout
    assert "video" in result.stdout
    assert "bench" in result.stdout


def test_cli_subcommands_help():
    for subcmd in ["demo", "video", "bench", "web"]:
        result = subprocess.run(
            [sys.executable, "-m", "pc_csg.cli", subcmd, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


def test_cli_imports_without_repo_in_syspath():
    # Simulate running installed pc-csg console entry point without repo root in sys.path
    code = """
import sys
sys.path = [p for p in sys.path if p not in ('', '.')]
from pc_csg.cli import main_video, main_demo, main_bench
from pc_csg.inference import real_video_engine
from pc_csg.inference import demo_engine
from pc_csg import benchmark
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed with stderr: {result.stderr}"

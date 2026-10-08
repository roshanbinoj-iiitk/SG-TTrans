import subprocess
import sys

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "pc_csg.cli", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "PC-CSG" in result.stdout
    assert "demo" in result.stdout
    assert "video" in result.stdout
    assert "bench" in result.stdout

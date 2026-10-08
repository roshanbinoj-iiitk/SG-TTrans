import subprocess
import sys

def test_reproduce_script_runs():
    result = subprocess.run(
        [sys.executable, "scripts/reproduce_paper_results.py", "--quick"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Script failed with stderr: {result.stderr}"
    assert "TABLE 1: LATENCY BREAKDOWN" in result.stdout
    assert "TABLE 2: PIKV HYPOTHESIS PRUNING" in result.stdout
    assert "PROPOSITION 1" in result.stdout
    assert "PROPOSITION 2" in result.stdout

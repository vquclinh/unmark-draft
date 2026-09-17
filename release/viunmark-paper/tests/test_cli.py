import subprocess
import sys


def test_cli_help_and_diagnostic():
    assert subprocess.run([sys.executable, "-m", "viunmark.cli", "--help"], check=False).returncode == 0
    result = subprocess.run(
        [sys.executable, "scripts/run_diagnostic.py", "--analysis", "scale-preflight"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "ViUnMark-Scale" in result.stdout

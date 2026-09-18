import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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


def test_asset_manifest_command_succeeds_without_external_files():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_assets.py",
            "--config",
            "configs/uit_vsfc/paper.json",
            "--manifest-only",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert '"required_model_asset_count": 22' in result.stdout


def test_high_level_runner_advertises_only_implemented_modes():
    help_result = subprocess.run(
        [sys.executable, "scripts/reproduce_uit_vsfc.py", "--help"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert help_result.returncode == 0
    assert "retrain-readouts" not in help_result.stdout
    assert "prepare" not in help_result.stdout
    assert "predict" not in help_result.stdout
    assert "{verify,diagnostics}" in help_result.stdout


def test_readme_python_commands_parse_or_are_documented_asset_checks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commands = [line.strip() for line in readme.splitlines() if line.strip().startswith("python scripts/")]
    assert commands
    for command in commands:
        script = command.split()[1]
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, command

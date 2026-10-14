"""Regression tests for the reported version / .env bugs."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys

import topiclayers


def test_version_matches_installed_metadata():
    assert topiclayers.__version__ == importlib.metadata.version("topiclayers")


def _run_in(directory, extra_env=None):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", "import os; import topiclayers; print(os.environ.get('TL_TEST_KEY', ''))"],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
    )


def test_dotenv_is_loaded(tmp_path):
    (tmp_path / ".env").write_text("TL_TEST_KEY=from-dotenv\n")
    result = _run_in(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "from-dotenv"


def test_real_env_not_overridden_by_dotenv(tmp_path):
    (tmp_path / ".env").write_text("TL_TEST_KEY=from-dotenv\n")
    result = _run_in(tmp_path, extra_env={"TL_TEST_KEY": "from-env"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "from-env"

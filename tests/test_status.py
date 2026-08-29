from __future__ import annotations

import json
from pathlib import Path

from topiclayers.status import (
    STAGES,
    RunStatus,
    format_seconds,
    historical_eta,
    load_status,
    print_status,
)


def test_start_writes_status_file(tmp_output_dir):
    rs = RunStatus(tmp_output_dir, "test_run")
    rs.start()

    data = load_status(tmp_output_dir / "run_status.json")
    assert data["name"] == "test_run"
    assert data["status"] == "running"
    assert data["stages"] == STAGES


def test_stage_transitions_and_progress(tmp_output_dir):
    rs = RunStatus(tmp_output_dir, "test_run")
    rs.start()

    rs.stage("data", total=100)
    assert load_status(tmp_output_dir / "run_status.json")["current_stage"] == "data"
    assert load_status(tmp_output_dir / "run_status.json")["stage_index"] == 1

    rs.update(50)
    rs.flush()
    data = load_status(tmp_output_dir / "run_status.json")
    assert data["processed"] == 50
    assert data["total"] == 100

    rs.stage("embed", total=100)
    assert load_status(tmp_output_dir / "run_status.json")["current_stage"] == "embed"

    rs.finish("success")
    data = load_status(tmp_output_dir / "run_status.json")
    assert data["status"] == "success"
    assert data["current_stage"] == "embed"


def test_eta_requires_elapsed_time(tmp_output_dir, monkeypatch):
    rs = RunStatus(tmp_output_dir, "test_run")
    rs.start()
    rs.stage("data", total=100)
    rs.update(50)
    data = load_status(tmp_output_dir / "run_status.json")
    assert data["eta_seconds"] is None


def test_stage_timings_recorded(tmp_output_dir):
    rs = RunStatus(tmp_output_dir, "test_run")
    rs.start()
    rs.stage("data")
    rs.stage("embed")
    rs.finish("success")

    timings_path = tmp_output_dir / "stage_timings.json"
    assert timings_path.exists()
    records = json.loads(timings_path.read_text())
    stages_seen = [r["stage"] for r in records]
    assert "data" in stages_seen
    assert "embed" in stages_seen


def test_historical_eta(tmp_output_dir):
    timings_path = tmp_output_dir / "stage_timings.json"
    timings_path.write_text(json.dumps([
        {"stage": "embed", "duration_seconds": 10.0},
        {"stage": "embed", "duration_seconds": 20.0},
        {"stage": "embed", "duration_seconds": 30.0},
        {"stage": "data", "duration_seconds": 5.0},
    ]))
    assert historical_eta(timings_path, "embed") == 20.0
    assert historical_eta(timings_path, "unknown") is None


def test_format_seconds():
    assert format_seconds(None) == "unknown"
    assert format_seconds(45) == "45s"
    assert format_seconds(90) == "1m 30s"
    assert format_seconds(3700) == "1h 01m"


def test_print_status_no_file(tmp_output_dir, capsys):
    code = print_status(tmp_output_dir)
    assert code == 1
    assert "No run_status.json" in capsys.readouterr().out


def test_print_status_running(tmp_output_dir, capsys):
    rs = RunStatus(tmp_output_dir, "test_run")
    rs.start()
    rs.stage("data", total=100)
    rs.update(25)
    rs.flush()

    code = print_status(tmp_output_dir)
    out = capsys.readouterr().out
    assert code == 0
    assert "Stage 2/7: data" in out
    assert "running" in out
    assert "25/100" in out

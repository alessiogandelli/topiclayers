"""Run status: live stage tracking and ETA for long-running pipeline runs."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STAGES = ["load", "data", "embed", "topic_model", "label", "network", "manifest"]
_STATUS_FILE = "run_status.json"
_TIMINGS_FILE = "stage_timings.json"
_MIN_UPDATE_INTERVAL = 1.0
_HEARTBEAT_INTERVAL = 5.0


class RunStatus:
    def __init__(self, output_dir: Path, name: str):
        self.output_dir = Path(output_dir)
        self.name = name
        self.path = self.output_dir / _STATUS_FILE
        self.timings_path = self.output_dir / "cache" / _TIMINGS_FILE
        self.current_stage: Optional[str] = None
        self.stage_index: Optional[int] = None
        self.stage_total: Optional[int] = None
        self.processed: Optional[int] = None
        self.started_at = datetime.now(timezone.utc)
        self.stage_started_at: Optional[datetime] = None
        self._last_write = 0.0
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._start_heartbeat()
        self._write(status="running")

    def _start_heartbeat(self) -> None:
        # keeps stage_elapsed_seconds fresh during opaque stages (e.g. UMAP +
        # HDBSCAN) that never call update()
        def beat():
            while not self._stop_heartbeat.wait(_HEARTBEAT_INTERVAL):
                if self.current_stage is not None:
                    self._write(status="running", force=True)

        self._heartbeat_thread = threading.Thread(target=beat, daemon=True)
        self._heartbeat_thread.start()

    def stage(self, name: str, total: Optional[int] = None) -> None:
        now = datetime.now(timezone.utc)
        if self.current_stage is not None and self.stage_started_at is not None:
            duration = (now - self.stage_started_at).total_seconds()
            self._record_timing(self.current_stage, duration)
        self.current_stage = name
        self.stage_index = STAGES.index(name) if name in STAGES else None
        self.stage_total = total
        self.processed = 0
        self.stage_started_at = now
        self._write(status="running", force=True)

    def update(self, processed: int, total: Optional[int] = None) -> None:
        self.processed = processed
        if total is not None:
            self.stage_total = total
        self._write(status="running", throttle=True)

    def flush(self) -> None:
        self._write(status="running", force=True)

    def finish(self, status: str = "success") -> None:
        self._stop_heartbeat.set()
        if self.current_stage is not None and self.stage_started_at is not None:
            duration = (datetime.now(timezone.utc) - self.stage_started_at).total_seconds()
            self._record_timing(self.current_stage, duration)
        self._write(status=status, force=True)

    def _eta_seconds(self) -> Optional[float]:
        if (
            self.processed is None
            or self.stage_total is None
            or self.stage_started_at is None
            or self.processed <= 0
            or self.stage_total <= self.processed
        ):
            return None
        elapsed = (datetime.now(timezone.utc) - self.stage_started_at).total_seconds()
        if elapsed <= 0:
            return None
        rate = self.processed / elapsed
        return (self.stage_total - self.processed) / rate

    def _write(self, status: str, force: bool = False, throttle: bool = False) -> None:
        now_monotonic = time.monotonic()
        if throttle and not force:
            if now_monotonic - self._last_write < _MIN_UPDATE_INTERVAL:
                return
        self._last_write = now_monotonic

        now = datetime.now(timezone.utc)
        payload = {
            "name": self.name,
            "status": status,
            "stages": STAGES,
            "current_stage": self.current_stage,
            "stage_index": self.stage_index,
            "stage_started_at": self.stage_started_at.isoformat() if self.stage_started_at else None,
            "stage_elapsed_seconds": round((now - self.stage_started_at).total_seconds(), 2) if self.stage_started_at else None,
            "total_elapsed_seconds": round((now - self.started_at).total_seconds(), 2),
            "processed": self.processed,
            "total": self.stage_total,
            "eta_seconds": round(self._eta_seconds(), 2) if self._eta_seconds() is not None else None,
            "updated_at": now.isoformat(),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(f".tmp{threading.get_ident()}")
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)

    def _record_timing(self, stage: str, duration: float) -> None:
        self.timings_path.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        if self.timings_path.exists():
            try:
                with open(self.timings_path) as f:
                    records = json.load(f)
            except Exception:
                records = []
        records.append({
            "stage": stage,
            "duration_seconds": round(duration, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        tmp = self.timings_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp, self.timings_path)


def load_status(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def historical_eta(timings_path: Path, stage: str, last_n: int = 3) -> Optional[float]:
    if not timings_path.exists():
        return None
    try:
        with open(timings_path) as f:
            records = json.load(f)
    except Exception:
        return None
    durations = sorted(r["duration_seconds"] for r in records if r.get("stage") == stage)
    if not durations:
        return None
    return durations[len(durations) // 2]


def format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def print_status(output_dir: Path) -> int:
    path = output_dir / _STATUS_FILE
    if not path.exists():
        print(f"No run_status.json found in {output_dir}.")
        print("Either no run has started there, or the config pointed to a different output_dir.")
        return 1

    data = load_status(path)
    status = data.get("status", "unknown")
    stage = data.get("current_stage")
    idx = data.get("stage_index")
    n_stages = len(data.get("stages", STAGES))

    if idx is not None and stage:
        print(f"Stage {idx + 1}/{n_stages}: {stage}")
    elif stage:
        print(f"Stage: {stage}")
    print(f"Status: {status}")
    print(f"Run: {data.get('name')}")

    if status == "running":
        total = data.get("total")
        processed = data.get("processed")
        if total:
            pct = 100 * processed / total if processed is not None else 0
            print(f"Progress: {processed}/{total} ({pct:.0f}%)")
        stage_elapsed = data.get("stage_elapsed_seconds")
        if stage_elapsed is not None:
            print(f"Stage elapsed: {format_seconds(stage_elapsed)}")
        eta = data.get("eta_seconds")
        if eta is not None:
            print(f"Stage ETA: ~{format_seconds(eta)}")
        else:
            hist = historical_eta(output_dir / "cache" / _TIMINGS_FILE, stage)
            if hist is not None:
                print(f"No live ETA for this stage; past runs took ~{format_seconds(hist)} (median).")
            else:
                print("No live ETA for this stage (no per-item counter, no past runs yet).")
        print(f"Total elapsed: {format_seconds(data.get('total_elapsed_seconds'))}")
    elif status in ("success", "failed"):
        print(f"Total elapsed: {format_seconds(data.get('total_elapsed_seconds'))}")
        print(f"Updated: {data.get('updated_at')}")
    return 0

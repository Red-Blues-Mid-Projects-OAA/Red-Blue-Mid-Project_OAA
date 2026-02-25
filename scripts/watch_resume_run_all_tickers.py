"""
Watchdog for multi-ticker run.

Behavior:
1) Monitor completed tickers from existing logs.
2) If worker process is not running and run is not finished, restart resume runner.
3) Repeat until all tickers (default top300) are completed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SUCCESS_RE = re.compile(r"\[([A-Z\.-]+)\]\s*success\s*\|")
ERROR_RE = re.compile(r"\[([A-Z\.-]+)\]\s*error:")


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_all_tickers(project_root: Path) -> list[str]:
    path = project_root / "DB" / "sp500_top300.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [str(t).strip().upper() for t in data]


def _extract_completed_tickers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    tickers = set()
    tickers.update(t.upper() for t in SUCCESS_RE.findall(text))
    tickers.update(t.upper() for t in ERROR_RE.findall(text))
    return tickers


def _running_worker_pids() -> list[int]:
    cmd = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*scripts.resume_run_all_tickers*' "
        "-or $_.CommandLine -like '*Classification.multi_ticker.run_all_tickers*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))
    return pids


def _start_resume_worker(project_root: Path) -> int:
    out_log = project_root / "run_all_tickers_resume.out.log"
    err_log = project_root / "run_all_tickers_resume.err.log"
    out_f = out_log.open("a", encoding="utf-8")
    err_f = err_log.open("a", encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "UTF-8")

    args = [
        sys.executable,
        "-u",
        "-m",
        "scripts.resume_run_all_tickers",
        "--force-retune-all",
        "--xgb-trials",
        "100",
        "--svm-trials",
        "100",
        "--rf-trials",
        "100",
        "--logreg-trials",
        "100",
    ]
    proc = subprocess.Popen(
        args,
        cwd=str(project_root),
        env=env,
        stdout=out_f,
        stderr=err_f,
    )
    return int(proc.pid)


def _watch(
    project_root: Path,
    poll_seconds: int,
    max_restarts: int,
    watchdog_log: Path,
) -> None:
    all_tickers = _load_all_tickers(project_root)
    total = len(all_tickers)
    if total == 0:
        raise RuntimeError("No tickers loaded from DB/sp500_top300.json")

    old_out = project_root / "run_all_tickers_100.out.log"
    resume_out = project_root / "run_all_tickers_resume.out.log"
    restarts = 0

    with watchdog_log.open("a", encoding="utf-8") as wf:
        wf.write(f"[{_now()}] WATCHDOG START total={total}\n")
        wf.flush()

        while True:
            completed = _extract_completed_tickers(old_out) | _extract_completed_tickers(resume_out)
            completed_count = len(completed)

            if completed_count >= total:
                wf.write(f"[{_now()}] DONE completed={completed_count}/{total}\n")
                wf.flush()
                break

            pids = _running_worker_pids()
            if not pids:
                if restarts >= max_restarts:
                    wf.write(
                        f"[{_now()}] STOP max_restarts reached "
                        f"(completed={completed_count}/{total})\n"
                    )
                    wf.flush()
                    break

                pid = _start_resume_worker(project_root)
                restarts += 1
                wf.write(
                    f"[{_now()}] RESTART pid={pid} "
                    f"completed={completed_count}/{total} restarts={restarts}\n"
                )
                wf.flush()
            else:
                wf.write(
                    f"[{_now()}] RUNNING pids={pids} "
                    f"completed={completed_count}/{total}\n"
                )
                wf.flush()

            time.sleep(max(5, poll_seconds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch and auto-resume multi-ticker run")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval")
    parser.add_argument("--max-restarts", type=int, default=200, help="Maximum auto restarts")
    parser.add_argument(
        "--watchdog-log",
        default="run_all_tickers_watchdog.log",
        help="Watchdog log file path (project-root relative)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    watchdog_log = project_root / args.watchdog_log
    _watch(
        project_root=project_root,
        poll_seconds=int(args.poll_seconds),
        max_restarts=int(args.max_restarts),
        watchdog_log=watchdog_log,
    )


if __name__ == "__main__":
    main()

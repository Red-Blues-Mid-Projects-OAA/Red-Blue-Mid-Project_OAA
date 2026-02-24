"""
Resume multi-ticker batch from the last unfinished ticker in an existing log.

Usage example:
  python -m scripts.resume_run_all_tickers --force-retune-all --xgb-trials 100 --svm-trials 100 --rf-trials 100 --logreg-trials 100
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import DB
import Classification.multi_ticker.run_all_tickers as runner


START_RE = re.compile(r"\[([A-Z\.-]+)\]\s*배치 실행 시작")
SUCCESS_RE = re.compile(r"\[([A-Z\.-]+)\]\s*success\s*\|")
ERROR_RE = re.compile(r"\[([A-Z\.-]+)\]\s*error:")


def _load_all_tickers() -> list[str]:
    path = Path("DB/sp500_top300.json")
    if path.exists():
        return [str(t).strip().upper() for t in json.loads(path.read_text(encoding="utf-8"))]
    return [str(t).strip().upper() for t in DB.TICKERS]


def _find_resume_index_from_log(all_tickers: list[str], log_path: Path) -> int:
    """
    Return 0-based index to resume from.
    - If last started ticker has completion in later log lines, resume at next ticker.
    - Otherwise, resume at that last started ticker.
    """
    if not log_path.exists():
        return 0

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    last_start_idx = -1
    last_start_ticker = None

    for i, line in enumerate(lines):
        m = START_RE.search(line)
        if m:
            last_start_idx = i
            last_start_ticker = m.group(1).upper()

    if last_start_ticker is None:
        return 0

    try:
        ticker_pos = all_tickers.index(last_start_ticker)
    except ValueError:
        return 0

    completed = False
    for line in lines[last_start_idx + 1 :]:
        m_s = SUCCESS_RE.search(line)
        if m_s and m_s.group(1).upper() == last_start_ticker:
            completed = True
            break
        m_e = ERROR_RE.search(line)
        if m_e and m_e.group(1).upper() == last_start_ticker:
            completed = True
            break

    if completed:
        return min(ticker_pos + 1, len(all_tickers))
    return ticker_pos


def _resolve_resume_log(arg_log: str) -> Path:
    """
    Resolve resume source log path.
    - explicit path: use as-is
    - "auto": choose the newest existing log between legacy/current logs
    """
    if arg_log != "auto":
        return Path(arg_log)

    candidates = [
        Path("run_all_tickers_resume.out.log"),
        Path("run_all_tickers_100.out.log"),
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return Path("run_all_tickers_resume.out.log")
    return max(existing, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume multi-ticker run from last unfinished ticker")
    parser.add_argument(
        "--log",
        default="auto",
        help='Path to existing stdout log. Use "auto" to select the newest available run log.',
    )
    parser.add_argument("--start-index", type=int, default=None, help="1-based explicit start index override")
    parser.add_argument("--force-retune-all", action="store_true", help="Retune all 4 models per ticker")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "regularized"], help="Optimization profile")
    parser.add_argument("--xgb-trials", type=int, default=100, help="XGB optimize trial count")
    parser.add_argument("--svm-trials", type=int, default=100, help="SVM optimize trial count")
    parser.add_argument("--rf-trials", type=int, default=100, help="RF optimize trial count")
    parser.add_argument("--logreg-trials", type=int, default=100, help="LogReg optimize trial count")
    args = parser.parse_args()

    all_tickers = _load_all_tickers()
    log_path = _resolve_resume_log(str(args.log))

    if args.start_index is not None:
        start_idx = max(0, args.start_index - 1)
    else:
        start_idx = _find_resume_index_from_log(all_tickers, log_path)

    remaining = all_tickers[start_idx:]
    if not remaining:
        print("[RESUME] no remaining tickers to run.")
        return

    print(
        f"[RESUME] log={log_path} | "
        f"[RESUME] start={start_idx + 1}/{len(all_tickers)}, "
        f"remaining={len(remaining)}, first={remaining[0]}, last={remaining[-1]}"
    )

    # Override module-level ticker list so runner only processes remaining universe.
    DB.TICKERS = remaining
    runner.TICKERS = remaining

    runner.run_all_tickers(
        force_retune_all=bool(args.force_retune_all),
        optimize_profile=args.profile,
        n_trials_by_model={
            "xgb": args.xgb_trials,
            "svm": args.svm_trials,
            "rf": args.rf_trials,
            "logreg": args.logreg_trials,
        },
    )


if __name__ == "__main__":
    main()

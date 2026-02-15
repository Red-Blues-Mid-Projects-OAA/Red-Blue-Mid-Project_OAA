#!/usr/bin/env python3
"""Import 정책 위반을 검사하는 로컬 스크립트."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = [ROOT / "Classification", ROOT / "DB"]

BAN_PATTERNS = [
    (
        re.compile(r"^\s*sys\.path\.insert\(", re.MULTILINE),
        "sys.path.insert 사용 금지",
    ),
    (
        re.compile(r"^\s*except\s+ModuleNotFoundError", re.MULTILINE),
        "ModuleNotFoundError fallback import 금지",
    ),
    (
        re.compile(r"^\s*from\s+models\.", re.MULTILINE),
        "로컬 models.* import 금지 (Classification.models.* 사용)",
    ),
    (
        re.compile(
            r"^\s*from\s+(model_config|model_gate|split_dataset|generate_target|"
            r"build_master_dataset|feature_engineering|calculate_aapl_volume_ratio|"
            r"prepare_market_features|risk_volatility_features|momentum|macro|volatility|volume|"
            r"optimize_hyperparams|"
            r"optimize_svm_hyperparams|optimize_logreg|optimize_rf_hyperparams|"
            r"run_model_pipeline|svm_pipeline|logic_model_pipeline|run_rf_pipeline|"
            r"validate_all_models)\s+import\b",
            re.MULTILINE,
        ),
        "루트 로컬 import 금지 (Classification.* 절대 import 사용)",
    ),
    (
        re.compile(r"^\s*from\s+Preprocessing\.", re.MULTILINE),
        "Preprocessing 상대 루트 import 금지 (Classification.Preprocessing.* 사용)",
    ),
]

DB_STOCK_MANAGER_DIRECT = re.compile(
    r"^\s*from\s+DB\.stock_db_manager\s+import\s+StockDBManager\b",
    re.MULTILINE,
)


def iter_py_files() -> list[Path]:
    files: list[Path] = []
    for base in TARGET_DIRS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append(path)
    return sorted(files)


def main() -> int:
    violations: list[tuple[Path, str]] = []

    for path in iter_py_files():
        text = path.read_text(encoding="utf-8")

        for pattern, message in BAN_PATTERNS:
            if pattern.search(text):
                violations.append((path, message))

        rel = path.relative_to(ROOT)
        if (
            rel.parts[0] == "DB"
            and rel.name != "__init__.py"
            and DB_STOCK_MANAGER_DIRECT.search(text)
        ):
            violations.append((path, "DB 내부에서 StockDBManager 직접 모듈 import 금지"))

    if violations:
        print("[FAIL] Import 정책 위반이 발견되었습니다.")
        for path, msg in violations:
            print(f"  - {path.relative_to(ROOT)}: {msg}")
        return 1

    print("[PASS] Import 정책 위반이 없습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
SVM 분류 파이프라인 (Stride 앙상블).

차트 형식은 xgboost_classifier_result.png와 동일하게 2x2 구성으로 맞춥니다.
1) Feature Importance (Permutation)
2) Ensemble Probability Distribution
3) Ensemble Train-Test Gap
4) IC Stability
"""

import sys
import hashlib
from pathlib import Path

if __package__ in (None, ""):
    _PROJECT_ROOT = next(
        (
            p
            for p in Path(__file__).resolve().parents
            if (p / "Classification").is_dir() and (p / "common").is_dir()
        ),
        None,
    )
    if _PROJECT_ROOT is not None:
        sys.path.append(str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report, precision_score
from sklearn.svm import SVC

from Classification.Preprocessing.split_dataset import N_MODELS, get_stride_splits, split_dataset
from Classification.model_config import (
    PERM_IMPORTANCE_REPEATS,
    PERM_IMPORTANCE_SEED,
    PERM_IMPORTANCE_TOPK_TABLE,
    ensure_artifact_dirs,
    get_model_params_path,
    get_model_result_path,
    load_json_artifact_only,
)
from Classification.model_gate import evaluate_gate, print_gate_result
from Classification.models.common.importance import compute_permutation_importance_ic

EXPECTED_OBJECTIVE_VERSION = "target_aligned_v3_svm_no_class_weight"
EXPECTED_CV_MODE = "single_holdout_2024Q2Q3"
TSM_GATE_PROFILE = "tsm_gatehard_v1"
TSM_EXPECTED_OBJECTIVE_VERSIONS = {
    "tsm_gatehard_v1_stage1",
    "tsm_gatehard_v1_stage2",
}


def _get_feature_hash(feature_cols):
    raw = "|".join(feature_cols)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _apply_direction(proba, direction_mode):
    if direction_mode == "normal":
        return proba
    if direction_mode == "inverted":
        return 1.0 - proba
    raise ValueError(f"지원하지 않는 direction_mode 입니다: {direction_mode}")


def _compute_recency_weights(index, recency_weight_lambda):
    if recency_weight_lambda <= 0.0:
        return None
    rank = np.arange(len(index), dtype=float)
    weights = 1.0 + recency_weight_lambda * rank
    weights = np.clip(weights, 1e-8, None)
    return weights / np.mean(weights)


def _normalize_class_weight_dict(class_weight):
    if not isinstance(class_weight, dict):
        return class_weight
    out = {}
    for k, v in class_weight.items():
        try:
            key = int(k)
        except Exception:
            key = k
        out[key] = float(v)
    return out


def _get_current_data_end_date(split):
    candidates = []
    for df in [split.train, split.val, split.test, split.final_train]:
        if len(df) > 0:
            candidates.append(df.index.max())
    if not candidates:
        return None
    return max(candidates).strftime("%Y-%m-%d")


def _is_param_file_stale(param_data, feature_cols, current_data_end_date, optimize_profile):
    required_meta = ["feature_hash", "data_end_date", "profile", "objective_version", "cv_mode"]
    if optimize_profile == TSM_GATE_PROFILE:
        required_meta.extend(
            [
                "strategy_stage",
                "direction_mode",
                "recency_weight_lambda",
                "class_weight_mode",
                "gate_objective_version",
            ]
        )
    for key in required_meta:
        if key not in param_data:
            return True, f"메타데이터 누락: {key}"

    current_feature_hash = _get_feature_hash(feature_cols)
    if param_data["feature_hash"] != current_feature_hash:
        return True, "feature_hash 불일치"

    file_data_end = param_data["data_end_date"]
    if current_data_end_date is not None and file_data_end < current_data_end_date:
        return True, f"data_end_date 구버전 ({file_data_end} < {current_data_end_date})"

    if param_data["profile"] != optimize_profile:
        return True, f"profile 불일치 ({param_data['profile']} != {optimize_profile})"

    if optimize_profile == TSM_GATE_PROFILE:
        expected_versions = TSM_EXPECTED_OBJECTIVE_VERSIONS
    else:
        expected_versions = {EXPECTED_OBJECTIVE_VERSION}

    if param_data["objective_version"] not in expected_versions:
        return True, (
            f"objective_version 불일치 "
            f"({param_data['objective_version']} not in {sorted(expected_versions)})"
        )

    if param_data["cv_mode"] != EXPECTED_CV_MODE:
        return True, f"cv_mode 불일치 ({param_data['cv_mode']} != {EXPECTED_CV_MODE})"

    return False, "최신 파라미터 사용 가능"


def _ensure_best_params(split, params_path, auto_optimize=False, optimize_profile="balanced"):
    param_data, _ = load_json_artifact_only(params_path)
    current_data_end_date = _get_current_data_end_date(split)
    feature_cols = split.feature_cols

    needs_optimize = False
    reason = ""

    if param_data is None:
        needs_optimize = True
        reason = "파라미터 파일 미존재"
    else:
        stale, reason = _is_param_file_stale(
            param_data,
            feature_cols,
            current_data_end_date,
            optimize_profile=optimize_profile,
        )
        needs_optimize = stale

    if needs_optimize:
        if auto_optimize:
            print("\n  ⚠️ SVM 자동 재튜닝을 실행합니다.")
            print(f"    사유: {reason}")
            from Classification.models.svm.optimize import optimize

            optimize(
                profile=optimize_profile,
                n_trials=100,
                ticker=split.ticker,
                benchmark=split.benchmark,
                auto_update=False,
                persist_total_features_on_update=False,
                feature_source_mode="db_first",
                strategy_stage="stage2" if optimize_profile == TSM_GATE_PROFILE else "stage2",
            )
        else:
            raise RuntimeError(
                "SVM 파라미터 아티팩트가 현재 정책과 불일치합니다. "
                f"auto_optimize=False 상태에서는 실행할 수 없습니다. 사유: {reason}"
            )


def load_svm_params(params_path):
    """best_svm_params.json을 로드하고, 없으면 기본값을 반환합니다."""
    default_params = {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42,
    }

    default_meta = {
        "decision_threshold": 0.5,
        "fit_mode": "final_train",
    }
    default_controls = {
        "direction_mode": "normal",
        "recency_weight_lambda": 0.0,
    }

    data, used_path = load_json_artifact_only(params_path)
    if data is None:
        return default_params, default_meta, default_controls

    try:
        params = data.get("best_params", data)
        if not isinstance(params, dict):
            return default_params, default_meta, default_controls

        merged = {**default_params, **params}
        merged["probability"] = True
        merged.setdefault("random_state", 42)
        merged["class_weight"] = _normalize_class_weight_dict(merged.get("class_weight"))
        meta = {
            "decision_threshold": 0.5,
            "fit_mode": data.get("fit_mode", "final_train"),
        }
        controls = {
            "direction_mode": str(data.get("direction_mode", "normal")),
            "recency_weight_lambda": float(data.get("recency_weight_lambda", 0.0)),
        }
        if controls["direction_mode"] not in {"normal", "inverted"}:
            controls["direction_mode"] = "normal"
        print(f"  SVM 파라미터 로드 경로: {used_path}")
        return merged, meta, controls
    except Exception as exc:
        print(f"[WARN] 파라미터 로드 실패: {exc}")
        return default_params, default_meta, default_controls


def safe_spearmanr(x, y):
    """상수열/짧은 시계열에서 NaN 안전 처리를 포함한 Spearman 계산."""
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return 0.0, 1.0, True
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0, 1.0, True
    ic, p_value = spearmanr(x, y)
    if np.isnan(ic):
        return 0.0, 1.0, True
    if np.isnan(p_value):
        return float(ic), 1.0, True
    return float(ic), float(p_value), False


def ensemble_predict_proba(models, x_df):
    """(model, scaler, offset) 리스트에 대해 앙상블 평균 확률을 반환합니다."""
    probas = []
    for model, scaler, _ in models:
        if scaler is None:
            probas.append(model.predict_proba(x_df)[:, 1])
        else:
            x_scaled = scaler.transform(x_df)
            probas.append(model.predict_proba(x_scaled)[:, 1])
    return np.mean(probas, axis=0)


def run_pipeline(
    auto_optimize=False,
    optimize_profile="balanced",
    return_metrics=False,
    return_details=False,
    ticker="AAPL",
    benchmark="SP500",
    drop_features=None,
    save_plot=True,
    compute_importance=True,
    split_override=None,
):
    """SVM stride 앙상블 학습/평가 및 결과 차트 저장."""
    print("\n" + "=" * 70)
    print("1. SVM Stride 앙상블 파이프라인 시작")
    print("=" * 70)

    if split_override is None:
        split = split_dataset(
            ticker=ticker,
            benchmark=benchmark,
            drop_features=drop_features,
        )
    else:
        split = split_override
        if drop_features:
            print("  [INFO] split_override가 제공되어 drop_features는 무시됩니다.")
    ticker = split.ticker
    benchmark = split.benchmark
    params_path = get_model_params_path("svm", ticker)
    result_path = get_model_result_path("svm", ticker)
    stride_splits = get_stride_splits(split)
    feature_cols = split.feature_cols
    x_test = split.test[feature_cols]
    y_test = split.test["Target_Class"].astype(int)

    _ensure_best_params(
        split,
        params_path=params_path,
        auto_optimize=auto_optimize,
        optimize_profile=optimize_profile,
    )
    svm_params, meta, controls = load_svm_params(params_path)
    threshold = 0.5
    fit_mode = meta.get("fit_mode", "final_train")
    direction_mode = controls["direction_mode"]
    recency_weight_lambda = controls["recency_weight_lambda"]
    print(f"\n사용 SVM 파라미터: {svm_params}")
    print(f"추론 임계값: {threshold:.2f} | 학습 모드: {fit_mode}")
    print(f"direction_mode: {direction_mode} | recency λ: {recency_weight_lambda:.6f}")

    models = []
    train_accs = []
    all_test_probas = []

    print("\n" + "=" * 70)
    print(f"2. Stride split별 Final Refit ({N_MODELS}개 모델)")
    print("=" * 70)

    for ss in stride_splits:
        if fit_mode == "train_only":
            x_refit = ss.train[feature_cols]
            y_refit = ss.train["Target_Class"].astype(int)
        else:
            x_refit = ss.final_train[feature_cols]
            y_refit = ss.final_train["Target_Class"].astype(int)

        # split_dataset()에서 이미 스케일된 입력을 받아 추가 스케일링을 하지 않습니다.
        model = SVC(**svm_params)
        fit_kwargs = {}
        recency_weights = _compute_recency_weights(
            ss.train.index if fit_mode == "train_only" else ss.final_train.index,
            recency_weight_lambda,
        )
        if recency_weights is not None:
            fit_kwargs["sample_weight"] = recency_weights
        model.fit(x_refit, y_refit, **fit_kwargs)
        models.append((model, None, ss.offset))

        refit_proba = model.predict_proba(x_refit)[:, 1]
        refit_proba = _apply_direction(refit_proba, direction_mode)
        y_refit_pred = (refit_proba >= threshold).astype(int)
        train_acc = accuracy_score(y_refit, y_refit_pred)
        train_accs.append(train_acc)
        test_proba = model.predict_proba(x_test)[:, 1]
        test_proba = _apply_direction(test_proba, direction_mode)
        all_test_probas.append(test_proba)

        print(
            f"  모델 {ss.offset}: Refit {len(x_refit):>4d}건 | "
            f"Train Acc {train_acc * 100:.1f}%"
        )

    ensemble_proba = np.mean(all_test_probas, axis=0)
    ensemble_pred = (ensemble_proba >= threshold).astype(int)
    avg_train_acc = float(np.mean(train_accs))

    acc = accuracy_score(y_test, ensemble_pred)
    prec = precision_score(y_test, ensemble_pred, zero_division=0)
    gap_signed = avg_train_acc - acc
    gap_abs = abs(gap_signed)
    proba_std_test = float(np.std(ensemble_proba))
    proba_unique_test = int(np.unique(np.round(ensemble_proba, 6)).size)

    excess_return = split.test[split.target_col] - split.test[split.benchmark_target_col]
    ic, p_value, ic_degenerate = safe_spearmanr(ensemble_proba, excess_return)

    mid_idx = len(split.test) // 2
    first_half = split.test.iloc[:mid_idx]
    second_half = split.test.iloc[mid_idx:]
    proba_first = ensemble_proba[:mid_idx]
    proba_second = ensemble_proba[mid_idx:]
    excess_first = first_half[split.target_col] - first_half[split.benchmark_target_col]
    excess_second = second_half[split.target_col] - second_half[split.benchmark_target_col]
    ic_first, p_first, _ = safe_spearmanr(proba_first, excess_first)
    ic_second, p_second, _ = safe_spearmanr(proba_second, excess_second)

    baseline_ic_fi = np.nan
    feat_imp_df = None
    top1_share = np.nan
    top3_share = np.nan
    hhi = np.nan

    if compute_importance:
        baseline_ic_fi, _, feat_imp_df = compute_permutation_importance_ic(
            x_test=x_test,
            y_test=y_test,
            alpha_diff=excess_return,
            predict_proba_fn=(
                lambda x_df: _apply_direction(ensemble_predict_proba(models, x_df), direction_mode)
            ),
            threshold=threshold,
            n_repeats=PERM_IMPORTANCE_REPEATS,
            seed=PERM_IMPORTANCE_SEED,
        )
        positive_deltas = feat_imp_df["ic_drop_mean"].clip(lower=0.0)
        positive_total = float(positive_deltas.sum())
        if positive_total > 0.0:
            share = positive_deltas / positive_total
            top1_share = float(share.iloc[0])
            top3_share = float(share.iloc[:3].sum())
            hhi = float((share ** 2).sum())

    print("\n" + "=" * 70)
    print("3. 성능 평가")
    print("=" * 70)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"IC       : {ic:+.4f} (p={p_value:.4f})")
    print(
        f"Gap      : signed={gap_signed * 100:.2f}%p "
        f"/ abs={gap_abs * 100:.2f}%p ({gap_abs * 10000:.0f}bp)"
    )
    if ic_degenerate:
        print("[WARN] IC degenerate: constant/near-constant probability input")
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=["Lose(0)", "Win(1)"]))

    if compute_importance and feat_imp_df is not None:
        print("\n[Permutation ΔIC Feature Importance Top 5]")
        for i, row in enumerate(feat_imp_df.head(5).itertuples(index=False), 1):
            print(
                f"  {i}. {row.feature:25s} "
                f"ΔIC {row.ic_drop_mean:+.4f} ± {row.ic_drop_std:.4f} | "
                f"Norm {row.ic_drop_norm_mean * 100:+.1f}%"
            )
    else:
        print("\n[Permutation ΔIC Feature Importance] 생략 (compute_importance=False)")

    print("\n" + "=" * 70)
    print("4. 결과 차트 저장 (xgboost 형식과 동일 2x2 레이아웃)")
    print("=" * 70)

    if save_plot:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # (1) 피처 중요도: Permutation ΔIC 평균/표준편차
        if compute_importance and feat_imp_df is not None and len(feat_imp_df) > 0:
            axes[0, 0].barh(
                feat_imp_df["feature"],
                feat_imp_df["ic_drop_mean"],
                xerr=feat_imp_df["ic_drop_std"],
                color="steelblue",
                edgecolor="black",
                alpha=0.85,
                error_kw={"elinewidth": 1.2, "capsize": 3},
            )
            axes[0, 0].set_title("Feature Importance (Permutation ΔIC, mean±std)", fontsize=13)
            axes[0, 0].set_xlabel("ΔIC = IC_baseline - IC_permuted")
            axes[0, 0].axvline(x=0.0, color="black", linestyle="-", linewidth=0.8)
            axes[0, 0].invert_yaxis()

            topk_df = feat_imp_df.head(PERM_IMPORTANCE_TOPK_TABLE)
            table_lines = ["rank | feature | ΔIC mean±std | normalized%"]
            for rank, row in enumerate(topk_df.itertuples(index=False), 1):
                feat_name = row.feature if len(row.feature) <= 18 else row.feature[:15] + "..."
                table_lines.append(
                    f"{rank:>2d} | {feat_name:18s} | "
                    f"{row.ic_drop_mean:+.4f}±{row.ic_drop_std:.4f} | "
                    f"{row.ic_drop_norm_mean * 100:+.1f}%"
                )

            axes[0, 0].text(
                1.02,
                1.00,
                "\n".join(table_lines),
                transform=axes[0, 0].transAxes,
                ha="left",
                va="top",
                fontsize=8,
                family="monospace",
                clip_on=False,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
            )
            axes[0, 0].text(
                0.0,
                -0.22,
                (
                    f"baseline IC={baseline_ic_fi:+.4f} | n_repeats={PERM_IMPORTANCE_REPEATS} | "
                    f"seed={PERM_IMPORTANCE_SEED} | normalized=ΔIC/|IC_baseline|"
                ),
                transform=axes[0, 0].transAxes,
                fontsize=8.5,
                ha="left",
                va="top",
            )
        else:
            axes[0, 0].axis("off")
            axes[0, 0].text(
                0.5,
                0.5,
                "Feature Importance skipped\n(compute_importance=False)",
                ha="center",
                va="center",
                fontsize=12,
            )

        # (2) 예측 확률 분포
        axes[0, 1].hist(
            ensemble_proba[y_test == 1], bins=30, alpha=0.6,
            label=f"Win ({ticker} > {benchmark})", color="green", edgecolor="black"
        )
        axes[0, 1].hist(
            ensemble_proba[y_test == 0], bins=30, alpha=0.6,
            label=f"Lose ({ticker} <= {benchmark})", color="red", edgecolor="black"
        )
        axes[0, 1].axvline(
            x=threshold, color="black", linestyle="--", label=f"Threshold ({threshold:.2f})"
        )
        axes[0, 1].set_title("Ensemble Probability Distribution", fontsize=13)
        axes[0, 1].set_xlabel(f"P({ticker} beats {benchmark})")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].legend()

        # (3) 학습/테스트 정확도 격차(일반화 성능 점검)
        gap_labels = ["Avg Train", "Test"]
        gap_values = [avg_train_acc * 100, acc * 100]
        gap_colors = ["#4CAF50", "#FF5722"]
        bars = axes[1, 0].bar(gap_labels, gap_values, color=gap_colors, edgecolor="black", width=0.5)
        for bar, val in zip(bars, gap_values):
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                fontsize=12,
                fontweight="bold",
            )
        axes[1, 0].set_title(
            f"Ensemble Train-Test Gap (signed={gap_signed * 100:.1f}%p, abs={gap_abs * 100:.1f}%p)",
            fontsize=13,
        )
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].set_ylim(0, 100)
        axes[1, 0].axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random (50%)")
        axes[1, 0].legend()

        # (4) IC 안정성: 전반기/후반기/전체 비교
        ic_labels = ["1st Half", "2nd Half", "Full"]
        ic_values = [
            0.0 if np.isnan(ic_first) else ic_first,
            0.0 if np.isnan(ic_second) else ic_second,
            0.0 if np.isnan(ic) else ic,
        ]
        ic_colors = ["green" if v > 0 else "red" for v in ic_values]
        bars = axes[1, 1].bar(ic_labels, ic_values, color=ic_colors, edgecolor="black", width=0.5)
        for bar, val in zip(bars, ic_values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.005 if val >= 0 else -0.02),
                f"{val:+.3f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
        axes[1, 1].set_title("IC Stability (Stride Ensemble)", fontsize=13)
        axes[1, 1].set_ylabel("Information Coefficient")
        axes[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.8)
        axes[1, 1].axhline(y=0.05, color="blue", linestyle="--", alpha=0.5, label="IC=0.05 (Strong)")
        axes[1, 1].legend()

        plt.tight_layout(rect=[0.0, 0.03, 0.83, 1.0])
        ensure_artifact_dirs()
        result_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(result_path, dpi=150)
        plt.close()
        print(f"차트 저장 완료: {result_path}")
    else:
        print(f"차트 저장 생략: save_plot=False ({result_path})")

    print("\n" + "=" * 70)
    print(f"최종 평가 요약 (Stride {N_MODELS}개 모델 SVM 앙상블)")
    print("=" * 70)
    print(f"  Accuracy       : {acc * 100:.2f}%")
    print(f"  Precision      : {prec * 100:.2f}%")
    print(f"  IC (전체)      : {ic:+.4f} (p={p_value:.4f})")
    print(f"  IC (전반기)    : {ic_first:+.4f} (p={p_first:.4f})")
    print(f"  IC (후반기)    : {ic_second:+.4f} (p={p_second:.4f})")
    print(
        f"  Train-Test Gap : signed={gap_signed * 100:.1f}%p, "
        f"abs={gap_abs * 100:.1f}%p ({gap_abs * 10000:.0f}bp)"
    )
    if feat_imp_df is not None and len(feat_imp_df) > 0:
        top_row = feat_imp_df.iloc[0]
        print(
            f"  Top Feature    : {top_row['feature']} "
            f"(ΔIC={top_row['ic_drop_mean']:+.4f}, norm={top_row['ic_drop_norm_mean'] * 100:+.1f}%)"
        )
    print(
        f"  Top3 합계      : {top3_share * 100:.1f}% (양의 ΔIC share)"
        if not np.isnan(top3_share)
        else "  Top3 합계      : N/A"
    )
    print(f"  HHI 집중도     : {hhi:.4f}" if not np.isnan(hhi) else "  HHI 집중도     : N/A")
    print("=" * 70)

    gate = evaluate_gate(
        {
            "accuracy": acc,
            "ic": ic,
            "gap": gap_abs,
            "gap_signed": gap_signed,
            "gap_abs": gap_abs,
            "ic_first": ic_first,
            "ic_second": ic_second,
        }
    )
    print_gate_result("SVM", gate)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "ic": float(ic),
        "ic_p_value": float(p_value),
        "gap": float(gap_abs),
        "gap_signed": float(gap_signed),
        "gap_abs": float(gap_abs),
        "ic_first": float(ic_first),
        "ic_second": float(ic_second),
        "proba_std_test": float(proba_std_test),
        "proba_unique_test": int(proba_unique_test),
        "ic_degenerate": bool(ic_degenerate),
        "overall_pass": bool(gate["pass_all"]),
    }

    if return_metrics:
        if return_details:
            return metrics, models, ensemble_proba, ic
        return metrics
    return models, ensemble_proba, ic


if __name__ == "__main__":
    run_pipeline(auto_optimize=False, optimize_profile="balanced")

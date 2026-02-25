"""
평균-분산 최적화(MVO) 포트폴리오 최적화 모듈.

목적함수: max E(R_portfolio) - (1/2) * λ * w^T Σ w
         = min (1/2) * λ * w^T Σ w - w^T μ

제약조건: sum(w) = 1, w_i ≥ 0 (공매도 금지)
"""

import numpy as np
from scipy.optimize import minimize


def optimize_portfolio(mu, cov_matrix, lambda_final):
    """
    평균-분산 최적화(MVO)를 수행하여 각 자산의 최적 투자 비중을 산출합니다.

    Args:
        mu (np.array): 각 자산의 기대수익률 벡터 (Adjusted_E_Total)
        cov_matrix (np.array): EWMA 공분산 행렬 (선택 종목의 sub-matrix)
        lambda_final (float): 투자자의 최종 위험 회피 계수 (λ)

    Returns:
        np.array: 각 자산의 최적 투자 비중 배열 (합=1, 각 원소 ≥ 0)
    """
    num_assets = len(mu)
    mu = np.array(mu, dtype=float)
    sigma = np.array(cov_matrix, dtype=float)

    # 목적함수: min (1/2) * λ * w^T Σ w - w^T μ
    def objective(w):
        portfolio_variance = w.T @ sigma @ w
        portfolio_return = w.T @ mu
        return 0.5 * lambda_final * portfolio_variance - portfolio_return

    # 제약조건: sum(w) = 1
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})

    # 바운드: w_i ≥ 0 (공매도 금지, 최소 비중 제한 없음)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))

    # 초기 비중: 1/N 동일가중
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    # 최적화 솔버 실행
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        print(f"⚠️ 최적화 실패: {result.message} → 1/N 동일가중 반환")
        return initial_weights

    # 소수점 4자리 반올림
    return np.round(result.x, 4)


if __name__ == "__main__":
    # 단독 테스트
    print("MVO 포트폴리오 최적화 단독 테스트")

    mu_test = np.array([0.05, 0.08, 0.12])
    cov_test = np.array([
        [0.001, 0.000, 0.000],
        [0.000, 0.004, 0.002],
        [0.000, 0.002, 0.010],
    ])

    for name, lam in [("거북이", 23.10), ("사자", 10.04), ("독수리", 3.50)]:
        w = optimize_portfolio(mu_test, cov_test, lam)
        ret = w @ mu_test
        print(f"  [{name}] λ={lam:.2f} → 비중={w} (합={np.sum(w):.4f}, 기대수익률={ret:.4f})")

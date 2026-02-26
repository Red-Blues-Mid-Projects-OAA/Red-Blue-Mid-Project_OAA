"""
평균-분산 최적화(MVO) 포트폴리오 최적화 모듈.

목적함수: max E(R_portfolio) - (1/2) * λ * w^T Σ w
         = min (1/2) * λ * w^T Σ w - w^T μ

제약조건: sum(w) = 1, w_i = 0 or (0.05 ≤ w_i ≤ max_w)
최소 종목 수 제약: sum(y) >= min_count
"""

import numpy as np
import cvxpy as cp
import warnings


def optimize_portfolio(mu, cov_matrix, lambda_final):
    """
    평균-분산 최적화(MVO)를 수행하여 각 자산의 최적 투자 비중을 산출합니다.
    혼합 정수 이차 계획법(MIQP)을 사용하여 리스크 타입별 최소 종목 수 할당과 
    최소 편입 비중(5%) 조건을 종목 탈락과 함께 동적으로 최적화합니다.

    Args:
        mu (np.array): 각 자산의 기대수익률 벡터 (Adjusted_E_Total)
        cov_matrix (np.array): EWMA 공분산 행렬 (선택 종목의 sub-matrix)
        lambda_final (float): 투자자의 최종 위험 회피 계수 (λ)

    Returns:
        np.array: 각 자산의 최적 투자 비중 배열 (합=1)
    """
    num_assets = len(mu)
    mu_arr = np.array(mu, dtype=float)
    sigma_arr = np.array(cov_matrix, dtype=float)

    # 종목 1개인 경우 예외처리
    if num_assets == 1:
        return np.array([1.0])

    # 리스크 타입별 설정 (새로운 람다 스케일 적용: 거북이~23.1, 강아지~7.7, 사자~4.62, 독수리~3.3)
    if lambda_final >= 20.0:    # 거북이
        max_w = 0.20
        min_count = 10
    elif lambda_final >= 7.0:   # 강아지
        max_w = 0.35
        min_count = 7
    elif lambda_final >= 4.0:   # 사자
        max_w = 0.50
        min_count = 4
    else:                       # 독수리
        max_w = 1.00
        min_count = 1

    # 전달된 종목 수가 min_count보다 적을 수 있으므로 보정
    min_count = min(min_count, num_assets)
    min_w = 0.05

    # Greedy Iterative Drop Heuristic
    # 1. 연속 QP를 풀어서 가장 가중치가 낮은 종목 1개씩 제거 (w_i = 0 강제)
    # 2. 풀에 남은 종목의 수가 min_count개가 될 때까지 반복
    # 3. 마지막으로 남은 min_count개의 종목에 대해서 엄격한 제약(w_i >= 0.05, w_i <= max_w) 적용
    
    active_indices = list(range(num_assets))

    while len(active_indices) > min_count:
        w = cp.Variable(num_assets)
        risk = cp.quad_form(w, sigma_arr)
        ret = mu_arr.T @ w
        objective = cp.Minimize(0.5 * lambda_final * risk - ret)
        
        # 중간 탐색 과정에서는 최소 비중(5%) 제약을 걸지 않고, 오직 0초과 제약만 둡니다.
        # 가장 경쟁력이 없는(가중치가 0에 가까운) 종목을 찾기 위함입니다.
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_w
        ]
        
        for i in range(num_assets):
            if i not in active_indices:
                constraints.append(w[i] == 0)

        prob = cp.Problem(objective, constraints)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                prob.solve(solver=cp.OSQP)
            except Exception:
                prob.solve(solver=cp.ECOS)

        if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
            break

        current_weights = np.array(w.value)
        
        # 현재 활성화된 종목 중에서 가중치가 가장 낮은 1개를 찾아 제거
        worst_i = min(active_indices, key=lambda i: current_weights[i])
        
        # 만약 가장 가중치가 낮은 종목조차 5% 이상이라면 더 이상 제거할 필요가 없으므로 조기 종료
        if current_weights[worst_i] >= min_w:
            break
            
        active_indices.remove(worst_i)

    # ──────────────────────────────────────────────────────────────────
    # [Final Solve] 남은 active_indices 종목들에 대해서 완벽한 제약조건 적용
    # ──────────────────────────────────────────────────────────────────
    w = cp.Variable(num_assets)
    risk = cp.quad_form(w, sigma_arr)
    ret = mu_arr.T @ w
    objective = cp.Minimize(0.5 * lambda_final * risk - ret)
    
    # 남은 종목에 max_w를 모두 주어도 1.0(100%)을 채울 수 없다면 해가 없으므로 max_w 강제 완화
    current_max_w = max_w if max_w * len(active_indices) >= 1.0 else 1.0

    final_constraints = [
        cp.sum(w) == 1,
    ]
    for i in range(num_assets):
        if i in active_indices:
            final_constraints.append(w[i] >= min_w)
            final_constraints.append(w[i] <= current_max_w)
        else:
            final_constraints.append(w[i] == 0)

    prob = cp.Problem(objective, final_constraints)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            prob.solve(solver=cp.OSQP)
        except Exception:
            try:
                prob.solve(solver=cp.ECOS)
            except Exception as e:
                print(f"⚠️ QP 연속최적화 실패: {e} → 1/N 동일가중 반환")
                return np.ones(num_assets) / num_assets

    if prob.status not in ["optimal", "optimal_inaccurate"] or w.value is None:
        print(f"⚠️ QP 최종해 불가. Status: {prob.status} → 1/N 반환")
        return np.ones(num_assets) / num_assets

    weights = np.array(w.value)
    weights = np.where(weights < min_w / 2, 0.0, weights)
    
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        weights = np.ones(num_assets) / num_assets

    return np.round(weights, 4)

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

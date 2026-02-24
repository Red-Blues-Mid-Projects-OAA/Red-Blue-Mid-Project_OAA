import numpy as np
from scipy.optimize import minimize
import pandas as pd

def optimize_portfolio(expected_returns, cov_matrix, lambda_final):
    """
    평균-분산 최적화(MVO)를 수행하여 각 자산의 최적 투자 비중을 산출합니다.
    
    Args:
        expected_returns (list or np.array): 각 자산의 예상 수익률 배열 (연환산 등 스케일된 값 선호)
        cov_matrix (list of lists or np.array): 각 자산간의 공분산 행렬
        lambda_final (float): 투자자의 최종 결정 위험 회피 계수 (Lambda)
        
    Returns:
        np.array: 각 자산의 최적 투자 비중 배열 (합=1)
    """
    num_assets = len(expected_returns)
    mu = np.array(expected_returns)
    sigma = np.array(cov_matrix)
    
    # 목적함수: min (0.5 * lambda * w.T * Sigma * w - w.T * mu)
    def objective_function(weights):
        portfolio_variance = weights.T @ sigma @ weights
        portfolio_return = weights.T @ mu
        return 0.5 * lambda_final * portfolio_variance - portfolio_return
        
    # 제약조건: sum(weights) == 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    # 바운드: 극단적인 쏠림 방지를 위해 모든 개별 종목 최소 2%, 최대 100% 한도 설정
    bounds = tuple((0.02, 1.0) for _ in range(num_assets))
    
    # 초기 비중: 1/N 동일가중
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    
    # 최적화 솔버 실행
    result = minimize(
        objective_function, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    if not result.success:
        print(f"최적화 실패: {result.message}")
        # 실패 시 1/N 초기 비중 반환
        return initial_weights
    
    # 소수점 4자리 반올림 처리
    return np.round(result.x, 4)

if __name__ == "__main__":
    # 단독 테스트를 위한 임의 Mock Data
    print("MVO 포트폴리오 최적화 단독 테스트 실행")
    
    # 임의로 생성한 3개 자산 수익률 (연환산 스케일: 5%, 8%, 12%)
    mu_mock = [0.05, 0.08, 0.12]
    
    # 임의로 생성한 3x3 공분산 행렬 (변동성이 리턴에 비례)
    cov_mock = [
        [0.001, 0.000, 0.000],   # 자산 A (저위험)
        [0.000, 0.004, 0.002],   # 자산 B (중위험)
        [0.000, 0.002, 0.010]    # 자산 C (고위험)
    ]
    
    print("\n[테스트 1] 보수적 성향 (거북이, lambda_final = 23.10)")
    w_cons = optimize_portfolio(mu_mock, cov_mock, 23.10)
    print(f"최적 비중: {w_cons}")
    print(f"비중 합산: {np.sum(w_cons):.4f}")
    
    print("\n[테스트 2] 중립 성향 (사자, lambda_final = 10.04)")
    w_neut = optimize_portfolio(mu_mock, cov_mock, 10.04)
    print(f"최적 비중: {w_neut}")
    print(f"비중 합산: {np.sum(w_neut):.4f}")
    
    print("\n[테스트 3] 매우 공격적 성향 (독수리, lambda_final = 3.50)")
    w_agg = optimize_portfolio(mu_mock, cov_mock, 3.50)
    print(f"최적 비중: {w_agg}")
    print(f"비중 합산: {np.sum(w_agg):.4f}")

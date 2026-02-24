from risk_profile import calculate_risk_profile

def test_risk_profile_v3():
    # 케이스 1: 거북이 (B형 0개) -> lambda_mbti = 23.10
    # 슬라이더: 950만 (-5%) -> lambda_numeric = 1.155 / 0.05 = 23.10
    # 1. 거북이/안전지형 (ISTJ) - ['I','I','I', 'S','S','S', 'T','T','T', 'J','J','J']
    answers1 = ['I']*3 + ['S']*3 + ['T']*3 + ['J']*3
    loss1 = 9500000
    res1 = calculate_risk_profile(answers1, loss1)
    print(f"CASE 1 (Turtle/Conservative): {res1}")
    assert res1['lambda_mbti'] == 23.10
    assert res1['lambda_numeric'] == 23.10
    assert res1['lambda_final'] == 23.10

    # 케이스 2: 독수리 (B형 12개) -> lambda_mbti = 3.50
    # 슬라이더: 670만 (-33%) -> lambda_numeric = 1.155 / 0.33 = 3.50
    # 2. 독수리/매우 공격적 (ENFP) - ['E','E','E', 'N','N','N', 'F','F','F', 'P','P','P']
    answers2 = ['E']*3 + ['N']*3 + ['F']*3 + ['P']*3
    loss2 = 6700000
    res2 = calculate_risk_profile(answers2, loss2)
    print(f"CASE 2 (Eagle/Aggressive): {res2}")
    assert res2['lambda_mbti'] == 3.50
    assert res2['lambda_numeric'] == 3.50
    assert res2['lambda_final'] == 3.50

    # 케이스 3: 독수리 (B형 12개, lambda=3.50) vs 보수적 슬라이더 (950만, lambda_numeric=23.10)
    # 최종 = (0.7 * 23.10) + (0.3 * 3.50) = 16.17 + 1.05 = 17.22
    answers3 = ['E']*3 + ['N']*3 + ['F']*3 + ['P']*3
    loss3 = 9500000
    res3 = calculate_risk_profile(answers3, loss3)
    print(f"CASE 3 (Aggressive MBTI but Conservative Slider): {res3}")
    assert res3['lambda_mbti'] == 3.50
    assert res3['lambda_numeric'] == 23.10
    assert res3['lambda_final'] == 17.22

    print("\n[V3] 모든 7:3 MVO 최적화 테스트 케이스 통과!")

if __name__ == "__main__":
    test_risk_profile_v3()

from risk_profile import calculate_risk_profile

def test_risk_profile_v2():
    # 케이스 1: 거북이 (B형 0개) -> lambda_mbti = 13.2
    # 슬라이더: 950만 (-5%) -> lambda_slider = 13.2
    answers1 = ['A'] * 12
    loss1 = 9500000
    res1 = calculate_risk_profile(answers1, loss1)
    print(f"CASE 1 (Turtle/Conservative): {res1}")
    assert res1['lambda_mbti'] == 13.2
    assert res1['lambda_slider'] == 13.2
    assert res1['lambda_final'] == 13.2

    # 케이스 2: 독수리 (B형 12개) -> lambda_mbti = 2.0
    # 슬라이더: 670만 (-33%) -> lambda_slider = 2.0
    answers2 = ['B'] * 12
    loss2 = 6700000
    res2 = calculate_risk_profile(answers2, loss2)
    print(f"CASE 2 (Eagle/Aggressive): {res2}")
    assert res2['lambda_mbti'] == 2.0
    assert res2['lambda_slider'] == 2.0
    assert res2['lambda_final'] == 2.0

    # 케이스 3: 독수리 (B형 12개, lambda=2.0) vs 보수적 슬라이더 (950만, lambda=13.2)
    # 허세 방지 필터링 -> 13.2 선택
    answers3 = ['B'] * 12
    loss3 = 9500000
    res3 = calculate_risk_profile(answers3, loss3)
    print(f"CASE 3 (Aggressive MBTI but Conservative Slider): {res3}")
    assert res3['lambda_mbti'] == 2.0
    assert res3['lambda_slider'] == 13.2
    assert res3['lambda_final'] == 13.2

    print("\n[V2] 모든 테스트 케이스 통과!")

if __name__ == "__main__":
    test_risk_profile_v2()

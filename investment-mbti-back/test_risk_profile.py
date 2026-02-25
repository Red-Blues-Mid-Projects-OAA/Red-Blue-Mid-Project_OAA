from risk_profile import calculate_risk_profile

def test_risk_profile_discrete():
    # 케이스 1: 완벽한 거북이 (MBTI 4, Slider 4 -> 최종 4)
    # B형 0개 -> Level 4
    # 슬라이더 950만 (-5%) -> Level 4
    # 점수: 0.7*4 + 0.3*4 = 4.0 -> final_level = 4
    answers1 = ['A'] * 12
    loss1 = 9500000
    res1 = calculate_risk_profile(answers1, loss1)
    print(f"CASE 1 (Turtle/Conservative): {res1}")
    assert res1['lambda_final'] == 23.10
    assert res1['persona'] == "안전 지향 (거북이)"

    # 케이스 2: 완벽한 독수리 (MBTI 1, Slider 1 -> 최종 1)
    # B형 12개 -> Level 1
    # 슬라이더 670만 (-33%) -> Level 1
    # 점수: 0.7*1 + 0.3*1 = 1.0 -> final_level = 1
    answers2 = ['B'] * 12
    loss2 = 6700000
    res2 = calculate_risk_profile(answers2, loss2)
    print(f"CASE 2 (Eagle/Aggressive): {res2}")
    assert res2['lambda_final'] == 3.50
    assert res2['persona'] == "공격적 독수리 (독수리)"

    # 케이스 3: 공격적 MBTI(Level 1) vs 보수적 슬라이더(Level 4)
    # B형 12개 -> Level 1
    # 슬라이더 950만 (-5%) -> Level 4
    # 점수: 0.7*4 + 0.3*1 = 2.8 + 0.3 = 3.1 -> final_level = 3
    # 반환되는 람다는 무조건통일 (Level 3 = 16.57)
    answers3 = ['B'] * 12
    loss3 = 9500000
    res3 = calculate_risk_profile(answers3, loss3)
    print(f"CASE 3 (Aggressive MBTI but Conservative Slider): {res3}")
    assert res3['lambda_final'] == 16.57
    assert res3['persona'] == "신중한 탐험가 (강아지)"

    # 케이스 4: 균형형 사자 (MBTI 2, Slider 2 -> 최종 2)
    # B형 7개 -> Level 2
    # 슬라이더 800만 (-20%) -> Level 2
    # 점수: 0.7*2 + 0.3*2 = 2.0 -> final_level = 2
    # 반환되는 람다는 무조건통일 (Level 2 = 10.04)
    answers4 = ['B'] * 7 + ['A'] * 5
    loss4 = 8000000
    res4 = calculate_risk_profile(answers4, loss4)
    print(f"CASE 4 (Lion/Balanced): {res4}")
    assert res4['lambda_final'] == 10.04
    assert res4['persona'] == "균형 잡힌 사자 (사자)"

    print("\n[V4] 이산형 매핑 테스트 케이스 4개(모든 페르소나) 통과!")

if __name__ == "__main__":
    test_risk_profile_discrete()

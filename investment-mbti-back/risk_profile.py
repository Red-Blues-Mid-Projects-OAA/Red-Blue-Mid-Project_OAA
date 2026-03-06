"""
[Risk Profile Module]
투자자 유형(MBTI) 기반 위험 회피 계수(lambda) 산출 백엔드 모듈

[프론트엔드 연동 가이드]
1. 프론트엔드에서는 아래 12개 설문 문항에 대한 답변을 'A' 또는 'B'의 리스트(length=12)로 수집합니다.
2. 각 문항의 첫 번째 옵션은 'A', 두 번째 옵션은 'B'로 매핑합니다. (B형 = E, N, F, P 성향)
3. 손실 한도 슬라이더 값(원 단위, 예: 8000000)을 수집합니다.
4. `calculate_risk_profile(answers, loss_limit_value)` 함수를 호출하여 결과를 받습니다.

[설문 문항 매핑]
Q1~Q3: I(A) vs E(B)
Q4~Q6: S(A) vs N(B)
Q7~Q9: T(A) vs F(B)
Q10~Q12: J(A) vs P(B)
"""


def get_lambda_by_level(level):
    """
    슬라이더(1%~40%)를 10% 단위의 4개 구간으로 나누었으므로,
    각 정수 레벨(1~4)별로 해당 구간의 중앙값(Midpoint)을 대표 Loss로 수리적 산출하여
    연속적인 1.155/Loss 반비례 곡선에서 4개의 이산적(Discrete) lambda로 매핑합니다.
    - Level 4 (1~10%)  -> 대표 Loss 5% (0.05)
    - Level 3 (10~20%) -> 대표 Loss 15% (0.15)
    - Level 2 (20~30%) -> 대표 Loss 25% (0.25)
    - Level 1 (30~40%) -> 대표 Loss 35% (0.35)
    """
    rep_loss = 0.45 - (level * 0.10)
    return round(1.155 / rep_loss, 2)


def calculate_risk_profile(answers, loss_limit_value, initial_investment):
    """
    사용자의 설문 답변과 손실 한도 설정을 바탕으로 투자 MBTI 페르소나와 
    최종 위험 회피 계수(lambda_final)를 산출합니다.

    Args:
        answers (list of str): 12개 문항에 대한 답변 리스트 (['A', 'B', ...])
        loss_limit_value (int): 슬라이더에서 선택된 금액 (원 단위, 예: 9500000)
        initial_investment (int): 사용자의 초기 투자 원금 (원 단위, 예: 10000000)

    Returns:
        dict: {
            'mbti': str, (예: 'ENFP')
            'mbti_nickname': str, (예: '재기발랄한 활동가')
            'persona': str, (예: '공격적 독수리 (독수리)')
            'persona_desc': str, (페르소나 설명)
            'lambda_final': float, (최종 선택된 보수적 람사)
            'loss_ratio_percent': float
        }
    """

    # 1. MBTI 유형 판별 (B타입 = E/N/F/P)
    mbti = ""
    mbti += "E" if answers[0:3].count('B') >= 2 else "I"
    mbti += "N" if answers[3:6].count('B') >= 2 else "S"
    mbti += "F" if answers[6:9].count('B') >= 2 else "T"
    mbti += "P" if answers[9:12].count('B') >= 2 else "J"

    # 2. STEP 1-1: MBTI 등급 (level_mbti) 산출
    # 1점(가장 공격적) ~ 4점(가장 보수적)
    count_b = answers.count('B')
    if count_b >= 10:
        level_mbti = 1
    elif count_b >= 7:
        level_mbti = 2
    elif count_b >= 4:
        level_mbti = 3
    else:  # 0 ~ 3
        level_mbti = 4

    # 3. STEP 1-2: 슬라이더 등급 (level_slider) 산출
    # Loss 슬라이더 범위: 1% ~ 40% (10% 단위로 등급 부여)
    loss_percent = abs((loss_limit_value - initial_investment) / initial_investment) * 100
    
    if loss_percent > 30:
        level_slider = 1  # 30% 초과 ~ 40% 이하 (독수리 성향)
    elif loss_percent > 20:
        level_slider = 2  # 20% 초과 ~ 30% 이하 (사자 성향)
    elif loss_percent > 10:
        level_slider = 3  # 10% 초과 ~ 20% 이하 (강아지 성향)
    else:
        level_slider = 4  # 1% ~ 10% 이하 (거북이 성향)

    # 4. STEP 2: 7:3 가중 평균 및 최종 등급 확정
    # 슬라이더 70%, MBTI 30% 비중 합산
    score_final = (0.7 * level_slider) + (0.3 * level_mbti)
    # 반올림하여 1~4등급 정수로 확정
    final_level = int(score_final + 0.5)
    

    # 5. STEP 3: 최종 람다 산출 및 페르소나 매핑
    
    mapping_data = {
        4: {
            "lambda_final": get_lambda_by_level(4),
            "persona": "노예 개미",
            "desc": "시장을 단 1%라도 확실히 앞서길 원하며, 지수를 하회하는 리스크를 극도로 경계합니다."
        },
        3: {
            "lambda_final": get_lambda_by_level(3),
            "persona": "월급루팡 개미",
            "desc": "데이터의 통계적 우위를 신뢰하며, 시장 수익률을 꾸준히 상회하는 안정적인 성장을 추구합니다."
        },
        2: {
            "lambda_final": get_lambda_by_level(2),
            "persona": "파이어족 개미",
            "desc": "시장 평균 이상의 확실한 초과 수익 달성을 지향하며, 높은 수익 기회를 위해 리스크를 적극적으로 수용합니다."
        },
        1: {
            "lambda_final": get_lambda_by_level(1),
            "persona": "YOLO 개미",
            "desc": "지수 추종보다는 모델이 포착한 강력한 시그널에 집중하여, 압도적인 성과를 위해 방대한 리스크를 감수합니다."
        }
    }
    
    selected_mapping = mapping_data[final_level]

    lambda_final = selected_mapping["lambda_final"]

    return {
        "mbti": mbti,
        "persona": selected_mapping["persona"],
        "persona_desc": selected_mapping["desc"],
        "lambda_final": lambda_final,
        "loss_ratio_percent": round(loss_percent, 2),
        "final_level": final_level
    }



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

def calculate_risk_profile(answers, loss_limit_value):
    """
    사용자의 설문 답변과 손실 한도 설정을 바탕으로 투자 MBTI 페르소나와 
    최종 위험 회피 계수(lambda_final)를 산출합니다.

    Args:
        answers (list of str): 12개 문항에 대한 답변 리스트 (['A', 'B', ...])
        loss_limit_value (int): 슬라이더에서 선택된 금액 (원 단위, 예: 8000000)

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
    loss_percent = abs((loss_limit_value - 10000000) / 10000000) * 100
    
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
    # 공식: lambda = (Sharpe 0.7 * 신뢰도 1.65) / Representative Loss = 1.155 / Loss
    # 각 정수 레벨에 역산된(대표격인) Loss 값을 매핑하여 람다 산출 (5%, 6.97%, 11.5%, 33%)
    lambda_map = {
        4: round(1.155 / 0.05, 2),    # 23.10 (거북이 대표 Loss: 5%)
        3: round(1.155 / 0.0697, 2),  # 16.57 (강아지 대표 Loss: 약 7%)
        2: round(1.155 / 0.115, 2),   # 10.04 (사자 대표 Loss: 11.5%)
        1: round(1.155 / 0.33, 2)     # 3.50  (독수리 대표 Loss: 33%)
    }
    
    mapping_data = {
        4: {
            "lambda_final": lambda_map[4],
            "persona": "안전 지향 (거북이)",
            "desc": "특징: 리스크에 매우 민감하며 원금 보존을 최우선으로 합니다."
        },
        3: {
            "lambda_final": lambda_map[3],
            "persona": "신중한 탐험가 (강아지)",
            "desc": "특징: 평균적인 투자자보다 다소 보수적이며, 분석적 근거를 중시합니다."
        },
        2: {
            "lambda_final": lambda_map[2],
            "persona": "균형 잡힌 사자 (사자)",
            "desc": "특징: 수익을 위해 적정 수준의 리스크를 감내할 수 있습니다."
        },
        1: {
            "lambda_final": lambda_map[1],
            "persona": "공격적 독수리 (독수리)",
            "desc": "특징: 리스크보다는 기회와 수익에 집중하며 높은 변동성을 견딥니다."
        }
    }
    
    selected_mapping = mapping_data[final_level]

    # 6. MBTI 별칭 매핑
    mbti_desc_map = {
        "ENTJ": "대담한 통솔자", "ENTP": "뜨거운 논쟁을 즐기는 변론가",
        "ENFJ": "정의로운 사회운동가", "ENFP": "재기발랄한 활동가",
        "ESTJ": "엄격한 관리자", "ESTP": "모험을 즐기는 사업가",
        "ESFJ": "사교적인 외교관", "ESFP": "자유로운 영혼의 연예인",
        "INTJ": "용의주도한 전략가", "INTP": "논리적인 사색가",
        "INFJ": "선의의 옹호자", "INFP": "열정적인 중재자",
        "ISTJ": "청렴결백한 논리주의자", "ISTP": "만능 재주꾼",
        "ISFJ": "용감한 수호자", "ISFP": "호기심 많은 예술가"
    }
    
    lambda_final = selected_mapping["lambda_final"]

    return {
        "mbti": mbti,
        "mbti_nickname": mbti_desc_map.get(mbti, ""),
        "persona": selected_mapping["persona"],
        "persona_desc": selected_mapping["desc"],
        "lambda_final": lambda_final,
        "loss_ratio_percent": round(loss_percent, 2)
    }



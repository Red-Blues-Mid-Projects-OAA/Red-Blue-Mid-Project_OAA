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
            'lambda_mbti': float, (설문 기반 임시 람다)
            'lambda_slider': float, (슬라이더 기반 임시 람다)
            'lambda_final': float, (최종 선택된 보수적 람다)
            'loss_ratio_percent': float
        }
    """
    if len(answers) != 12:
        raise ValueError("12개의 답변이 필요합니다.")

    # 1. MBTI 유형 판별 (B타입 = E/N/F/P)
    mbti = ""
    mbti += "E" if answers[0:3].count('B') >= 2 else "I"
    mbti += "N" if answers[3:6].count('B') >= 2 else "S"
    mbti += "F" if answers[6:9].count('B') >= 2 else "T"
    mbti += "P" if answers[9:12].count('B') >= 2 else "J"

    # 2. 투자자 유형 구분 및 임시 람다(lambda_mbti) 산출 (2.0 ~ 13.2 범위)
    count_b = answers.count('B')

    if count_b <= 3:
        persona = "안전 지향 (거북이)"
        lambda_mbti = 13.2
        desc = "특징: 리스크에 매우 민감하며 원금 보존을 최우선으로 합니다."
    elif count_b <= 6:
        persona = "신중한 탐험가 (강아지)"
        lambda_mbti = 9.47
        desc = "특징: 평균적인 투자자보다 다소 보수적이며, 분석적 근거를 중시합니다."
    elif count_b <= 9:
        persona = "균형 잡힌 사자 (사자)"
        lambda_mbti = 5.73
        desc = "특징: 수익을 위해 적정 수준의 리스크를 감내할 수 있습니다."
    else:
        persona = "공격적 독수리 (독수리)"
        lambda_mbti = 2.0
        desc = "특징: 리스크보다는 기회와 수익에 집중하며 높은 변동성을 견딥니다."

    # 3. 슬라이더 기반 기초 람다(lambda_slider) 산출 (0.66 / |L|)
    loss_ratio = abs((loss_limit_value - 10000000) / 10000000)
    if loss_ratio <= 0.01: # 0 또는 과도하게 작은 값 방어
        loss_ratio = 0.05
    
    lambda_slider = round(0.66 / loss_ratio, 4)

    # 4. 최종 람다(lambda_final) 산출: 허세 방지 필터링 (Conservative fallback)
    lambda_final = max(lambda_mbti, lambda_slider)

    # 5. MBTI 별칭 매핑
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
    
    return {
        "mbti": mbti,
        "mbti_nickname": mbti_desc_map.get(mbti, ""),
        "persona": persona,
        "persona_desc": desc,
        "lambda_mbti": lambda_mbti,
        "lambda_slider": lambda_slider,
        "lambda_final": round(lambda_final, 4),
        "loss_ratio_percent": round(loss_ratio * 100, 2)
    }


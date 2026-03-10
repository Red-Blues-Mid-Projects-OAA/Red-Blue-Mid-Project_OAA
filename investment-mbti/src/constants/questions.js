/*
 * 이 파일은 설문 화면에서 보여 줄 질문과 선택지를 한곳에 모아 둔 상수 파일입니다.
 * 설정값과 고정 데이터가 한곳에 모여 있어 개발 서버, API 주소, 린트 규칙, 화면 공통 상수를 바꾸더라도 호출부 수정 범위를 최소화할 수 있습니다.
 */

const imagePath = (fileName) => `${import.meta.env.BASE_URL}images/${fileName}`;

export const QUESTIONS = [
  {
    id: 1,
    type: 'E_I',
    question: 'Q1. 자고 일어났더니 오픈채팅방이 "A종목 상한가 간다!"로 도배됐다면?',
    image: imagePath('Q1.png'),
    optionA: {
      text: '채팅방 알림을 끄고 뒤돌아 내 공부를 시작한다.',
      value: 'I',
    },
    optionB: {
      text: '바로 나무위키랑 종목토론방 들어가서 민심을 확인한다.',
      value: 'E',
    },
  },
  {
    id: 2,
    type: 'E_I',
    question: 'Q2. 내가 산 주식이 연일 뉴스에 나오며 전 국민의 관심을 받는다면?',
    image: imagePath('Q2.png'),
    optionA: {
      text: '과열된 관심은 버블일 것 같아 겁이 나고 불안해서 팔고 싶다.',
      value: 'I',
    },
    optionB: {
      text: '세상이 내 안목을 알아주는 것 같아 짜릿하고 수익률 인증샷을 찍고 싶다.',
      value: 'E',
    },
  },
  {
    id: 3,
    type: 'E_I',
    question: 'Q3. 투자로 크게 한탕 하거나 대차게 말아먹었을 때 당신은?',
    image: imagePath('Q3.png'),
    optionA: {
      text: '침대에 누워 천장을 보며 만세를 외치거나 혼자 고뇌한다.',
      value: 'I',
    },
    optionB: {
      text: '당장 단톡방에 "오늘 소고기 쏜다"를 외치거나 "살려줘"를 외친다.',
      value: 'E',
    },
  },
  {
    id: 4,
    type: 'S_N',
    question: 'Q4. 최신 스마트폰을 살 때 당신이 가장 먼저 하는 행동은?',
    image: imagePath('Q4.png'),
    optionA: {
      text: '스펙과 가격 등 숫자와 팩트가 적힌 표를 꼼꼼하게 본다.',
      value: 'S',
    },
    optionB: {
      text: '브랜드가 가진 이미지와 감성을 마음으로 느낀다.',
      value: 'N',
    },
  },
  {
    id: 5,
    type: 'S_N',
    question: 'Q5. 기업 분석을 할 때 당신의 심장을 뛰게 하는 것은?',
    image: imagePath('Q5.png'),
    optionA: {
      text: '정확하게 우상향하는 재무제표의 아름다운 숫자들.',
      value: 'S',
    },
    optionB: {
      text: '인류의 역사를 바꿀 신기술과 CEO의 미친 비전이 담긴 인터뷰.',
      value: 'N',
    },
  },
  {
    id: 6,
    type: 'S_N',
    question: 'Q6. "인생 한 방"을 노린다면 당신의 선택은?',
    image: imagePath('Q6.png'),
    optionA: {
      text: '월급처럼 꼬박꼬박 지급되는 배당주.',
      value: 'S',
    },
    optionB: {
      text: '지금은 적자지만 나중에 100배 갈 수도 있는 유망주.',
      value: 'N',
    },
  },
  {
    id: 7,
    type: 'T_F',
    question: 'Q7. 오래 보유했던 주식이 어느 날 갑자기 떨어진다면?',
    image: imagePath('Q7.png'),
    optionA: {
      text: '칼같이 매도 버튼을 누르고 로그아웃한다.',
      value: 'T',
    },
    optionB: {
      text: '언젠간 오르겠지라고 생각하며 눈물로 기도한다.',
      value: 'F',
    },
  },
  {
    id: 8,
    type: 'T_F',
    question: 'Q8. 단짝 친구가 "이거 진짜 나만 아는 정보야"라며 종목을 추천한다면?',
    image: imagePath('Q8.png'),
    optionA: {
      text: '친구의 말이라도 논리가 없으면 얄짤없이 필터링한다.',
      value: 'T',
    },
    optionB: {
      text: '나를 생각해 준 친구의 정성을 봐서라도 일단 매수 버튼에 손이 간다.',
      value: 'F',
    },
  },
  {
    id: 9,
    type: 'T_F',
    question: 'Q9. 당신이 생각하는 최고의 투자 고수는 누구?',
    image: imagePath('Q9.png'),
    optionA: {
      text: '감정에 휘둘리지 않고 통계와 확률로 시장을 이겨버리는 \'인간 엑셀\'.',
      value: 'T',
    },
    optionB: {
      text: '시장의 보이지 않는 공포와 환희를 본능적으로 읽어내는 \'직감의 마법사\'.',
      value: 'F',
    },
  },
  {
    id: 10,
    type: 'J_P',
    question: 'Q10. 여행 가기 전, 당신의 구글 맵(지도) 상태는?',
    image: imagePath('Q10.png'),
    optionA: {
      text: '시간대별 이동 동선과 맛집 리스트 등 시나리오가 완벽하다.',
      value: 'J',
    },
    optionB: {
      text: '항공권과 숙소만 있으면 충분! 나머지는 현지 분위기 따라 정한다.',
      value: 'P',
    },
  },
  {
    id: 11,
    type: 'J_P',
    question: 'Q11. 내 주식 계좌를 점검하는 스타일은?',
    image: imagePath('Q11.png'),
    optionA: {
      text: '정해진 날에 주기적으로 정리한다.',
      value: 'J',
    },
    optionB: {
      text: '잊고 지내다 생각날 때 유연하게 대응한다.',
      value: 'P',
    },
  },
  {
    id: 12,
    type: 'J_P',
    question: 'Q12. 약속 장소로 가던 중 갑자기 지하철이 고장 나 멈춰버렸다면?',
    image: imagePath('Q12.png'),
    optionA: {
      text: '즉시 다른 경로를 검색하고 지인에게 정확히 몇 분 늦을지 통보한다.',
      value: 'J',
    },
    optionB: {
      text: '일단 상황을 지켜보며 기다리는 동안 스마트폰으로 내 할 일을 한다.',
      value: 'P',
    },
  }
];

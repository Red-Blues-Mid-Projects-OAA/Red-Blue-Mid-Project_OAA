import React, { useState } from 'react';
import Intro from './components/Intro';
import Question from './components/Question';
import LossSlider from './components/LossSlider';
import Loading from './components/Loading';
import Result from './components/Result';
import { QUESTIONS } from './constants/questions';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('INTRO'); // INTRO, QUESTION, SLIDER, LOADING, RESULT
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [lossLimit, setLossLimit] = useState(null);
  const [resultData, setResultData] = useState(null);

  const handleStart = () => {
    setCurrentView('QUESTION');
  };

  const handleAnswer = (answerData) => {
    const newAnswers = [...answers, answerData];
    setAnswers(newAnswers);

    if (currentQuestionIndex < QUESTIONS.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
    } else {
      setCurrentView('SLIDER');
    }
  };

  const handleSliderComplete = (value) => {
    setLossLimit(value);
    setCurrentView('LOADING');
    submitToMockAPI(answers, value);
  };

  const submitToMockAPI = (mbtiAnswers, sliderValue) => {
    // 백엔드 API 연동을 모방하는 Mock API
    console.log('--- 전송할 데이터 ---');
    console.log('MBTI Answers:', mbtiAnswers);
    console.log('손실 한도(%):', sliderValue);

    setTimeout(() => {
      // Mock Data 생성 로직 (단순 시뮬레이션)
      const isConservative = sliderValue >= -15; // 손실을 적게 감내 (숫자가 클수록 안전형)

      const mockResult = {
        title: isConservative ? '안전제일 금고지기 거북이 🐢' : '강심장 불나방 야수 🦁',
        description: isConservative
          ? '돌다리도 천 번 두드려보고 건너는 철저한 안정 추구형입니다.'
          : '위험을 두려워하지 않고 기꺼이 감수하며 큰 보상을 노리는 공격적인 투자자입니다.',
        features: isConservative
          ? [
            '원금 손실을 극도로 꺼려 안정적인 자산을 선호합니다.',
            '천천히, 하지만 확실하게 불려나가는 복리의 마법을 믿습니다.',
            '장기 국채와 예적금 비중이 높습니다.'
          ]
          : [
            '하이 리스크, 하이 리턴! 변동성을 즐깁니다.',
            '단기 모멘텀 및 성장주를 선호합니다.',
            '위기가 오면 오히려 저점 매수의 기회라 생각합니다.'
          ],
        recommendedStocks: isConservative
          ? [
            { rank: 1, ticker: 'JNJ', name: '존슨앤드존슨', volatility: 'Low', color: '#10b981' },
            { rank: 2, ticker: 'PG', name: '프로스터 앤 갬블', volatility: 'Low', color: '#10b981' },
            { rank: 3, ticker: 'KO', name: '코카콜라', volatility: 'Low', color: '#10b981' },
            { rank: 4, ticker: 'PEP', name: '펩시코', volatility: 'Low', color: '#10b981' },
            { rank: 5, ticker: 'WM', name: '웨이스트 매니지먼트', volatility: 'Low', color: '#10b981' },
            { rank: 6, ticker: 'COST', name: '코스트코', volatility: 'Low', color: '#10b981' },
            { rank: 7, ticker: 'V', name: '비자', volatility: 'Low', color: '#3b82f6' },
            { rank: 8, ticker: 'MRK', name: '머크', volatility: 'Low', color: '#3b82f6' },
            { rank: 9, ticker: 'UNH', name: '유나이티드헬스', volatility: 'Low', color: '#3b82f6' },
            { rank: 10, ticker: 'BRK.B', name: '버크셔 해서웨이', volatility: 'Low', color: '#3b82f6' }
          ]
          : [
            { rank: 1, ticker: 'MSTR', name: '마이크로스트레티지', volatility: 'High', color: '#ef4444' },
            { rank: 2, ticker: 'COIN', name: '코인베이스', volatility: 'High', color: '#ef4444' },
            { rank: 3, ticker: 'SMCI', name: '슈퍼마이크로', volatility: 'High', color: '#ef4444' },
            { rank: 4, ticker: 'TSLA', name: '테슬라', volatility: 'High', color: '#ef4444' },
            { rank: 5, ticker: 'PLTR', name: '팔란티어', volatility: 'High', color: '#f59e0b' },
            { rank: 6, ticker: 'NVDA', name: '엔비디아', volatility: 'High', color: '#f59e0b' },
            { rank: 7, ticker: 'ARM', name: 'ARM 홀딩스', volatility: 'High', color: '#f59e0b' },
            { rank: 8, ticker: 'HOOD', name: '로빈후드', volatility: 'High', color: '#f59e0b' },
            { rank: 9, ticker: 'ROKU', name: '로쿠', volatility: 'High', color: '#f59e0b' },
            { rank: 10, ticker: 'AMD', name: 'AMD', volatility: 'High', color: '#f59e0b' }
          ]
      };

      setResultData(mockResult);
      setCurrentView('RESULT');
    }, 2500); // 2.5초 지연
  };

  const handleRestart = () => {
    setAnswers([]);
    setCurrentQuestionIndex(0);
    setLossLimit(null);
    setResultData(null);
    setCurrentView('INTRO');
  };

  return (
    <div className="app-container">
      {currentView === 'INTRO' && <Intro onStart={handleStart} />}

      {currentView === 'QUESTION' && (
        <Question
          data={QUESTIONS[currentQuestionIndex]}
          step={currentQuestionIndex + 1}
          totalSteps={QUESTIONS.length}
          onAnswer={handleAnswer}
        />
      )}

      {currentView === 'SLIDER' && <LossSlider onComplete={handleSliderComplete} />}

      {currentView === 'LOADING' && <Loading />}

      {currentView === 'RESULT' && <Result personaData={resultData} onRestart={handleRestart} />}
    </div>
  );
}

export default App;

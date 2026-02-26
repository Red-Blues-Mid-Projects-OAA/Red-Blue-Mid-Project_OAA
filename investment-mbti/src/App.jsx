import React, { useState } from 'react';
import Intro from './components/Intro';
import Question from './components/Question';
import LossSlider from './components/LossSlider';
import Loading from './components/Loading';
import DashboardResult from './components/DashboardResult';
import { QUESTIONS } from './constants/questions';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('INTRO'); // INTRO, QUESTION, SLIDER, LOADING, RESULT, RECOMMENDATION
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

  // 새로운 UX 요구사항: 뒤로 가기
  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1);
      // 가장 최근에 고른 답변을 제거
      setAnswers(prev => prev.slice(0, -1));
    }
  };

  const handleSliderComplete = (value) => {
    setLossLimit(value);
    setCurrentView('LOADING');
    handleSubmit(answers, value);
  };

  const handleSubmit = async (mbtiAnswers, sliderValue) => {
    console.log('--- 백엔드로 전송할 데이터 ---');
    console.log('MBTI Answers:', mbtiAnswers);
    console.log('손실 한도(%):', sliderValue);

    try {
      // 1. 프론트엔드 포맷팅: answers (객체 -> 문자열 A/B), lossLimit (퍼센트 -> 원화 계산)
      const payloadAnswers = mbtiAnswers.map(a => a.optionKey);
      const krwLossLimit = 100000 * (100 + Number(sliderValue)); // -5% -> 9500000

      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          answers: payloadAnswers,
          loss_limit_value: krwLossLimit
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      const responseJson = await response.json();
      const responseData = responseJson.data;

      console.log('백엔드 응답:', responseData);

      // 2. 백엔드에서 반환하는 JSON 형태에 맞춰 resultData 구성
      const finalResult = {
        title: responseData.persona,
        description: responseData.description,
        features: responseData.features,
        mbti: responseData.mbti,
        mbtiNickname: responseData.mbti_nickname,
        recommendedStocks: responseData.recommended_stocks,
        portfolioAnalysis: responseData.portfolio_analysis,
        chartData: responseData.chart_data,
        forecastData: responseData.forecast_data,
        rawAnswers: responseData.raw_answers,
        finalLevel: responseData.final_level,
        lambdaFinal: responseData.lambda_final,
      };

      // 시각적 효과를 위한 약간의 지연 후 결과 렌더링
      setTimeout(() => {
        setResultData(finalResult);
        setCurrentView('RESULT');
      }, 1500);

    } catch (error) {
      console.error('API Error:', error);
      alert('데이터 전송에 실패했습니다. 백엔드 서버가 켜져 있는지 확인해주세요.');
      setCurrentView('INTRO');
    }
  };

  const handleRestart = () => {
    setAnswers([]);
    setCurrentQuestionIndex(0);
    setLossLimit(null);
    setResultData(null);
    setCurrentView('INTRO');
  };

  const handleShowRecommendation = () => {
    setCurrentView('RECOMMENDATION');
  };

  return (
    // 전체 컨테이너 및 뷰포트 확장 반응형 대응
    <div className="app-container max-w-[1920px] mx-auto px-4 min-h-screen py-8">
      {currentView === 'INTRO' && <Intro onStart={handleStart} />}

      {currentView === 'QUESTION' && (
        <Question
          data={QUESTIONS[currentQuestionIndex]}
          step={currentQuestionIndex + 1}
          totalSteps={QUESTIONS.length}
          onAnswer={handleAnswer}
          onBack={handleBack}
        />
      )}

      {currentView === 'SLIDER' && <LossSlider onComplete={handleSliderComplete} />}

      {currentView === 'LOADING' && <Loading />}

      {currentView === 'RESULT' && <DashboardResult personaData={resultData} onRestart={handleRestart} />}
    </div>
  );
}

export default App;

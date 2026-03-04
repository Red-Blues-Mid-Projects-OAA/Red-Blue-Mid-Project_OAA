import React, { useState } from 'react';
import Intro from './components/Intro';
import InvestmentAmount from './components/InvestmentAmount';
import Question from './components/Question';
import LossSlider from './components/LossSlider';
import Loading from './components/Loading';
import DashboardResult from './components/DashboardResult';
import PortfolioSelection from './components/PortfolioSelection';
import { QUESTIONS } from './constants/questions';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('INTRO'); // INTRO, INVESTMENT_AMOUNT, QUESTION, SLIDER, LOADING, SELECTION, OPTIMIZING, RESULT
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [answers, setAnswers] = useState([]);
  const [investmentAmount, setInvestmentAmount] = useState(null); // 만원 단위
  const [lossLimit, setLossLimit] = useState(null);
  const [resultData, setResultData] = useState(null);
  const [optimizedData, setOptimizedData] = useState(null);
  const [savedSelectedTickers, setSavedSelectedTickers] = useState(null); // 사용자가 선택한 종목 보존

  const handleStart = () => {
    setCurrentView('INVESTMENT_AMOUNT');
  };

  // 투자금 입력 완료 → 질문 시작
  const handleInvestmentComplete = (amount) => {
    setInvestmentAmount(amount);
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

  // 질문 화면 뒤로가기 (Q1에서는 투자금 입력 화면으로)
  const handleBack = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(prev => prev - 1);
      setAnswers(prev => prev.slice(0, -1));
    } else {
      setCurrentView('INVESTMENT_AMOUNT');
    }
  };

  // 슬라이더 화면 뒤로가기 → 마지막 질문으로 복귀
  const handleBackFromSlider = () => {
    setCurrentQuestionIndex(QUESTIONS.length - 1);
    setAnswers(prev => prev.slice(0, -1));
    setCurrentView('QUESTION');
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
      // 프론트엔드 포맷팅: answers (객체 → 문자열 A/B), lossLimit (퍼센트 → 원화 계산)
      const payloadAnswers = mbtiAnswers.map(a => a.optionKey);
      const baseMoney = (investmentAmount || 1000) * 10000;
      const krwLossLimit = baseMoney * (1 + Number(sliderValue) / 100); // sliderValue는 음수

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

      const finalResult = {
        title: responseData.persona,
        description: responseData.description,
        features: responseData.features,
        mbti: responseData.mbti,
        mbtiNickname: responseData.mbti_nickname,
        recommendedStocks: responseData.recommended_stocks,
        allMatchingStocks: responseData.all_matching_stocks,
        portfolioAnalysis: responseData.portfolio_analysis,
        chartData: responseData.chart_data,
        forecastData: responseData.forecast_data,
        rawAnswers: responseData.raw_answers,
        finalLevel: responseData.final_level,
        lambdaFinal: responseData.lambda_final,
      };

      setTimeout(() => {
        setResultData(finalResult);
        setCurrentView('SELECTION');
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
    setInvestmentAmount(null);
    setLossLimit(null);
    setResultData(null);
    setOptimizedData(null);
    setSavedSelectedTickers(null);
    setCurrentView('INTRO');
  };

  const handleBackFromResult = () => {
    setCurrentView('SELECTION');
  };

  // 종목 선택 완료 → /api/optimize-final 호출 → RESULT
  const handleConfirmSelection = async (selectedTickers) => {
    setCurrentView('OPTIMIZING');
    try {
      const response = await fetch('http://localhost:8000/api/optimize-final', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_tickers: selectedTickers,
          lambda_final: resultData.lambdaFinal,
          final_level: resultData.finalLevel,
        }),
      });
      if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
      const json = await response.json();
      setOptimizedData(json.data);
      setCurrentView('RESULT');
    } catch (error) {
      console.error('Optimize-final API Error:', error);
      alert('포트폴리오 최적화에 실패했습니다.');
      setCurrentView('SELECTION');
    }
  };


  return (
    <div className="app-container max-w-[1920px] mx-auto px-4 min-h-screen py-8">
      {currentView === 'INTRO' && <Intro onStart={handleStart} />}

      {currentView === 'INVESTMENT_AMOUNT' && (
        <InvestmentAmount
          onComplete={handleInvestmentComplete}
          onBack={() => setCurrentView('INTRO')}
        />
      )}

      {currentView === 'QUESTION' && (
        <Question
          data={QUESTIONS[currentQuestionIndex]}
          step={currentQuestionIndex + 1}
          totalSteps={QUESTIONS.length}
          onAnswer={handleAnswer}
          onBack={handleBack}
        />
      )}

      {currentView === 'SLIDER' && (
        <LossSlider
          onComplete={handleSliderComplete}
          onBack={handleBackFromSlider}
          investmentAmount={investmentAmount}
        />
      )}

      {currentView === 'LOADING' && <Loading />}
      {currentView === 'OPTIMIZING' && <Loading />}

      {currentView === 'SELECTION' && resultData && (
        <PortfolioSelection
          recommendedStocks={resultData.recommendedStocks || []}
          finalLevel={resultData.finalLevel}
          lambdaFinal={resultData.lambdaFinal}
          savedSelectedTickers={savedSelectedTickers}
          onSelectedTickersChange={setSavedSelectedTickers}
          onBack={handleRestart}
          onRestart={handleRestart}
          onConfirm={handleConfirmSelection}
        />
      )}

      {currentView === 'RESULT' && (
        <DashboardResult
          personaData={resultData}
          optimizedData={optimizedData}
          investmentAmount={investmentAmount}
          onRestart={handleRestart}
          onBack={handleBackFromResult}
        />
      )}
    </div>
  );
}

export default App;

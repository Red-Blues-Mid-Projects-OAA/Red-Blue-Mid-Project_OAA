/*
 * 이 파일은 투자 MBTI 프론트엔드의 최상위 컴포넌트입니다. 화면 전환과 API 호출 흐름을 전체적으로 관리합니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React, { useState } from 'react';
import Intro from './components/Intro';
import InvestmentAmount from './components/InvestmentAmount';
import Question from './components/Question';
import LossSlider from './components/LossSlider';
import Loading from './components/Loading';
import DashboardResult from './components/DashboardResult';
import PortfolioSelection from './components/PortfolioSelection';
import { QUESTIONS } from './constants/questions';
import { buildApiUrl, hasApiBase, API_BASE, getApiConfigErrorMessage } from './config/api';
import './App.css';

/**
 * 앱 전체 화면 흐름을 조정하는 최상위 컴포넌트입니다.
 */

function App() {
  console.log('App Initialized. API URL:', import.meta.env.VITE_API_URL);
  // currentView는 전체 단일 페이지 앱에서 어떤 화면을 보여 줄지 결정하는 중앙 상태입니다.
  const [currentView, setCurrentView] = useState('INTRO'); // INTRO, INVESTMENT_AMOUNT, QUESTION, SLIDER, LOADING, SELECTION, OPTIMIZING, RESULT
  // currentQuestionIndex는 QUESTIONS 배열 중 현재 몇 번째 문항을 노출하는지 나타냅니다.
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  // answers는 각 문항에서 선택한 옵션 객체를 순서대로 누적한 배열입니다.
  const [answers, setAnswers] = useState([]);
  // investmentAmount는 사용자가 입력한 투자금을 "만원" 단위 그대로 보관합니다.
  const [investmentAmount, setInvestmentAmount] = useState(null);
  // lossLimit은 슬라이더에서 선택한 손실 한도 퍼센트 값입니다.
  const [lossLimit, setLossLimit] = useState(null);
  // resultData는 /api/analyze 응답을 프론트엔드 표시 구조에 맞게 재가공한 1차 결과입니다.
  const [resultData, setResultData] = useState(null);
  // optimizedData는 사용자가 종목을 고른 뒤 /api/optimize-final이 반환한 최종 최적화 결과입니다.
  const [optimizedData, setOptimizedData] = useState(null);
  // savedSelectedTickers는 결과 화면에서 뒤로 갔을 때 선택 상태를 복원하기 위한 임시 저장소입니다.
  const [savedSelectedTickers, setSavedSelectedTickers] = useState(null);

  // 사용자 입력, 버튼 클릭, API 요청을 처리하는 함수들입니다.
  const handleStart = () => {
    setCurrentView('INVESTMENT_AMOUNT');
  };

  // 투자금 입력 완료 → 질문 시작
  const handleInvestmentComplete = (amount) => {
    setInvestmentAmount(amount);
    setCurrentView('QUESTION');
  };

  const handleAnswer = (answerData) => {
    // 새 응답을 기존 answers 뒤에 붙여 다음 질문 또는 슬라이더 단계로 이동합니다.
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
    // 슬라이더 값은 별도 상태로 보존하고, 동시에 분석 API 제출 단계로 진입합니다.
    setLossLimit(value);
    setCurrentView('LOADING');
    handleSubmit(answers, value);
  };

  const handleSubmit = async (mbtiAnswers, sliderValue) => {
    if (!hasApiBase) {
      alert(getApiConfigErrorMessage());
      setCurrentView('INTRO');
      return;
    }

    console.log('--- 백엔드로 전송할 데이터 ---');
    console.log('MBTI Answers:', mbtiAnswers);
    console.log('손실 한도(%):', sliderValue);

    try {
      // 프론트엔드 포맷팅: answers (객체 → 문자열 A/B), lossLimit (퍼센트 → 원화 계산)
      // payloadAnswers는 백엔드가 기대하는 단순 배열 형태의 설문 응답입니다.
      const payloadAnswers = mbtiAnswers.map(a => a.optionKey);
      // baseMoney는 사용자가 입력한 만원 단위를 실제 원화로 환산한 기준 투자금입니다.
      const baseMoney = (investmentAmount || 1000) * 10000;
      // krwLossLimit는 손실 한도 퍼센트를 적용한 최소 허용 잔액 기준 금액입니다.
      const krwLossLimit = baseMoney * (1 + Number(sliderValue) / 100); // sliderValue는 음수

      const response = await fetch(buildApiUrl('/api/analyze'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Bypass-Tunnel-Reminder': 'true'
        },
        body: JSON.stringify({
          answers: payloadAnswers,
          loss_limit_value: krwLossLimit,
          initial_investment: baseMoney
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      // responseJson은 API 공통 envelope, responseData는 실제 결과 payload입니다.
      const responseJson = await response.json();
      const responseData = responseJson.data;

      console.log('백엔드 응답:', responseData);

      // finalResult는 하위 화면이 기대하는 camelCase 구조로 다시 조립한 프론트엔드 전용 객체입니다.
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

      // 로딩 애니메이션이 너무 짧게 끝나지 않도록 약간의 지연 후 선택 화면으로 이동합니다.
      setTimeout(() => {
        setResultData(finalResult);
        setCurrentView('SELECTION');
      }, 1500);

    } catch (error) {
      console.error('API Error:', error, 'API_BASE:', API_BASE);
      alert('데이터 전송에 실패했습니다. 백엔드 서버가 켜져 있는지 확인해주세요.');
      setCurrentView('INTRO');
    }
  };

  const handleRestart = () => {
    // 처음부터 다시 시작할 수 있도록 모든 입력/응답/최적화 상태를 초기화합니다.
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
    // 종목 선택 확정 직후에는 결과 화면 대신 최적화 전용 로딩 화면을 먼저 보여 줍니다.
    setCurrentView('OPTIMIZING');
    if (!hasApiBase) {
      alert(getApiConfigErrorMessage());
      setCurrentView('SELECTION');
      return;
    }
    try {
      const response = await fetch(buildApiUrl('/api/optimize-final'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Bypass-Tunnel-Reminder': 'true'
        },
        body: JSON.stringify({
          // selected_tickers는 사용자가 고른 최종 종목 집합입니다.
          selected_tickers: selectedTickers,
          // lambda_final과 final_level은 1차 분석 단계에서 계산된 성향 파라미터를 그대로 전달합니다.
          lambda_final: resultData.lambdaFinal,
          final_level: resultData.finalLevel,
        }),
      });
      if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
      const json = await response.json();
      setOptimizedData(json.data);
      setCurrentView('RESULT');
    } catch (error) {
      console.error('Optimize-final API Error:', error, 'API_BASE:', API_BASE);
      alert('포트폴리오 최적화에 실패했습니다.');
      setCurrentView('SELECTION');
    }
  };

  // 일부 초기 화면은 1스크린 레이아웃을 유지하고, 결과 화면은 세로 스크롤을 허용합니다.
  const isViewportLockedView =
    currentView === 'INTRO' ||
    currentView === 'INVESTMENT_AMOUNT' ||
    currentView === 'QUESTION' ||
    currentView === 'SLIDER';

  // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
  return (
    <div className={`app-container max-w-[1920px] mx-auto ${isViewportLockedView ? 'app-container--fit-screen' : ''}`}>
      {/* 진입부: 소개 화면에서 검사 시작 버튼만 노출합니다. */}
      {currentView === 'INTRO' && <Intro onStart={handleStart} />}

      {/* 금액 입력부: 투자금이 확정되면 질문 단계로 넘어갑니다. */}
      {currentView === 'INVESTMENT_AMOUNT' && (
        <InvestmentAmount
          onComplete={handleInvestmentComplete}
          onBack={() => setCurrentView('INTRO')}
        />
      )}

      {/* 설문부: QUESTIONS 배열을 한 문제씩 순차적으로 렌더링합니다. */}
      {currentView === 'QUESTION' && (
        <Question
          data={QUESTIONS[currentQuestionIndex]}
          step={currentQuestionIndex + 1}
          totalSteps={QUESTIONS.length}
          onAnswer={handleAnswer}
          onBack={handleBack}
        />
      )}

      {/* 손실 허용 한도 선택부: 마지막 설문 후 위험 허용 범위를 수치로 받습니다. */}
      {currentView === 'SLIDER' && (
        <LossSlider
          onComplete={handleSliderComplete}
          onBack={handleBackFromSlider}
          investmentAmount={investmentAmount}
        />
      )}

      {/* 분석/최적화 단계는 같은 Loading 컴포넌트를 재사용하되, currentView로 문맥만 구분합니다. */}
      {currentView === 'LOADING' && <Loading />}
      {currentView === 'OPTIMIZING' && <Loading />}

      {/* 종목 선택부: 1차 분석 결과를 바탕으로 사용자가 최종 포트폴리오 후보를 고릅니다. */}
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

      {/* 결과부: 최적화가 끝난 뒤 최종 대시보드와 종목 카드, 차트를 렌더링합니다. */}
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

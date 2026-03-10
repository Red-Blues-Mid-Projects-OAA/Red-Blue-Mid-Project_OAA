/*
 * 이 파일은 대시보드 결과 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React, { useState } from 'react';
import { RefreshCw, ArrowLeft, TrendingUp, Sparkles, ShieldAlert } from 'lucide-react';
import MbtiBarChart from './MbtiBarChart';
import PortfolioPieChart from './PortfolioPieChart';
import CumulativeReturnChart from './CumulativeReturnChart';
import StockCard from './StockCard';
import './DashboardResult.css';
import './PortfolioSelection.css';

const PERSONA_IMAGE_BY_LEVEL = {
    4: 'slave',
    3: 'worker',
    2: 'fire',
    1: 'yolo',
};

// 각 MBTI 축이 어떤 라벨과 색상으로 그려질지 정의하는 화면 전용 메타데이터입니다.
const BAR_CONFIG = [
    { key: 'energy', label: '시장 반응', left: '외향형', right: '내향형', gradient: 'linear-gradient(90deg, #a7f3d0 0%, #3b82f6 100%)' },
    { key: 'insight', label: '가치 판단', left: '감각형', right: '직관형', gradient: 'linear-gradient(90deg, #a7f3d0 0%, #3b82f6 100%)' },
    { key: 'logic', label: '의사 결정', left: '사고형', right: '감정형', gradient: 'linear-gradient(90deg, #a7f3d0 0%, #3b82f6 100%)' },
    { key: 'style', label: '대응 방식', left: '계획형', right: '유연형', gradient: 'linear-gradient(90deg, #a7f3d0 0%, #3b82f6 100%)' },
];

/**
 * 입력값을 안전한 숫자로 바꿔 계산에 사용할 수 있게 합니다.
 */
function toFiniteNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

/**
 * 퍼센트 수치를 보기 쉬운 문자열로 포맷합니다.
 */
function formatPercent(value, digits = 2, forceSign = true) {
    if (value == null || Number.isNaN(Number(value))) {
        return '-';
    }
    const n = Number(value);
    const sign = forceSign ? (n >= 0 ? '+' : '') : '';
    return `${sign}${n.toFixed(digits)}%`;
}

// 퍼센트 수치를 투자금 기준 원화 문자열로 변환
function formatKRW(pct, manwon, showSign = true) {
    // 퍼센트 수익률/손실률을 "만원" 단위 투자금 기준 실제 원화 증감액으로 환산합니다.
    const won = (pct / 100) * manwon * 10000;
    // 표시 단위만 바뀌어도 부호 정보는 유지해야 하므로 절대값과 부호를 분리합니다.
    const absWon = Math.abs(won);
    const sign = won >= 0 ? (showSign ? '+' : '') : '-';

    if (absWon >= 100000000) {
        return `${sign}${(absWon / 100000000).toFixed(2)}억`;
    } else if (absWon >= 10000) {
        return `${sign}${Math.round(absWon / 10000).toLocaleString()}만`;
    }
    return `${sign}${Math.round(absWon).toLocaleString()}원`;
}

/**
 * 투자 성향 제목이나 단계에 맞는 캐릭터 이미지를 고릅니다.
 */
function getPersonaImage(title, level) {
    if (PERSONA_IMAGE_BY_LEVEL[level]) {
        return PERSONA_IMAGE_BY_LEVEL[level];
    }
    const safeTitle = String(title || '');
    if (safeTitle.includes('노예')) return 'slave';
    if (safeTitle.includes('월급루팡')) return 'worker';
    if (safeTitle.includes('파이어')) return 'fire';
    return 'yolo';
}

/**
 * 원본 답변 12개를 MBTI 4축 점수로 환산합니다.
 */
function getMbtiScores(rawAnswers = []) {
    if (rawAnswers.length < 12) {
        return {
            energy: 50,
            insight: 50,
            logic: 50,
            style: 50,
        };
    }

    // answerAt은 빈 값 방어와 대소문자 정규화를 함께 처리하는 축약 헬퍼입니다.
    const answerAt = (idx) => String(rawAnswers[idx] || '').toUpperCase();
    const countRange = (start, end, expected) => {
        let count = 0;
        for (let i = start; i < end; i += 1) {
            if (answerAt(i) === expected) count += 1;
        }
        return count;
    };

    return {
        energy: (countRange(0, 3, 'A') / 3) * 100,
        insight: (countRange(3, 6, 'B') / 3) * 100,
        logic: (countRange(6, 9, 'B') / 3) * 100,
        style: (countRange(9, 12, 'B') / 3) * 100,
    };
}

/**
 * 종목 비중을 반영해 기간별 과거 수익률의 가중평균을 계산합니다.
 */
function getWeightedHistoricalReturn(stocks, key = '3M') {
    if (!stocks.length) return 0;

    const normalized = stocks.map((stock) => {
        // histVal은 기간별 과거 수익률, weight는 카드에 표시되는 실제 비중입니다.
        const histVal = toFiniteNumber(stock?.historical_returns?.[key], 0);
        const weight = Math.max(toFiniteNumber(stock?.weight, 0), 0);
        return { histVal, weight };
    });

    const totalWeight = normalized.reduce((sum, row) => sum + row.weight, 0);
    if (totalWeight > 0) {
        const weightedSum = normalized.reduce((sum, row) => sum + (row.histVal * row.weight), 0);
        return weightedSum / totalWeight;
    }

    const avg = normalized.reduce((sum, row) => sum + row.histVal, 0) / normalized.length;
    return avg;
}

/**
 * 대시보드 결과 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function DashboardResult({ personaData, optimizedData, investmentAmount, onRestart, onBack }) {
    if (!personaData) return null;

    // displayMode는 같은 메트릭을 퍼센트 기준으로 볼지 원화 기준으로 볼지 결정합니다.
    const [displayMode, setDisplayMode] = useState('pct');
    // isKRW는 버튼 상태뿐 아니라 helper 문구, 카드 라벨, 값 포맷까지 함께 바꾸는 파생 플래그입니다.
    const isKRW = displayMode === 'krw' && investmentAmount;
    // invAmt는 원화 환산 공식을 위해 항상 숫자형 기본값을 유지합니다.
    const invAmt = investmentAmount || 0;

    // hasOptimized는 1차 추천 결과 대신 최종 최적화 결과를 우선 표시해야 하는지 나타냅니다.
    const hasOptimized = optimizedData && optimizedData.optimized_stocks;
    // displayStocks는 실제 화면의 카드/차트/원형 그래프가 공유하는 기준 종목 목록입니다.
    const displayStocks = hasOptimized
        ? [...optimizedData.optimized_stocks].sort((a, b) => toFiniteNumber(b.weight, 0) - toFiniteNumber(a.weight, 0))
        : [...(personaData.recommendedStocks || [])].sort((a, b) => toFiniteNumber(b.weight, 0) - toFiniteNumber(a.weight, 0));

    // rawAnswers는 MBTI 막대 시각화용 원본 설문 응답 배열입니다.
    const rawAnswers = personaData.rawAnswers || [];
    // mbti는 아바타 상단에 크게 노출되는 대표 코드 문자열입니다.
    const mbti = personaData.mbti || 'ENTJ';
    // mbtiScores는 rawAnswers 12개를 4개 축 점수로 압축한 결과입니다.
    const mbtiScores = getMbtiScores(rawAnswers);
    // portfolioAnalysis는 1차 추천 응답에 포함된 메트릭 묶음입니다.
    const portfolioAnalysis = personaData.portfolioAnalysis || {};

    // historical3m은 과거 실적 카드용 값이고, 최적화 데이터가 있으면 백엔드 재계산 결과를 우선 사용합니다.
    const historical3m = hasOptimized
        ? toFiniteNumber(optimizedData.past_3m_return, 0)
        : getWeightedHistoricalReturn(displayStocks, '3M');
    // expected3m은 향후 3개월 기대수익률 카드와 차트 설명 배지에 함께 쓰입니다.
    const expected3m = hasOptimized
        ? toFiniteNumber(optimizedData.portfolio_expected_return_3m_simple, 0)
        : toFiniteNumber(portfolioAnalysis.expected_return_simple, 0);
    // volatility60dPct는 백엔드에서 소수 비율로 올 수 있어 화면 표시 전 퍼센트 단위로 변환합니다.
    const volatility60dPct = hasOptimized
        ? toFiniteNumber(optimizedData.portfolio_volatility, 0) * 100
        : toFiniteNumber(portfolioAnalysis.volatility_60d, 0) * 100;
    // var5는 손실 위험 지표이므로 기존 결과가 양수로 들어와도 음수 방향으로 정규화해 표시합니다.
    const var5 = hasOptimized
        ? toFiniteNumber(optimizedData.var_5, 0)
        : -Math.abs(toFiniteNumber(portfolioAnalysis.var_5, 0));
    // personaImage는 title 문자열보다 level 매핑을 우선 적용해 대표 캐릭터를 결정합니다.
    const personaImage = getPersonaImage(personaData.title, personaData.finalLevel);

    // chartData와 forecastData는 최적화 전/후 모두 동일한 차트 컴포넌트 인터페이스를 유지합니다.
    const chartData = hasOptimized ? optimizedData.chart_data : personaData.chartData;
    const forecastData = hasOptimized ? optimizedData.forecast_data : personaData.forecastData;

    // pScores는 수익률/위험도 바와 위험도 계산 설명 문구를 동시에 채우는 스코어 묶음입니다.
    const pScores = hasOptimized
        ? (optimizedData?.portfolio_scores || { return_pct: 0, risk_pct: 0, risk_pct_naive: 0, diversification_benefit: 0 })
        : { return_pct: 0, risk_pct: 0, risk_pct_naive: 0, diversification_benefit: 0 };

    // metricCards는 동일한 렌더링 껍데기에 서로 다른 메트릭을 주입하기 위한 표시용 배열입니다.
    const metricCards = [
        {
            label: isKRW ? '과거 3개월 수익' : '과거 3개월 수익률',
            value: isKRW ? formatKRW(historical3m, invAmt) : formatPercent(historical3m),
            helper: '보유 비중 가중 평균',
            tone: historical3m >= 0 ? 'up' : 'down',
            icon: <TrendingUp size={18} />,
        },
        {
            label: isKRW ? '예측 3개월 수익' : '예측 3개월 수익률',
            value: isKRW ? formatKRW(expected3m, invAmt) : formatPercent(expected3m),
            helper: isKRW ? '모델 기반 기대 수익' : '모델 기반 기대 수익률',
            tone: expected3m >= 0 ? 'up' : 'down',
            icon: <Sparkles size={18} />,
        },
        {
            label: isKRW ? '변동액' : '변동성',
            value: isKRW ? formatKRW(volatility60dPct, invAmt, false) : `${volatility60dPct.toFixed(2)}%`,
            helper: '최근 60일 표준편차',
            tone: 'neutral',
            icon: <ShieldAlert size={18} />,
        },
        {
            label: 'VaR 5%',
            value: isKRW ? formatKRW(var5, invAmt) : formatPercent(var5),
            helper: '95% 신뢰수준 손실 수준',
            tone: 'down',
            icon: <ShieldAlert size={18} />,
        },
    ];

    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
    return (
        <div className="premium-dashboard">
            <div className="premium-orb orb-a" />
            <div className="premium-orb orb-b" />

            <header className="premium-header reveal delay-1">
                <p className="eyebrow" style={{ marginBottom: 0 }}>Investment MBTI Report</p>
                <div className="premium-header-row">
                    <h1>추천 포트폴리오 리포트</h1>
                    <p className="subtitle premium-header-note">성향 분석과 시장 데이터를 결합한 시뮬레이션 결과입니다.</p>
                </div>
            </header>

            <section className="premium-layout">
                {/* 좌측 패널은 성향 요약과 MBTI 축 시각화를, 우측 컬럼은 성과/차트/점수/종목 목록을 담당합니다. */}
                <aside className="glass-panel profile-panel reveal delay-2">
                    <div className="profile-avatar-wrap">
                        <img
                            src={`${import.meta.env.BASE_URL}images/${personaImage}.png`}
                            alt="투자 성향 아바타"
                            className="profile-avatar"
                            onError={(e) => {
                                e.currentTarget.style.display = 'none';
                            }}
                        />
                    </div>

                    <h2 className="mbti-code">{mbti}</h2>
                    <h3 className="persona-title">{personaData.title}</h3>
                    <p className="persona-desc">{personaData.description}</p>

                    <div className="mbti-bars">
                        {BAR_CONFIG.map((item) => (
                            <MbtiBarChart
                                key={item.key}
                                dimension={item.label}
                                score={mbtiScores[item.key]}
                                leftLabel={item.left}
                                rightLabel={item.right}
                                gradient={item.gradient}
                            />
                        ))}
                    </div>
                </aside>

                <div className="main-column">
                    <article className="glass-panel composition-panel reveal delay-3">
                        {/* composition-panel은 비중 분포와 핵심 메트릭 카드를 한 번에 보여 주는 요약 섹션입니다. */}
                        <div className="panel-heading">
                            <h3>추천 포트폴리오 구성</h3>
                            <div className="panel-heading-right">
                                {/* % / ₩ 토글 버튼 */}
                                <div className="mode-toggle">
                                    <button
                                        type="button"
                                        className={`mode-btn${displayMode === 'pct' ? ' active' : ''}`}
                                        onClick={() => setDisplayMode('pct')}
                                    >%</button>
                                    <button
                                        type="button"
                                        className={`mode-btn${displayMode === 'krw' ? ' active' : ''}`}
                                        onClick={() => setDisplayMode('krw')}
                                        disabled={!investmentAmount}
                                        title={!investmentAmount ? '투자 금액 정보가 없습니다' : '원화로 표시'}
                                    >₩</button>
                                </div>
                                <span className="chip" dangerouslySetInnerHTML={{
                                    __html: (portfolioAnalysis.risk_category || '균형형').replace(
                                        /(간신히|가볍게|거뜬히|무참히)/,
                                        '<strong>$1</strong>'
                                    )
                                }} />
                            </div>
                        </div>

                        <div className="composition-content">
                            <PortfolioPieChart stocks={displayStocks} size={240} />

                            <div className="metrics-grid" key={displayMode}>
                                {metricCards.map((metric, idx) => (
                                    <div
                                        key={metric.label}
                                        className={`metric-card tone-${metric.tone} metric-flip`}
                                        style={{ animationDelay: `${idx * 60}ms` }}
                                    >
                                        <div className="metric-head">
                                            <span className="metric-icon">{metric.icon}</span>
                                            <span className="metric-label">{metric.label}</span>
                                        </div>
                                        <p className="metric-value">{metric.value}</p>
                                        <p className="metric-helper">{metric.helper}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </article>

                    <article className="glass-panel chart-panel reveal delay-4">
                        {/* chart-panel은 과거 실적과 미래 예측 경로를 한 축 위에 겹쳐 보여 주는 시계열 섹션입니다. */}
                        <div className="panel-heading chart-heading">
                            <h3>포트폴리오 성과 예측</h3>
                            <span className="chart-heading-sub">과거 1년 + 향후 3개월</span>
                            <span className="chip chart-chip">3개월 뒤 예상 수익률 확률 분포</span>
                        </div>

                        <CumulativeReturnChart
                            chartData={chartData}
                            forecastData={forecastData}
                            var5Display={var5}
                            warnings={chartData?.warnings || []}
                        />
                    </article>

                    {/* 수익률/위험도 바 */}
                    <article className="glass-panel score-bars-panel reveal delay-4">
                        {/* score-bars-panel은 수익률/위험도를 0~100 스케일 바 형태로 요약합니다. */}
                        <div className="panel-heading score-heading">
                            <h3>포트폴리오 스코어</h3>
                            <div className="risk-note">
                                <span style={{ fontWeight: 600, color: '#475569' }}>위험도 산출</span><br />
                                단순 가중합 {pScores.risk_pct_naive}
                                {pScores.diversification_benefit > 0 && (
                                    <> − 분산효과 {pScores.diversification_benefit}</>
                                )}
                                {' '}= <strong style={{ color: '#0f172a' }}>{pScores.risk_pct}</strong>
                            </div>
                        </div>
                        <div className="ps-bars-wrap">
                            {/* 수익률 바 */}
                            <div className="ps-bar-row">
                                <span className="ps-bar-label">(예상) 수익률 :</span>
                                <div className="ps-bar-track">
                                    <div
                                        className="ps-bar-fill ps-bar-fill--return"
                                        style={{ width: `${pScores.return_pct}%` }}
                                    />
                                </div>
                                <span className="ps-bar-value">{pScores.return_pct} / 100</span>
                            </div>
                            {/* 위험도 바 */}
                            <div className="ps-bar-row">
                                <span className="ps-bar-label">(예상) 위험도 :</span>
                                <div className="ps-bar-track">
                                    <div
                                        className="ps-bar-fill ps-bar-fill--risk"
                                        style={{ width: `${pScores.risk_pct_naive}%` }}
                                    />
                                    {pScores.diversification_benefit > 0 && (
                                        <div
                                            className="ps-bar-diversification"
                                            style={{
                                                left: `${pScores.risk_pct}%`,
                                                width: `${pScores.diversification_benefit}%`,
                                            }}
                                        />
                                    )}
                                </div>
                                <span className="ps-bar-value">{pScores.risk_pct} / 100</span>
                            </div>
                        </div>
                    </article>
                </div>
            </section>

            <section className="glass-panel stocks-panel reveal delay-5">
                {/* stocks-panel은 최종 표시 종목 목록을 카드 그리드로 렌더링하는 상세 섹션입니다. */}
                <div className="panel-heading stocks-heading">
                    <h3>포트폴리오 종목</h3>
                    <span className="chip">{displayStocks.length}개 종목</span>
                </div>

                {displayStocks.length > 0 ? (
                    <div className="stock-grid">
                        {displayStocks.map((stock) => (
                            <StockCard
                                key={stock.ticker}
                                stock={stock}
                                displayMode={displayMode}
                                investmentAmount={invAmt}
                            />
                        ))}
                    </div>
                ) : (
                    <div className="empty-state">추천 가능한 종목 데이터가 없습니다.</div>
                )}
            </section>

            <div className="result-actions reveal delay-5">
                <button type="button" className="restart-button" onClick={onRestart}>
                    <RefreshCw size={16} />
                    처음으로
                </button>
                {onBack && (
                    <button type="button" className="restart-button restart-button--back" onClick={onBack}>
                        <ArrowLeft size={16} />
                        이전으로
                    </button>
                )}
            </div>
        </div>
    );
}

export default DashboardResult;

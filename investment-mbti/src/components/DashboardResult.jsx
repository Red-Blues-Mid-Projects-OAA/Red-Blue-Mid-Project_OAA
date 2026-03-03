import React from 'react';
import { RefreshCw, Sparkles, TrendingUp, ShieldAlert } from 'lucide-react';
import MbtiBarChart from './MbtiBarChart';
import PortfolioPieChart from './PortfolioPieChart';
import CumulativeReturnChart from './CumulativeReturnChart';
import StockCard from './StockCard';
import './DashboardResult.css';

const PERSONA_IMAGE_BY_LEVEL = {
    4: 'slave',
    3: 'worker',
    2: 'fire',
    1: 'yolo',
};

const BAR_CONFIG = [
    { key: 'energy', label: '시장 반응', left: '외향형', right: '내향형', gradient: 'linear-gradient(90deg, #a7f3d0 0%, #3b82f6 100%)' },
    { key: 'insight', label: '가치 판단', left: '감각형', right: '직관형', gradient: 'linear-gradient(90deg, #bfdbfe 0%, #2563eb 100%)' },
    { key: 'logic', label: '의사 결정', left: '사고형', right: '감정형', gradient: 'linear-gradient(90deg, #dbeafe 0%, #1d4ed8 100%)' },
    { key: 'style', label: '대응 방식', left: '계획형', right: '유연형', gradient: 'linear-gradient(90deg, #c7d2fe 0%, #3b82f6 100%)' },
];

function toFiniteNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

function formatPercent(value, digits = 2, forceSign = true) {
    if (value == null || Number.isNaN(Number(value))) {
        return '-';
    }
    const n = Number(value);
    const sign = forceSign ? (n >= 0 ? '+' : '') : '';
    return `${sign}${n.toFixed(digits)}%`;
}

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

function getMbtiScores(rawAnswers = []) {
    if (rawAnswers.length < 12) {
        return {
            energy: 50,
            insight: 50,
            logic: 50,
            style: 50,
        };
    }

    const answerAt = (idx) => String(rawAnswers[idx] || '').toUpperCase();
    const countRange = (start, end, expected) => {
        let count = 0;
        for (let i = start; i < end; i += 1) {
            if (answerAt(i) === expected) count += 1;
        }
        return count;
    };

    return {
        energy: (countRange(0, 3, 'A') / 3) * 100, // I
        insight: (countRange(3, 6, 'B') / 3) * 100, // N
        logic: (countRange(6, 9, 'B') / 3) * 100, // F
        style: (countRange(9, 12, 'B') / 3) * 100, // P
    };
}

function getWeightedHistoricalReturn(stocks, key = '3M') {
    if (!stocks.length) return 0;

    const normalized = stocks.map((stock) => {
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

function DashboardResult({ personaData, onRestart, onShowSelection }) {
    if (!personaData) return null;

    const recommendedStocks = [...(personaData.recommendedStocks || [])]
        .sort((a, b) => toFiniteNumber(b.weight, 0) - toFiniteNumber(a.weight, 0));

    const rawAnswers = personaData.rawAnswers || [];
    const mbti = personaData.mbti || 'ENTJ';
    const mbtiScores = getMbtiScores(rawAnswers);
    const portfolioAnalysis = personaData.portfolioAnalysis || {};
    const historical3m = getWeightedHistoricalReturn(recommendedStocks, '3M');
    const expected3m = toFiniteNumber(portfolioAnalysis.expected_return_simple, 0);
    const volatility60dPct = toFiniteNumber(portfolioAnalysis.volatility_60d, 0) * 100;
    const var5 = -Math.abs(toFiniteNumber(portfolioAnalysis.var_5, 0));
    const personaImage = getPersonaImage(personaData.title, personaData.finalLevel);

    const metricCards = [
        {
            label: '과거 3개월 수익률',
            value: formatPercent(historical3m),
            helper: '보유 비중 가중 평균',
            tone: historical3m >= 0 ? 'up' : 'down',
            icon: <TrendingUp size={18} />,
        },
        {
            label: '예측 3개월 수익률',
            value: formatPercent(expected3m),
            helper: '모델 기반 기대 수익률',
            tone: expected3m >= 0 ? 'up' : 'down',
            icon: <Sparkles size={18} />,
        },
        {
            label: '변동성',
            value: `${volatility60dPct.toFixed(2)}%`,
            helper: '최근 60일 표준편차',
            tone: 'neutral',
            icon: <ShieldAlert size={18} />,
        },
        {
            label: 'VaR 5%',
            value: formatPercent(var5),
            helper: '95% 신뢰수준 손실 경계',
            tone: 'down',
            icon: <ShieldAlert size={18} />,
        },
    ];

    return (
        <div className="premium-dashboard">
            <div className="premium-orb orb-a" />
            <div className="premium-orb orb-b" />

            <header className="premium-header reveal delay-1">
                <p className="eyebrow">Investment MBTI Report</p>
                <h1>추천 포트폴리오 리포트</h1>
                <p className="subtitle">성향 분석과 시장 데이터를 결합한 시뮬레이션 결과입니다.</p>
            </header>

            <section className="premium-layout">
                <aside className="glass-panel profile-panel reveal delay-2">
                    <div className="profile-avatar-wrap">
                        <img
                            src={`/images/${personaImage}.png`}
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
                        <div className="panel-heading">
                            <h3>추천 포트폴리오 구성</h3>
                            <span className="chip" dangerouslySetInnerHTML={{
                                __html: (portfolioAnalysis.risk_category || '균형형').replace(
                                    /(간신히|가볍게|거뜬히|무참히)/,
                                    '<strong>$1</strong>'
                                )
                            }} />
                        </div>

                        <div className="composition-content">
                            <PortfolioPieChart stocks={recommendedStocks} size={240} />

                            <div className="metrics-grid">
                                {metricCards.map((metric) => (
                                    <div key={metric.label} className={`metric-card tone-${metric.tone}`}>
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
                        <div className="panel-heading">
                            <h3>포트폴리오 성과 예측</h3>
                            <span className="chip">과거 1년 + 향후 3개월</span>
                        </div>

                        <CumulativeReturnChart
                            chartData={personaData.chartData}
                            forecastData={personaData.forecastData}
                            warnings={personaData.chartData?.warnings || []}
                        />
                    </article>
                </div>
            </section>

            <section className="glass-panel stocks-panel reveal delay-5">
                <div className="panel-heading">
                    <h3>추천 종목</h3>
                    <span className="chip">{recommendedStocks.length}개 종목</span>
                </div>

                {recommendedStocks.length > 0 ? (
                    <div className="stock-grid">
                        {recommendedStocks.map((stock) => (
                            <StockCard key={stock.ticker} stock={stock} />
                        ))}
                    </div>
                ) : (
                    <div className="empty-state">추천 가능한 종목 데이터가 없습니다.</div>
                )}
            </section>

            <div className="result-actions reveal delay-5">
                <button type="button" className="restart-button" onClick={onRestart}>
                    <RefreshCw size={16} />
                    다시 분석하기
                </button>
                {onShowSelection && (
                    <button type="button" className="restart-button" style={{ marginLeft: '0.6rem', background: 'linear-gradient(180deg, #eff6ff 0%, #dbeafe 100%)' }} onClick={onShowSelection}>
                        <Sparkles size={16} />
                        종목 선택하기
                    </button>
                )}
            </div>
        </div>
    );
}

export default DashboardResult;


/*
 * 이 파일은 종목 카드 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React from 'react';
import './DashboardResult.css';

/**
 * 입력값을 안전한 숫자로 바꿔 계산에 사용할 수 있게 합니다.
 */
function toFiniteNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

/**
 * 수익률의 부호에 따라 화면에 쓸 스타일 클래스를 고릅니다.
 */
function signClass(value) {
    return value >= 0 ? 'up' : 'down';
}

/**
 * 숫자 앞에 부호를 붙여 퍼센트 문자열로 보여 줍니다.
 */
function formatSigned(value, digits = 2) {
    const n = toFiniteNumber(value, 0);
    return `${n >= 0 ? '+' : ''}${n.toFixed(digits)}%`;
}

// 종목별 수익률을 투자 비중 기반 원화로 변환
function formatStockKRW(pct, weight, manwon) {
    const stockAllocation = (weight / 100) * manwon * 10000; // 이 종목에 배분된 금액 (원)
    const won = (pct / 100) * stockAllocation;
    const absWon = Math.abs(won);
    const sign = won >= 0 ? '+' : '-';

    if (absWon >= 100000000) {
        return `${sign}${(absWon / 100000000).toFixed(2)}억`;
    } else if (absWon >= 10000) {
        return `${sign}${Math.round(absWon / 10000).toLocaleString()}만`;
    }
    return `${sign}${Math.round(absWon).toLocaleString()}원`;
}

/**
 * 종목 카드 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function StockCard({ stock, displayMode, investmentAmount }) {
    const ticker = String(stock?.ticker || '').toUpperCase();
    const name = String(stock?.name || ticker || '-');
    const weight = Math.max(toFiniteNumber(stock?.weight, 0), 0);
    const riskRank = toFiniteNumber(stock?.risk_rank, 0);
    const expected3m = toFiniteNumber(stock?.expectedReturn3M_simple ?? stock?.expectedReturn3M, 0);

    const hist = stock?.historical_returns || {};
    const periods = [
        { label: '1M', value: toFiniteNumber(hist?.['1M'], 0) },
        { label: '3M', value: toFiniteNumber(hist?.['3M'], 0) },
        { label: '6M', value: toFiniteNumber(hist?.['6M'], 0) },
        { label: '12M', value: toFiniteNumber(hist?.['12M'], 0) },
    ];

    // ₩ 모드: 투자금이 있을 때만 원화로 표시
    const isKRW = displayMode === 'krw' && investmentAmount;
    const fmt = (pct, digits = 2) =>
        isKRW
            ? formatStockKRW(pct, weight, investmentAmount)
            : formatSigned(pct, digits);

    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
    return (
        <article className="premium-stock-card">
            <div className="stock-card-top">
                <div className="stock-card-head">
                    <h4 className="stock-card-title">{name}</h4>
                    <p className="stock-card-ticker">{ticker}</p>
                </div>

                <div className="stock-card-badges">
                    <span className="stock-badge weight">Ratio {weight.toFixed(1)}%</span>
                    <span className="stock-badge risk">Risk {riskRank}위</span>
                </div>
            </div>

            <div className="stock-card-return">
                <span className="label">예상 3개월</span>
                <span className={`value ${signClass(expected3m)}`}>{fmt(expected3m)}</span>
            </div>

            <div className="stock-card-periods">
                {periods.map((period) => (
                    <div className="stock-period" key={`${ticker}-${period.label}`}>
                        <span className="period-label">{period.label}</span>
                        <span className={`period-value ${signClass(period.value)}`}>
                            {fmt(period.value, 1)}
                        </span>
                    </div>
                ))}
            </div>
        </article>
    );
}

export default StockCard;

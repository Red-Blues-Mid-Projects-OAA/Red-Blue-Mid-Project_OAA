import React from 'react';
import './DashboardResult.css';

function toFiniteNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

function signClass(value) {
    return value >= 0 ? 'up' : 'down';
}

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

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

function StockCard({ stock }) {
    const ticker = String(stock?.ticker || '').toUpperCase();
    const name = String(stock?.name || ticker || '-');
    const weight = Math.max(toFiniteNumber(stock?.weight, 0), 0);
    const riskScore = Math.max(0, Math.round(toFiniteNumber(stock?.risk_score, 0)));
    const expected3m = toFiniteNumber(stock?.expectedReturn3M_simple ?? stock?.expectedReturn3M, 0);

    const hist = stock?.historical_returns || {};
    const periods = [
        { label: '1M', value: toFiniteNumber(hist?.['1M'], 0) },
        { label: '3M', value: toFiniteNumber(hist?.['3M'], 0) },
        { label: '6M', value: toFiniteNumber(hist?.['6M'], 0) },
        { label: '12M', value: toFiniteNumber(hist?.['12M'], 0) },
    ];

    return (
        <article className="premium-stock-card">
            <div className="stock-card-top">
                <div>
                    <h4 className="stock-card-title">{name}</h4>
                    <p className="stock-card-ticker">{ticker}</p>
                </div>

                <div className="stock-card-badges">
                    <span className="stock-badge weight">{weight.toFixed(2)}%</span>
                    <span className="stock-badge risk">Risk {riskScore}</span>
                </div>
            </div>

            <div className="stock-card-return">
                <span className="label">예상 3개월</span>
                <span className={`value ${signClass(expected3m)}`}>{formatSigned(expected3m)}</span>
            </div>

            <div className="stock-card-periods">
                {periods.map((period) => (
                    <div className="stock-period" key={`${ticker}-${period.label}`}>
                        <span className="period-label">{period.label}</span>
                        <span className={`period-value ${signClass(period.value)}`}>
                            {formatSigned(period.value, 1)}
                        </span>
                    </div>
                ))}
            </div>
        </article>
    );
}

export default StockCard;

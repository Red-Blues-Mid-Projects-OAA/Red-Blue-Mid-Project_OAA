import React from 'react';
import './DashboardResult.css';

const COLORS = [
    '#1d4ed8',
    '#2563eb',
    '#3b82f6',
    '#60a5fa',
    '#93c5fd',
    '#bfdbfe',
    '#dbeafe',
    '#e2e8f0',
];

function toFiniteNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

function arcPath(cx, cy, outerR, innerR, startAngle, endAngle) {
    const startRad = (Math.PI / 180) * startAngle;
    const endRad = (Math.PI / 180) * endAngle;
    const largeArc = Math.abs(endAngle - startAngle) > 180 ? 1 : 0;

    const x1 = cx + outerR * Math.cos(startRad);
    const y1 = cy + outerR * Math.sin(startRad);
    const x2 = cx + outerR * Math.cos(endRad);
    const y2 = cy + outerR * Math.sin(endRad);

    const x3 = cx + innerR * Math.cos(endRad);
    const y3 = cy + innerR * Math.sin(endRad);
    const x4 = cx + innerR * Math.cos(startRad);
    const y4 = cy + innerR * Math.sin(startRad);

    return [
        `M ${x1} ${y1}`,
        `A ${outerR} ${outerR} 0 ${largeArc} 1 ${x2} ${y2}`,
        `L ${x3} ${y3}`,
        `A ${innerR} ${innerR} 0 ${largeArc} 0 ${x4} ${y4}`,
        'Z',
    ].join(' ');
}

function PortfolioPieChart({ stocks, size = 240 }) {
    const validStocks = (stocks || [])
        .map((stock) => ({
            ...stock,
            ticker: String(stock?.ticker || '').toUpperCase(),
            weight: Math.max(toFiniteNumber(stock?.weight, 0), 0),
        }))
        .filter((stock) => stock.weight > 0)
        .sort((a, b) => b.weight - a.weight);

    if (!validStocks.length) {
        return <div className="chart-fallback" style={{ width: size, height: size }}>비중 데이터 없음</div>;
    }

    const totalWeight = validStocks.reduce((sum, stock) => sum + stock.weight, 0);
    const cx = size / 2;
    const cy = size / 2;
    const outerR = (size / 2) - 12;
    const innerR = outerR * 0.56;

    let currentAngle = -90;
    const slices = validStocks.map((stock, idx) => {
        const angle = (stock.weight / totalWeight) * 360;
        const start = currentAngle;
        const end = currentAngle + angle;
        currentAngle = end;

        return {
            ...stock,
            color: COLORS[idx % COLORS.length],
            path: arcPath(cx, cy, outerR, innerR, start, end),
        };
    });

    const topLegend = slices.slice(0, 6);

    return (
        <div className="premium-pie-wrap">
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-label="포트폴리오 비중 도넛 차트">
                <circle cx={cx} cy={cy} r={outerR} fill="#f8fafc" stroke="#dbeafe" strokeWidth="1.4" />

                {slices.map((slice) => (
                    <path
                        key={slice.ticker}
                        d={slice.path}
                        fill={slice.color}
                        stroke="#ffffff"
                        strokeWidth="1.2"
                        opacity="0.96"
                    >
                        <title>{slice.ticker}: {slice.weight.toFixed(2)}%</title>
                    </path>
                ))}

                <circle cx={cx} cy={cy} r={innerR} fill="#ffffff" stroke="#e2e8f0" strokeWidth="1.2" />
                <text x={cx} y={cy - 6} textAnchor="middle" fill="#64748b" fontSize="11" fontWeight="700">
                    추천 포트폴리오
                </text>
                <text x={cx} y={cy + 14} textAnchor="middle" fill="#0f172a" fontSize="18" fontWeight="800">
                    {validStocks.length} 종목
                </text>
            </svg>

            <div className="premium-pie-legend">
                {topLegend.map((slice) => (
                    <div className="premium-pie-legend-item" key={`${slice.ticker}-legend`}>
                        <span className="premium-pie-legend-left">
                            <span className="premium-pie-dot" style={{ backgroundColor: slice.color }} />
                            <span className="premium-pie-ticker">{slice.ticker}</span>
                        </span>
                        <span className="premium-pie-weight">{slice.weight.toFixed(2)}%</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default PortfolioPieChart;


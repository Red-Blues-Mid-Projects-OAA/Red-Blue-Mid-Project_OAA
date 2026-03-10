/*
 * 이 파일은 포트폴리오 원형 차트 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React, { useState, useCallback } from 'react';
import './DashboardResult.css';

/* ── 세련된 파스텔 톤 10색 팔레트 ── */
const COLORS = [
    '#7C9CF5', // 소프트 블루
    '#A78BFA', // 라벤더
    '#6EE7B7', // 민트
    '#FCD34D', // 레몬
    '#F9A8D4', // 로즈
    '#67E8F9', // 아쿠아
    '#FDA4AF', // 코랄
    '#C4B5FD', // 페리윙클
    '#86EFAC', // 라이트 그린
    '#FDBA74', // 피치
];

/**
 * 입력값을 안전한 숫자로 바꿔 계산에 사용할 수 있게 합니다.
 */
function toFiniteNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

/**
 * 도넛 차트 한 조각을 그릴 SVG 경로 문자열을 만듭니다.
 */
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

/* ── 커스텀 hover 툴팁 ── */
function PieTooltip({ slice, x, y, visible }) {
    if (!visible || !slice) return null;
    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
    return (
        <div
            className="pie-tooltip"
            style={{
                left: x,
                top: y,
            }}
        >
            <div className="pie-tooltip-header">
                <span className="pie-tooltip-dot" style={{ backgroundColor: slice.color }} />
                <strong>{slice.ticker}</strong>
            </div>
            <div className="pie-tooltip-row">
                <span>비중</span>
                <span className="pie-tooltip-value">{slice.weight.toFixed(1)}%</span>
            </div>
        </div>
    );
}

/**
 * 포트폴리오 원형 차트 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function PortfolioPieChart({ stocks, size = 240 }) {
    // 화면에서 계속 바뀌는 값을 state로 보관합니다.
    const [hoverIdx, setHoverIdx] = useState(null);
    const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });

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
            idx,
            color: COLORS[idx % COLORS.length],
            path: arcPath(cx, cy, outerR, innerR, start, end),
            // 호버 시 살짝 커진 path
            hoverPath: arcPath(cx, cy, outerR + 5, innerR - 2, start, end),
        };
    });

    const topLegend = slices.slice(0, 6);

    // 사용자 입력, 버튼 클릭, API 요청을 처리하는 함수들입니다.
    const handleMouseMove = useCallback((e) => {
        const rect = e.currentTarget.closest('.premium-pie-wrap')?.getBoundingClientRect();
        if (!rect) return;
        setTooltipPos({
            x: e.clientX - rect.left + 14,
            y: e.clientY - rect.top - 10,
        });
    }, []);

    return (
        <div className="premium-pie-wrap" style={{ position: 'relative' }}>
            <svg
                width={size}
                height={size + 10}
                viewBox={`-5 -5 ${size + 10} ${size + 10}`}
                aria-label="포트폴리오 비중 도넛 차트"
            >
                <circle cx={cx} cy={cy} r={outerR} fill="#f8fafc" stroke="#e2e8f0" strokeWidth="1" />

                {slices.map((slice) => (
                    <path
                        key={slice.ticker}
                        d={hoverIdx === slice.idx ? slice.hoverPath : slice.path}
                        fill={slice.color}
                        stroke="#ffffff"
                        strokeWidth={hoverIdx === slice.idx ? 2.8 : 2}
                        opacity={hoverIdx === null || hoverIdx === slice.idx ? 1 : 0.55}
                        style={{
                            transition: 'all 0.22s ease',
                            filter: hoverIdx === slice.idx ? 'drop-shadow(0 3px 8px rgba(0,0,0,0.18))' : 'none',
                            cursor: 'pointer',
                        }}
                        onMouseEnter={() => setHoverIdx(slice.idx)}
                        onMouseMove={handleMouseMove}
                        onMouseLeave={() => setHoverIdx(null)}
                    />
                ))}

                <circle cx={cx} cy={cy} r={innerR} fill="#ffffff" stroke="#e2e8f0" strokeWidth="1" />
                <text x={cx} y={cy - 6} textAnchor="middle" fill="#64748b" fontSize="11" fontWeight="700">
                    추천 포트폴리오
                </text>
                <text x={cx} y={cy + 14} textAnchor="middle" fill="#0f172a" fontSize="18" fontWeight="800">
                    {validStocks.length} 종목
                </text>
            </svg>

            <PieTooltip
                slice={hoverIdx !== null ? slices[hoverIdx] : null}
                x={tooltipPos.x}
                y={tooltipPos.y}
                visible={hoverIdx !== null}
            />

            <div className="premium-pie-legend">
                {topLegend.map((slice) => (
                    <div
                        className={`premium-pie-legend-item ${hoverIdx === slice.idx ? 'legend-active' : ''}`}
                        key={`${slice.ticker}-legend`}
                        onMouseEnter={() => setHoverIdx(slice.idx)}
                        onMouseLeave={() => setHoverIdx(null)}
                    >
                        <span className="premium-pie-legend-left">
                            <span className="premium-pie-dot" style={{ backgroundColor: slice.color }} />
                            <span className="premium-pie-ticker">{slice.ticker}</span>
                        </span>
                        <span className="premium-pie-weight">{slice.weight.toFixed(1)}%</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default PortfolioPieChart;

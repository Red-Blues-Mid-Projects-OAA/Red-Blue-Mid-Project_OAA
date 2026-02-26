import React from 'react';

/**
 * SVG 기반 포트폴리오 비중 파이(도넛) 차트 컴포넌트.
 * 외부 차트 라이브러리 없이 순수 SVG로 구현.
 */

// 파이차트 색상 팔레트
const COLORS = [
    '#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
];

function PortfolioPieChart({ stocks, size = 220 }) {
    // 비중이 0이 아닌 종목만 필터링 후 정렬
    const validStocks = (stocks || [])
        .filter(s => s.weight > 0)
        .sort((a, b) => b.weight - a.weight);

    if (validStocks.length === 0) {
        return (
            <div className="flex items-center justify-center" style={{ width: size, height: size }}>
                <span className="text-gray-500 text-sm">데이터 없음</span>
            </div>
        );
    }

    const cx = size / 2;
    const cy = size / 2;
    const outerR = size / 2 - 10;
    const innerR = outerR * 0.55; // 도넛 두께

    // 비중 합계 정규화
    const totalWeight = validStocks.reduce((sum, s) => sum + s.weight, 0);

    // 파이 조각 경로 생성
    let startAngle = -90; // 12시 방향 시작
    const slices = validStocks.map((stock, i) => {
        const sliceAngle = (stock.weight / totalWeight) * 360;
        const endAngle = startAngle + sliceAngle;

        // 라디안 변환
        const startRad = (startAngle * Math.PI) / 180;
        const endRad = (endAngle * Math.PI) / 180;

        // 외부 호
        const x1 = cx + outerR * Math.cos(startRad);
        const y1 = cy + outerR * Math.sin(startRad);
        const x2 = cx + outerR * Math.cos(endRad);
        const y2 = cy + outerR * Math.sin(endRad);

        // 내부 호 (도넛 구멍)
        const x3 = cx + innerR * Math.cos(endRad);
        const y3 = cy + innerR * Math.sin(endRad);
        const x4 = cx + innerR * Math.cos(startRad);
        const y4 = cy + innerR * Math.sin(startRad);

        const largeArc = sliceAngle > 180 ? 1 : 0;

        const path = [
            `M ${x1} ${y1}`,
            `A ${outerR} ${outerR} 0 ${largeArc} 1 ${x2} ${y2}`,
            `L ${x3} ${y3}`,
            `A ${innerR} ${innerR} 0 ${largeArc} 0 ${x4} ${y4}`,
            'Z',
        ].join(' ');

        const result = {
            path,
            color: COLORS[i % COLORS.length],
            ticker: stock.ticker,
            weight: stock.weight,
        };

        startAngle = endAngle;
        return result;
    });

    return (
        <div className="flex flex-col items-center">
            <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
                {slices.map((slice, i) => (
                    <path
                        key={i}
                        d={slice.path}
                        fill={slice.color}
                        stroke="#1e293b"
                        strokeWidth="2"
                        className="transition-opacity hover:opacity-80"
                    >
                        <title>{slice.ticker}: {slice.weight.toFixed(1)}%</title>
                    </path>
                ))}
                {/* 중심 텍스트 */}
                <text x={cx} y={cy - 6} textAnchor="middle" fill="#94a3b8" fontSize="11" fontWeight="600">
                    포트폴리오
                </text>
                <text x={cx} y={cy + 12} textAnchor="middle" fill="#f8fafc" fontSize="14" fontWeight="800">
                    {validStocks.length}종목
                </text>
            </svg>

            {/* 범례 (하단) */}
            <div className="flex flex-wrap justify-center gap-x-3 gap-y-1 mt-2 max-w-[260px]">
                {slices.map((slice, i) => (
                    <div key={i} className="flex items-center gap-1">
                        <span
                            className="w-2.5 h-2.5 rounded-sm inline-block shrink-0"
                            style={{ backgroundColor: slice.color }}
                        />
                        <span className="text-[10px] text-gray-400">
                            {slice.ticker} {slice.weight.toFixed(1)}%
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default PortfolioPieChart;

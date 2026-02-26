import React from 'react';
import {
    ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, ReferenceLine, Legend
} from 'recharts';

/**
 * 누적수익률 차트 컴포넌트 (recharts 기반).
 *
 * - 과거 1년: 포트폴리오 (파란 실선) + S&P 500 (회색 실선)
 * - 현재(t) 이후 3개월: 예측선 (점선) + Monte Carlo 경로 (투명 회색)
 * - 차트 우측 끝: 시뮬레이션 최종 분포 표시 (텍스트)
 */
function CumulativeReturnChart({ chartData, forecastData, warnings = [] }) {
    if (!chartData || !chartData.dates || chartData.dates.length === 0) {
        return (
            <div className="flex items-center justify-center h-64 bg-gray-800/30 rounded-2xl border border-dashed border-gray-700">
                <span className="text-gray-500">차트 데이터를 불러오는 중...</span>
            </div>
        );
    }

    // 1. 과거 데이터 포매팅
    const historicalData = chartData.dates.map((date, i) => ({
        date: date.substring(5), // "MM-DD" 포맷
        fullDate: date,
        portfolio: chartData.portfolio[i],
        sp500: chartData.sp500[i],
    }));

    // 2. 예측 데이터 연결 (있는 경우)
    let combinedData = [...historicalData];
    let forecastStartIdx = historicalData.length;

    if (forecastData && forecastData.expected_line) {
        // 마지막 과거 데이터 포인트를 기준으로 예측 연결
        const lastHistValue = historicalData[historicalData.length - 1]?.portfolio || 0;

        forecastData.forecast_days.forEach((day, i) => {
            if (day === 0) return; // 첫 포인트는 이미 과거 데이터에 포함

            const forecastDate = `+${day}d`;
            combinedData.push({
                date: day % 21 === 0 ? `+${Math.round(day / 21)}M` : '',
                fullDate: forecastDate,
                forecast: lastHistValue + forecastData.expected_line[i],
                // Monte Carlo 경로 범위 (상한/하한)
                mcUpper: lastHistValue + (forecastData.final_distribution?.percentile_95 || 0) * (i / forecastData.forecast_days.length),
                mcLower: lastHistValue + (forecastData.final_distribution?.percentile_5 || 0) * (i / forecastData.forecast_days.length),
            });
        });
    }

    // 3. 커스텀 툴팁
    const CustomTooltip = ({ active, payload, label }) => {
        if (!active || !payload || payload.length === 0) return null;
        return (
            <div className="bg-gray-900/95 border border-gray-700 rounded-xl px-3 py-2 text-xs shadow-xl">
                <p className="text-gray-400 mb-1">{payload[0]?.payload?.fullDate || label}</p>
                {payload.map((entry, i) => (
                    entry.value != null && (
                        <p key={i} style={{ color: entry.color }} className="font-semibold">
                            {entry.name}: {entry.value >= 0 ? '+' : ''}{entry.value.toFixed(2)}%
                        </p>
                    )
                ))}
            </div>
        );
    };

    return (
        <div className="w-full">
            {/* 경고 문구 (최근 상장 종목 등) */}
            {warnings.length > 0 && (
                <div className="mb-2 px-3 py-1.5 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                    {warnings.map((w, i) => (
                        <p key={i} className="text-yellow-400 text-xs">⚠️ {w}</p>
                    ))}
                </div>
            )}

            <ResponsiveContainer width="100%" height={320}>
                <ComposedChart data={combinedData} margin={{ top: 10, right: 15, left: -15, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.4} />

                    <XAxis
                        dataKey="date"
                        tick={{ fill: '#6b7280', fontSize: 10 }}
                        axisLine={{ stroke: '#4b5563' }}
                        tickLine={false}
                        interval="preserveStartEnd"
                    />
                    <YAxis
                        tick={{ fill: '#6b7280', fontSize: 10 }}
                        axisLine={{ stroke: '#4b5563' }}
                        tickLine={false}
                        tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}%`}
                    />

                    <Tooltip content={<CustomTooltip />} />

                    <ReferenceLine y={0} stroke="#6b7280" strokeDasharray="4 4" opacity={0.5} />

                    {/* 현재 시점 표시 (과거 데이터 마지막 인덱스) */}
                    {forecastData && (
                        <ReferenceLine
                            x={historicalData[historicalData.length - 1]?.date}
                            stroke="#8b5cf6"
                            strokeWidth={2}
                            strokeDasharray="4 4"
                            label={{ value: '현재(t)', fill: '#8b5cf6', fontSize: 11, position: 'top' }}
                        />
                    )}

                    {/* Monte Carlo 범위 (음영) */}
                    <Area
                        dataKey="mcUpper"
                        stroke="none"
                        fill="#6b7280"
                        fillOpacity={0.08}
                        dot={false}
                        activeDot={false}
                        legendType="none"
                    />
                    <Area
                        dataKey="mcLower"
                        stroke="none"
                        fill="#6b7280"
                        fillOpacity={0.08}
                        dot={false}
                        activeDot={false}
                        legendType="none"
                    />

                    {/* S&P 500 벤치마크 (회색 실선) */}
                    <Line
                        type="monotone"
                        dataKey="sp500"
                        stroke="#6b7280"
                        strokeWidth={1.5}
                        dot={false}
                        name="S&P 500"
                    />

                    {/* 포트폴리오 과거 (파란 실선) */}
                    <Line
                        type="monotone"
                        dataKey="portfolio"
                        stroke="#3b82f6"
                        strokeWidth={2.5}
                        dot={false}
                        name="포트폴리오"
                    />

                    {/* 예측선 (보라색 점선) */}
                    <Line
                        type="monotone"
                        dataKey="forecast"
                        stroke="#8b5cf6"
                        strokeWidth={2}
                        strokeDasharray="6 3"
                        dot={false}
                        name="예측 (3M)"
                    />

                    <Legend
                        wrapperStyle={{ fontSize: '11px', color: '#9ca3af' }}
                        iconType="line"
                    />
                </ComposedChart>
            </ResponsiveContainer>

            {/* 시뮬레이션 최종 분포 (하단 라벨) */}
            {forecastData?.final_distribution && (
                <div className="flex justify-center gap-4 mt-2 text-[10px] text-gray-500">
                    <span>시뮬레이션 평균: <b className="text-gray-300">{forecastData.final_distribution.mean > 0 ? '+' : ''}{forecastData.final_distribution.mean}%</b></span>
                    <span>5%ile: <b className="text-red-400">{forecastData.final_distribution.percentile_5}%</b></span>
                    <span>95%ile: <b className="text-green-400">+{forecastData.final_distribution.percentile_95}%</b></span>
                </div>
            )}

            {/* 하단 X축 마일스톤 */}
            <div className="flex justify-between text-[10px] text-gray-600 mt-1 px-4">
                <span>과거 1년</span>
                <span>현재(t)</span>
                <span>+3개월</span>
            </div>
        </div>
    );
}

export default CumulativeReturnChart;

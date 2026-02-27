import React, { useMemo } from 'react';
import {
    Area,
    CartesianGrid,
    ComposedChart,
    Line,
    ReferenceLine,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
    Customized,
} from 'recharts';
import './DashboardResult.css';

function toFiniteNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

function formatSigned(value) {
    const n = toFiniteNumber(value, 0);
    return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
}

/* ── 시뮬레이션 끝에 이어 붙이는 정규분포 오버레이 ── */
function DistributionOverlay({ formattedGraphicalItems, yAxisMap, distribution }) {
    if (!distribution || !yAxisMap || !formattedGraphicalItems) return null;

    const yAxis = yAxisMap[Object.keys(yAxisMap)[0]];
    if (!yAxis) return null;

    const mean = toFiniteNumber(distribution.mean, 0);
    const std = Math.max(0.001, Math.abs(toFiniteNumber(distribution.std, 0.01)));
    const minValue = mean - std * 3;
    const maxValue = mean + std * 3;

    // yAxis 범위에서 실제 pixel 좌표 계산
    const yScale = yAxis.scale;
    if (!yScale) return null;

    // 차트 오른쪽 끝에서 그리기
    const chartRight = yAxis.x + yAxis.width + (yAxis.mirror ? 0 : 0);
    const curveWidth = 60;

    const samples = 50;
    const points = [];
    let peakDensity = Number.NEGATIVE_INFINITY;

    for (let i = 0; i <= samples; i += 1) {
        const t = i / samples;
        const value = minValue + (maxValue - minValue) * t;
        const z = (value - mean) / std;
        const density = Math.exp(-0.5 * z * z);
        points.push({ value, density });
        if (density > peakDensity) peakDensity = density;
    }

    const curvePoints = points
        .map(({ value, density }) => {
            const y = yScale(value);
            if (y == null || !Number.isFinite(y)) return null;
            const x = chartRight + (density / peakDensity) * curveWidth;
            return { x, y };
        })
        .filter(Boolean);

    if (curvePoints.length < 2) return null;

    const areaPath = `M ${chartRight},${curvePoints[0].y} ` +
        curvePoints.map(p => `L ${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ') +
        ` L ${chartRight},${curvePoints[curvePoints.length - 1].y} Z`;

    const linePath = curvePoints.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`
    ).join(' ');

    const meanY = yScale(mean);

    return (
        <g>
            <defs>
                <linearGradient id="distOverlayFill" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#93c5fd" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#2563eb" stopOpacity="0.08" />
                </linearGradient>
            </defs>
            <path d={areaPath} fill="url(#distOverlayFill)" />
            <path d={linePath} fill="none" stroke="#2563eb" strokeWidth="2" />
            {meanY != null && Number.isFinite(meanY) && (
                <line
                    x1={chartRight}
                    y1={meanY}
                    x2={chartRight + curveWidth}
                    y2={meanY}
                    stroke="#1e3a8a"
                    strokeDasharray="3 3"
                    strokeWidth="1.2"
                />
            )}
        </g>
    );
}

function buildSeries(chartData, forecastData) {
    const historicalRows = (chartData?.dates || []).map((date, idx) => ({
        x: idx,
        fullDate: date,
        portfolio: toFiniteNumber(chartData?.portfolio?.[idx], 0),
        sp500: toFiniteNumber(chartData?.sp500?.[idx], 0),
    }));

    if (!historicalRows.length) {
        return { rows: [], pathKeys: [], currentIndex: 0, lastIndex: 0 };
    }

    const currentIndex = historicalRows.length - 1;
    const rows = [...historicalRows];
    const hasForecast = Array.isArray(forecastData?.forecast_days) && forecastData.forecast_days.length > 1;
    const pathCount = Math.min(Array.isArray(forecastData?.mc_paths) ? forecastData.mc_paths.length : 0, 36);
    const pathKeys = Array.from({ length: pathCount }, (_, idx) => `mc_${idx}`);
    const lastHist = historicalRows[currentIndex].portfolio;

    rows[currentIndex].forecast = lastHist;
    for (let p = 0; p < pathCount; p += 1) {
        rows[currentIndex][pathKeys[p]] = lastHist;
    }

    if (hasForecast) {
        const total = forecastData.forecast_days.length;
        for (let i = 1; i < total; i += 1) {
            const day = forecastData.forecast_days[i];
            const row = {
                x: currentIndex + i,
                fullDate: `+${day}일`,
                forecast: lastHist + toFiniteNumber(forecastData.expected_line?.[i], 0),
                portfolio: null,
                sp500: null,
            };

            for (let p = 0; p < pathCount; p += 1) {
                const pathValue = forecastData.mc_paths?.[p]?.[i];
                row[pathKeys[p]] = pathValue == null ? null : (lastHist + toFiniteNumber(pathValue, 0));
            }

            rows.push(row);
        }
    }

    return {
        rows,
        pathKeys,
        currentIndex,
        lastIndex: rows[rows.length - 1].x,
    };
}

function CustomTooltip({ active, payload, label }) {
    if (!active || !payload || payload.length === 0) return null;

    const visible = payload.filter((item) => (
        ['portfolio', 'sp500', 'forecast'].includes(item.dataKey) && item.value != null
    ));

    if (!visible.length) return null;

    const fullDate = payload?.[0]?.payload?.fullDate || label;

    return (
        <div
            style={{
                background: 'rgba(255,255,255,0.96)',
                border: '1px solid rgba(148,163,184,0.35)',
                borderRadius: 12,
                padding: '8px 10px',
                boxShadow: '0 10px 24px rgba(15, 23, 42, 0.12)',
            }}
        >
            <p style={{ margin: 0, fontSize: 11, color: '#64748b' }}>{fullDate}</p>
            {visible.map((entry) => (
                <p
                    key={`${entry.name}-${entry.dataKey}`}
                    style={{
                        margin: '4px 0 0',
                        fontSize: 12,
                        fontWeight: 700,
                        color: entry.color,
                    }}
                >
                    {entry.name}: {formatSigned(entry.value)}
                </p>
            ))}
        </div>
    );
}

/* ── 커스텀 범례 (색상 원 + 텍스트) ── */
const LEGEND_ITEMS = [
    { color: '#2563eb', label: '포트폴리오 (과거)' },
    { color: '#94a3b8', label: 'S&P 500 벤치마크' },
    { color: '#1e3a8a', label: '기대 경로 (예측)' },
    { color: 'rgba(148,163,184,0.35)', label: 'MC 시뮬레이션 경로' },
];

function CumulativeReturnChart({ chartData, forecastData, warnings = [] }) {
    const series = useMemo(
        () => buildSeries(chartData, forecastData),
        [chartData, forecastData],
    );

    const distribution = forecastData?.final_distribution || null;

    if (!series.rows.length) {
        return <div className="chart-fallback">차트 데이터를 불러오는 중입니다.</div>;
    }

    const xTicks = series.lastIndex > series.currentIndex
        ? [0, series.currentIndex, series.lastIndex]
        : [0, series.currentIndex];

    const xTickFormatter = (xValue) => {
        if (xValue === 0) return '과거 1년';
        if (xValue === series.currentIndex) return '현재(t)';
        if (xValue === series.lastIndex && series.lastIndex !== series.currentIndex) return '+3개월';
        return '';
    };

    return (
        <div>
            {warnings.map((warning) => (
                <div className="chart-warning" key={warning}>
                    {warning}
                </div>
            ))}

            <div className="premium-chart-unified">
                <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={series.rows} margin={{ top: 8, right: 80, left: -20, bottom: 4 }}>
                        <defs>
                            <linearGradient id="portfolioAreaFill" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.25" />
                                <stop offset="100%" stopColor="#60a5fa" stopOpacity="0.02" />
                            </linearGradient>
                        </defs>

                        <CartesianGrid strokeDasharray="4 4" stroke="#cbd5e1" />

                        <XAxis
                            dataKey="x"
                            type="number"
                            domain={[0, series.lastIndex]}
                            ticks={xTicks}
                            tickFormatter={xTickFormatter}
                            tick={{ fill: '#64748b', fontSize: 11 }}
                            axisLine={{ stroke: '#cbd5e1' }}
                            tickLine={false}
                        />

                        <YAxis
                            tick={{ fill: '#64748b', fontSize: 11 }}
                            tickFormatter={(value) => `${value > 0 ? '+' : ''}${toFiniteNumber(value, 0).toFixed(0)}%`}
                            axisLine={{ stroke: '#cbd5e1' }}
                            tickLine={false}
                        />

                        <Tooltip content={<CustomTooltip />} labelFormatter={(value) => xTickFormatter(value)} />

                        <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="3 3" />
                        <ReferenceLine
                            x={series.currentIndex}
                            stroke="#475569"
                            strokeDasharray="3 3"
                            strokeOpacity={0.7}
                        />

                        <Area
                            type="monotone"
                            dataKey="portfolio"
                            stroke="none"
                            fill="url(#portfolioAreaFill)"
                            connectNulls={false}
                            isAnimationActive={false}
                        />

                        <Line
                            type="monotone"
                            dataKey="sp500"
                            name="S&P 500"
                            stroke="#94a3b8"
                            strokeWidth={1.6}
                            dot={false}
                            isAnimationActive={false}
                        />

                        <Line
                            type="monotone"
                            dataKey="portfolio"
                            name="포트폴리오(과거)"
                            stroke="#2563eb"
                            strokeWidth={2.35}
                            dot={false}
                            isAnimationActive={false}
                        />

                        {series.pathKeys.map((pathKey) => (
                            <Line
                                key={pathKey}
                                type="monotone"
                                dataKey={pathKey}
                                stroke="#94a3b8"
                                strokeOpacity={0.24}
                                strokeWidth={1}
                                dot={false}
                                connectNulls
                                isAnimationActive={false}
                                legendType="none"
                            />
                        ))}

                        <Line
                            type="monotone"
                            dataKey="forecast"
                            name="기대 경로"
                            stroke="#1e3a8a"
                            strokeWidth={2.3}
                            dot={false}
                            connectNulls
                            isAnimationActive={false}
                        />

                        {/* 정규분포를 시뮬레이션 끝에 이어 붙이기 */}
                        {distribution && (
                            <Customized
                                component={(props) => (
                                    <DistributionOverlay {...props} distribution={distribution} />
                                )}
                            />
                        )}
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            {/* 커스텀 범례: 색상 원 + 텍스트 */}
            <div className="chart-legend">
                {LEGEND_ITEMS.map((item) => (
                    <div className="chart-legend-item" key={item.label}>
                        <span className="chart-legend-dot" style={{ backgroundColor: item.color }} />
                        <span>{item.label}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default CumulativeReturnChart;

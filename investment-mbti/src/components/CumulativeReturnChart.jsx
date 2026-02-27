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

function buildGaussianShape(distribution) {
    if (!distribution) return null;

    const mean = toFiniteNumber(distribution.mean, 0);
    const std = Math.max(0.001, Math.abs(toFiniteNumber(distribution.std, 0.01)));
    const minValue = mean - (std * 3);
    const maxValue = mean + (std * 3);

    const samples = 68;
    const plotTop = 10;
    const plotHeight = 196;
    const baseX = 16;
    const curveWidth = 62;

    const points = [];
    let peakDensity = Number.NEGATIVE_INFINITY;

    for (let i = 0; i <= samples; i += 1) {
        const t = i / samples;
        const value = maxValue - ((maxValue - minValue) * t);
        const z = (value - mean) / std;
        const density = Math.exp(-0.5 * z * z);
        points.push({ t, density });
        if (density > peakDensity) peakDensity = density;
    }

    const curvePoints = points.map(({ t, density }) => {
        const y = plotTop + (plotHeight * t);
        const x = baseX + ((density / peakDensity) * curveWidth);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    });

    const areaPath = `M ${baseX},${plotTop} L ${curvePoints.join(' L ')} L ${baseX},${plotTop + plotHeight} Z`;
    const linePath = `M ${curvePoints.join(' L ')}`;
    const meanT = (maxValue - mean) / (maxValue - minValue);
    const meanY = plotTop + (plotHeight * meanT);

    return {
        areaPath,
        linePath,
        meanY,
    };
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

function CumulativeReturnChart({ chartData, forecastData, warnings = [] }) {
    const series = useMemo(
        () => buildSeries(chartData, forecastData),
        [chartData, forecastData],
    );

    const distributionShape = useMemo(
        () => buildGaussianShape(forecastData?.final_distribution),
        [forecastData],
    );

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

    const summary = forecastData?.final_distribution || null;

    return (
        <div>
            {warnings.map((warning) => (
                <div className="chart-warning" key={warning}>
                    {warning}
                </div>
            ))}

            <div className="premium-chart-shell">
                <div className="premium-chart-main">
                    <ResponsiveContainer width="100%" height={300}>
                        <ComposedChart data={series.rows} margin={{ top: 8, right: 10, left: -20, bottom: 4 }}>
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
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>

                <aside className="premium-distribution" aria-label="시뮬레이션 분포">
                    {distributionShape ? (
                        <svg viewBox="0 0 96 220">
                            <defs>
                                <linearGradient id="distFill" x1="0" y1="0" x2="1" y2="1">
                                    <stop offset="0%" stopColor="#93c5fd" stopOpacity="0.5" />
                                    <stop offset="100%" stopColor="#2563eb" stopOpacity="0.12" />
                                </linearGradient>
                            </defs>

                            <line x1="16" y1="10" x2="16" y2="206" stroke="#cbd5e1" strokeWidth="1.1" />
                            <path d={distributionShape.areaPath} fill="url(#distFill)" />
                            <path d={distributionShape.linePath} fill="none" stroke="#2563eb" strokeWidth="2.2" />
                            <line x1="16" y1={distributionShape.meanY} x2="80" y2={distributionShape.meanY} stroke="#1e3a8a" strokeDasharray="3 3" />
                        </svg>
                    ) : (
                        <svg viewBox="0 0 96 220">
                            <line x1="16" y1="10" x2="16" y2="206" stroke="#cbd5e1" strokeWidth="1.1" />
                            <path d="M 16 108 Q 56 30 72 108 Q 56 186 16 108" fill="rgba(147,197,253,0.32)" stroke="#2563eb" strokeWidth="2.2" />
                        </svg>
                    )}
                </aside>
            </div>

            <div className="chart-footnote">
                <span>포트폴리오(파랑) / 벤치마크(회색) / 기대 경로(진청)</span>
                {summary ? (
                    <span>
                        평균 {formatSigned(summary.mean)} · 5%ile {formatSigned(summary.percentile_5)} · 95%ile {formatSigned(summary.percentile_95)}
                    </span>
                ) : (
                    <span>시뮬레이션 요약 준비 중</span>
                )}
            </div>
        </div>
    );
}

export default CumulativeReturnChart;

import React, { useMemo } from 'react';
import {
    Area,
    AreaChart,
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

/* ── 유틸 ── */
function toFiniteNumber(value, fallback = 0) {
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

function formatSigned(value) {
    const n = toFiniteNumber(value, 0);
    return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
}

/* ── 시계열 데이터 빌드 ── */
function buildSeries(chartData, forecastData) {
    const historicalRows = (chartData?.dates || []).map((date, idx) => ({
        x: idx,
        fullDate: date,
        portfolio: toFiniteNumber(chartData?.portfolio?.[idx], 0),
        sp500: toFiniteNumber(chartData?.sp500?.[idx], 0),
    }));

    if (!historicalRows.length) {
        return { rows: [], pathKeys: [], currentIndex: 0, lastIndex: 0, lastHist: 0 };
    }

    const currentIndex = historicalRows.length - 1;
    const rows = [...historicalRows];
    const hasForecast = Array.isArray(forecastData?.forecast_days) && forecastData.forecast_days.length > 1;
    const pathCount = Math.min(Array.isArray(forecastData?.mc_paths) ? forecastData.mc_paths.length : 0, 100);
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

    return { rows, pathKeys, currentIndex, lastIndex: rows[rows.length - 1].x, lastHist };
}

/* ── 커스텀 Tooltip ── */
function CustomTooltip({ active, payload, label }) {
    if (!active || !payload || payload.length === 0) return null;

    const visible = payload.filter((item) => (
        ['portfolio', 'sp500', 'forecast'].includes(item.dataKey) && item.value != null
        && item.name !== 'portfolio'
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

/* ── 커스텀 범례 ── */
const LEGEND_ITEMS = [
    { color: '#3B82F6', label: '포트폴리오 (과거)' },
    { color: '#F59E0B', label: 'S&P 500 벤치마크' },
    { color: '#8b5cf6', label: '기대 경로 (예측)' },
    { color: '#94a3b8', label: 'MC 시뮬레이션 경로' },
];

/* ── 메인 차트 컴포넌트 ── */
function CumulativeReturnChart({ chartData, forecastData, warnings = [] }) {
    const series = useMemo(
        () => buildSeries(chartData, forecastData),
        [chartData, forecastData],
    );

    // 분포 데이터 (Flexbox 오른쪽 vertical AreaChart용)
    const distributionBins = forecastData?.distribution_bins || [];
    const var5Value = forecastData?.var_5_value ?? forecastData?.final_distribution?.percentile_5 ?? null;
    const lastHist = series.lastHist || 0;

    // distribution bins에 lastHist baseline 더해서 Y축 동기화
    const adjustedBins = useMemo(() => {
        return distributionBins.map(bin => ({
            returnBin: bin.returnBin + lastHist,
            frequency: bin.frequency,
        }));
    }, [distributionBins, lastHist]);
    const adjustedVar5 = var5Value != null ? var5Value + lastHist : null;

    // Y축 글로벌 minmax (두 차트 동기화)
    const { globalMin, globalMax } = useMemo(() => {
        let min = Infinity;
        let max = -Infinity;
        for (const row of series.rows) {
            const vals = [row.portfolio, row.sp500, row.forecast];
            // MC paths
            for (const k of series.pathKeys) {
                if (row[k] != null) vals.push(row[k]);
            }
            for (const v of vals) {
                if (v != null && Number.isFinite(v)) {
                    if (v < min) min = v;
                    if (v > max) max = v;
                }
            }
        }
        // distribution bins도 포함
        for (const b of adjustedBins) {
            if (b.returnBin < min) min = b.returnBin;
            if (b.returnBin > max) max = b.returnBin;
        }
        const pad = Math.max((max - min) * 0.08, 2);
        return {
            globalMin: Math.floor(min - pad),
            globalMax: Math.ceil(max + pad),
        };
    }, [series, adjustedBins]);

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

    const hasDistribution = adjustedBins.length > 0;

    const var5Color = '#ef4444'; // 위험도 바와 톤온톤 매칭

    // 분산 차트 VaR 5% 전용 Tooltip
    const CustomDistTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            return (
                <div style={{
                    background: 'rgba(255,255,255,0.95)',
                    border: '1px solid #e2e8f0',
                    borderRadius: '6px',
                    padding: '8px 12px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    fontSize: '0.75rem',
                    color: '#334155',
                    zIndex: 100,
                }}>
                    <div style={{ fontWeight: 600, color: '#0f172a', marginBottom: '4px' }}>
                        Value at Risk (5%)
                    </div>
                    <div>손실 수준: <strong style={{ color: var5Color }}>{adjustedVar5?.toFixed(2)}%</strong></div>
                    <div style={{ marginTop: '6px', fontSize: '0.7rem', color: '#64748b', maxWidth: '180px', whiteSpace: 'normal', lineHeight: 1.4 }}>
                        100일 중 가장 운이 나쁜 5일이 찾아왔을 때, <strong>'최소한 이만큼은 잃을 수 있다'</strong>고 각오해야 하는 손실의 마지노선
                    </div>
                </div>
            );
        }
        return null;
    };

    return (
        <div>
            {warnings.map((warning) => (
                <div className="chart-warning" key={warning}>
                    {warning}
                </div>
            ))}

            <div style={{ display: 'flex', width: '100%', height: 310, alignItems: 'stretch' }}>
                {/* ── 메인 차트 (좌측) ── */}
                <div style={{ flex: hasDistribution ? '0 0 84%' : '1 1 100%', height: '100%', position: 'relative' }}>

                    {/* 상단 뱃지 (현재(t) 기준 정렬) */}
                    <div style={{
                        position: 'absolute',
                        top: '-32px',
                        right: '0',
                        fontSize: '0.7rem',
                        fontWeight: '600',
                        color: '#6366f1',
                        backgroundColor: '#e0e7ff',
                        padding: '4px 10px',
                        borderRadius: '99px',
                        zIndex: 10,
                    }}>
                        과거 1년 + 향후 3개월
                    </div>
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={series.rows} margin={{ top: 8, right: 0, left: -20, bottom: 4 }}>
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
                                domain={[globalMin, globalMax]}
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
                                tooltipType="none"
                            />

                            <Line
                                type="monotone"
                                dataKey="sp500"
                                name="S&P 500"
                                stroke="#F59E0B"
                                strokeWidth={1.6}
                                dot={false}
                                isAnimationActive={false}
                            />

                            <Line
                                type="monotone"
                                dataKey="portfolio"
                                name="포트폴리오(과거)"
                                stroke="#3B82F6"
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
                                    strokeOpacity={0.18}
                                    strokeWidth={0.8}
                                    dot={false}
                                    activeDot={false}
                                    connectNulls
                                    isAnimationActive={false}
                                    legendType="none"
                                />
                            ))}

                            <Line
                                type="monotone"
                                dataKey="forecast"
                                name="기대 경로"
                                stroke="#8b5cf6"
                                strokeWidth={2.5}
                                dot={false}
                                connectNulls
                                isAnimationActive={false}
                            />
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>

                {/* ── 확률 분포 차트 (우측, Vertical AreaChart) ── */}
                {hasDistribution && (
                    <div style={{ flex: '0 0 16%', height: '100%', marginLeft: -1 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart
                                layout="vertical"
                                data={adjustedBins}
                                margin={{ top: 8, right: 12, bottom: 4, left: 0 }}
                            >
                                {/* X축 (빈도수) — 숨김 */}
                                <XAxis type="number" hide />

                                {/* Y축 (수익률) — 메인 차트와 동일 domain, 숨김 */}
                                <YAxis
                                    type="number"
                                    dataKey="returnBin"
                                    domain={[globalMin, globalMax]}
                                    hide
                                    reversed
                                />

                                {/* 부드러운 확률 밀도 곡선 (보라색 계열) */}
                                <defs>
                                    <linearGradient id="distFillGrad" x1="0" y1="0" x2="1" y2="0">
                                        <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.05} />
                                        <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.4} />
                                    </linearGradient>
                                </defs>
                                <Area
                                    dataKey="frequency"
                                    type="basis"
                                    stroke="none"
                                    fill="url(#distFillGrad)"
                                    isAnimationActive={false}
                                    activeDot={false} /* 마우스 오버 시 생기는 점 제거 */
                                />

                                {/* VaR 5% Hover Tooltip (cursor 숨김) */}
                                <Tooltip
                                    content={<CustomDistTooltip />}
                                    cursor={false}
                                />

                                {/* VaR 5% 기준선 (빨간색 계열 매칭) */}
                                {adjustedVar5 != null && (
                                    <ReferenceLine
                                        y={adjustedVar5}
                                        stroke={var5Color}
                                        strokeDasharray="5 3"
                                        strokeWidth={1.5}
                                        ifOverflow="visible"
                                    />
                                )}
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                )}

                {/* 3개월 뒤 확률 분포 레이블 - 전체 우측 최상단에 고정 */}
                {hasDistribution && (
                    <div style={{
                        position: 'absolute',
                        top: '-30px', /* padding 조정 */
                        right: '0',
                        fontSize: '0.75rem',
                        fontWeight: '600',
                        color: '#64748b',
                        padding: '4px 8px',
                        zIndex: 10,
                    }}>
                        3개월 뒤 예상 수익률 확률 분포
                    </div>
                )}
            </div>

            {/* 커스텀 범례 */}
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

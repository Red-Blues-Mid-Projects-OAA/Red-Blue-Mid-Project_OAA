import React from 'react';

/**
 * 읽기 전용 종목 카드 컴포넌트.
 * 종목명, 티커, 위험도 점수(0~100), 투자 비중, 그리고 4개 기간별 누적수익률을 표시합니다.
 * 체크박스/선택 기능 없음 — 순수 정보 표시용.
 */
function StockCard({ stock }) {
    // 기간별 수익률 데이터 (단순수익률 %)
    const hist = stock.historical_returns || { "1M": 0, "3M": 0, "6M": 0, "12M": 0 };
    const periods = [
        { label: "1M", value: hist["1M"] },
        { label: "3M", value: hist["3M"] },
        { label: "6M", value: hist["6M"] },
        { label: "12M", value: hist["12M"] },
    ];

    // 위험도 점수에 따른 색상 결정
    const getRiskColor = (score) => {
        if (score <= 25) return "#10b981";  // 낮음 (녹색)
        if (score <= 50) return "#3b82f6";  // 중간 (파란색)
        if (score <= 75) return "#f59e0b";  // 높음 (노란색)
        return "#ef4444";                   // 매우 높음 (빨간색)
    };

    // 수익률 양/음에 따른 색상
    const getReturnColor = (val) => val >= 0 ? "#10b981" : "#ef4444";
    const getReturnPrefix = (val) => val >= 0 ? "+" : "";

    const riskScore = stock.risk_score ?? 0;
    const riskColor = getRiskColor(riskScore);

    return (
        <div className="bg-gray-800/60 border border-gray-700/50 rounded-2xl p-4 hover:border-gray-600 transition-all duration-200">
            {/* 상단: 티커 + 종목명 + 위험도 점수 */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <span className="text-lg font-bold text-white">{stock.ticker}</span>
                        {stock.weight > 0 && (
                            <span className="text-xs bg-violet-500/20 text-violet-400 px-2 py-0.5 rounded-full font-semibold">
                                {stock.weight.toFixed(1)}%
                            </span>
                        )}
                    </div>
                    <span className="text-sm text-gray-400 block truncate">{stock.name}</span>
                </div>
                {/* 위험도 점수 배지 */}
                <div className="flex flex-col items-center ml-2 shrink-0">
                    <span className="text-xs text-gray-500 mb-0.5">위험도</span>
                    <span
                        className="text-lg font-black rounded-lg px-2 py-0.5"
                        style={{ color: riskColor, backgroundColor: `${riskColor}15` }}
                    >
                        {riskScore}
                    </span>
                </div>
            </div>

            {/* 하단 4분할: 1M / 3M / 6M / 12M 수익률 */}
            <div className="grid grid-cols-4 gap-1">
                {periods.map((p) => (
                    <div key={p.label} className="text-center bg-gray-900/50 rounded-lg py-1.5 px-1">
                        <span className="text-[10px] text-gray-500 block">{p.label}</span>
                        <span
                            className="text-xs font-bold block"
                            style={{ color: getReturnColor(p.value) }}
                        >
                            {getReturnPrefix(p.value)}{p.value.toFixed(1)}%
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default StockCard;

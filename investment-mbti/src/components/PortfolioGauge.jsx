/*
 * 이 파일은 포트폴리오 게이지 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React from 'react';

/**
 * 포트폴리오 게이지 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function PortfolioGauge({ value, label, minText, maxText, colorClass }) {
    // value represents the percentage width (0-100)

    return (
        <div className="flex flex-col gap-2 mb-6">
            <div className="flex justify-between items-end mb-1">
                <span className="font-bold text-lg text-white">{label}</span>
                <span className={`text-xl font-bold ${colorClass.text}`}>{value.toFixed(1)} / 10.0</span>
            </div>

            <div className="relative w-full h-8 bg-[#482861] rounded-full border-2 border-[#ba8fce] shadow-inner p-1">
                <div
                    className="absolute top-1 bottom-1 left-1 rounded-full transition-all duration-500 ease-out bg-gradient-to-r from-[#ffe28a] to-[#f9a826] shadow-[0_0_12px_rgba(249,168,38,0.6)]"
                    style={{ width: `calc(${(value / 10) * 100}% - 8px)` }}
                ></div>
            </div>

            <div className="flex justify-between text-xs text-gray-400 font-medium px-1">
                <span>{minText}</span>
                <span>{maxText}</span>
            </div>
        </div>
    );
}

export default PortfolioGauge;

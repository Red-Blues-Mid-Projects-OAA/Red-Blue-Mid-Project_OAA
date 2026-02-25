import React, { useEffect, useState } from 'react';

function MbtiBarChart({ dimension, score, leftLabel, rightLabel, colorClass }) {
    const [animatedScore, setAnimatedScore] = useState(50); // Start animation from center

    useEffect(() => {
        // 애니메이션 효과: 중앙(50)에서 실제 점수로 부드럽게 이동
        const timer = setTimeout(() => {
            setAnimatedScore(score);
        }, 150);
        return () => clearTimeout(timer);
    }, [score]);

    return (
        <div className="w-full mb-8">
            {/* 상단 텍스트 및 라벨 영역 */}
            <div className="flex justify-between items-end text-sm font-bold text-gray-400 px-2 mb-3">
                <span className={`transition-colors duration-500 ${animatedScore < 50 ? 'text-white drop-shadow-[0_0_5px_rgba(255,255,255,0.5)]' : ''}`}>
                    {leftLabel}
                </span>
                <span className="text-white text-base font-extrabold tracking-widest bg-slate-800 px-4 py-1.5 rounded-xl border border-slate-600 shadow-sm">
                    {dimension}
                </span>
                <span className={`transition-colors duration-500 ${animatedScore > 50 ? 'text-white drop-shadow-[0_0_5px_rgba(255,255,255,0.5)]' : ''}`}>
                    {rightLabel}
                </span>
            </div>

            {/* 차트 배경 및 컬러 바 영역 */}
            <div className="relative w-full h-8 rounded-full shadow-inner border border-slate-600/50 overflow-visible bg-slate-800">
                {/* Full-width gradient track to represent the spectrum */}
                <div className={`absolute inset-0 opacity-80 rounded-full ${colorClass}`}></div>

                {/* 중앙 눈금선 (기준점) */}
                <div className="absolute left-1/2 top-0 bottom-0 w-[2px] bg-white/30 z-10 hidden md:block"></div>

                {/* 슬라이더 지시자(Thumb) */}
                <div
                    className="absolute top-1/2 -translate-y-1/2 w-5 h-[140%] bg-white rounded-full shadow-[0_0_15px_rgba(0,0,0,0.8)] transition-all duration-1000 ease-out z-20 border-2 border-slate-800 flex justify-center items-center"
                    style={{ left: `calc(${animatedScore}% - 10px)` }}
                >
                    <div className="w-1 h-1/2 bg-slate-400/80 rounded-full"></div>
                </div>
            </div>
        </div>
    );
}

export default MbtiBarChart;

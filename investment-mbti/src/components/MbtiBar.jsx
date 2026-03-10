/*
 * 이 파일은 MBTI 막대 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React from 'react';

/**
 * MBTI 막대 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function MbtiBar({ mbtiType }) {
    // mbtiType ex: "ENTJ"
    // 1: E/I
    const isE = mbtiType[0] === 'E';
    // 2: N/S
    const isN = mbtiType[1] === 'N';
    // 3: T/F
    const isT = mbtiType[2] === 'T';
    // 4: J/P
    const isJ = mbtiType[3] === 'J';

    const bars = [
        {
            title: "시장 반응",
            leftLabel: "내향",
            rightLabel: "외향",
            isRightSelected: isE, // E면 오른쪽(외향), I면 왼쪽(내향)
        },
        {
            title: "가치 판단",
            leftLabel: "직관",
            rightLabel: "감각",
            isRightSelected: !isN, // S면 오른쪽(감각), N이면 왼쪽(직관)
        },
        {
            title: "의사 결정",
            leftLabel: "감정",
            rightLabel: "사고",
            isRightSelected: isT, // T면 오른쪽(사고), F면 왼쪽(감정)
        },
        {
            title: "대응",
            leftLabel: "유연",
            rightLabel: "계획",
            isRightSelected: isJ, // J면 오른쪽(계획), P면 왼쪽(유연) 
        }
    ];

    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
    return (
        <div className="flex flex-col gap-6 w-full max-w-md mx-auto">
            {bars.map((bar, idx) => (
                <div key={idx} className="relative flex items-center justify-between w-full h-12 bg-[#482861] rounded-full border-2 border-[#ba8fce] shadow-inner p-1 px-5">

                    {/* Background Gradient Bar */}
                    <div className={`absolute top-1 bottom-1 ${bar.isRightSelected ? 'right-1' : 'left-1'} w-[calc(50%-4px)] rounded-full bg-gradient-to-r from-[#ffe28a] to-[#f9a826] shadow-[0_0_12px_rgba(249,168,38,0.6)]`}></div>

                    {/* Left Label */}
                    <span className={`z-10 text-sm font-bold ${!bar.isRightSelected ? 'text-gray-900 drop-shadow-md' : 'text-gray-400'}`}>
                        {bar.leftLabel}
                    </span>

                    {/* Center Title */}
                    <span className="z-10 text-xs text-gray-300 absolute left-1/2 -translate-x-1/2 bg-[#361e4a] px-3 py-1 rounded-md shadow-lg border border-[#ba8fce]/30">
                        {bar.title}
                    </span>

                    {/* Right Label */}
                    <span className={`z-10 text-sm font-bold ${bar.isRightSelected ? 'text-gray-900 drop-shadow-md' : 'text-gray-400'}`}>
                        {bar.rightLabel}
                    </span>

                    {/* Marker */}
                    <div
                        className={`absolute top-1/2 -translate-y-1/2 w-8 h-8 bg-white/20 backdrop-blur-sm rounded-full border-2 border-white flex items-center justify-center shadow-lg transition-all duration-700 ease-out z-20`}
                        style={{
                            left: bar.isRightSelected ? 'calc(100% - 2.5rem)' : '0.5rem'
                        }}
                    >
                        <span className="text-white text-sm font-bold drop-shadow-md">$</span>
                    </div>

                </div>
            ))}
        </div>
    );
}

export default MbtiBar;

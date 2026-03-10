/*
 * 이 파일은 종목 목록 한 줄 항목 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React from 'react';

/**
 * 종목 목록 항목 컴포넌트가 화면 표시를 담당합니다.
 */

function StockListItem({ stock, isSelected, onToggle }) {
    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
    return (
        <label
            className={`stock-item flex items-center p-5 rounded-xl border transition-all duration-300 cursor-pointer group
            ${isSelected ? 'bg-violet-900/30 border-violet-500/50 shadow-[0_0_15px_rgba(139,92,246,0.15)]' : 'bg-gray-800/40 border-gray-700 hover:bg-gray-800'}
        `}
        >
            <div className="shrink-0 flex items-center pr-4">
                <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => onToggle(stock.ticker)}
                    className="w-5 h-5 accent-violet-500 cursor-pointer"
                />
            </div>

            <div className="stock-info text-left flex-grow">
                <div className="flex justify-between items-center mb-1">
                    <span className={`text-lg font-bold block transition-colors ${isSelected ? 'text-violet-400' : 'text-gray-200 group-hover:text-blue-400'}`}>
                        {stock.ticker}
                    </span>
                </div>
                <span className="stock-name text-sm text-gray-400 block">{stock.name}</span>
            </div>
        </label>
    );
}

export default StockListItem;

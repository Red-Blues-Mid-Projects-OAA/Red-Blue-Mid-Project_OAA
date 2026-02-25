import React from 'react';

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

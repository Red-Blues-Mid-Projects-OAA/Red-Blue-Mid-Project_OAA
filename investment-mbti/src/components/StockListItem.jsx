import React from 'react';

function StockListItem({ stock, isSelected, onToggle }) {
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

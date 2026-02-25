import React, { useState, useEffect } from 'react';
import PortfolioGauge from './PortfolioGauge';
import StockListItem from './StockListItem';

function Recommendation({ portfolioData, onRestart }) {
    // 1. 상태: 선택된 종목의 ticker들을 Set이나 배열로 관리
    const [selectedStocks, setSelectedStocks] = useState([]);

    // 2. 초기화: 처음 로드 시 상위 3개~4개 종목을 기본 체크 상태로 설정
    useEffect(() => {
        if (portfolioData && portfolioData.length > 0) {
            const initialSelected = portfolioData.slice(0, 4).map(s => s.ticker);
            setSelectedStocks(initialSelected);
        }
    }, [portfolioData]);

    // 3. 토글 핸들러
    const handleToggle = (ticker) => {
        setSelectedStocks(prev =>
            prev.includes(ticker)
                ? prev.filter(t => t !== ticker)
                : [...prev, ticker]
        );
    };

    // 4. 동적 계산 (위험성 & 수익성 스케일 1-10)
    // 수익률(expectedReturn3M) 기준:
    // - 보통 1~3% (Low)
    // - 4~8% (Medium)
    // - 10~20%+ (High)
    // 위험성(Volatility) 기준: Low(2), Medium(5), High(9) 등으로 매핑

    let avgReturnDisplay = 0;
    let avgRiskDisplay = 0;

    if (selectedStocks.length > 0) {
        const activeStocks = portfolioData.filter(s => selectedStocks.includes(s.ticker));

        const totalReturn = activeStocks.reduce((sum, s) => sum + (s.expectedReturn3M || 0), 0);
        const avgReturn = totalReturn / activeStocks.length;

        // 수익성을 1~10 스케일로 단순 매핑 (예: 20% 수익률을 10점으로 가정)
        avgReturnDisplay = Math.min((avgReturn / 20) * 10, 10);
        // 최소치 보정
        if (avgReturnDisplay < 1.0) avgReturnDisplay = 1.0 + (avgReturnDisplay / 2);

        const volScoreMap = { 'Very Low': 1, 'Low': 3, 'Medium': 6, 'High': 9 };
        const totalRisk = activeStocks.reduce((sum, s) => sum + (volScoreMap[s.volatility] || 5), 0);
        avgRiskDisplay = totalRisk / activeStocks.length;
    }

    return (
        <div className="recommendation-container animate-fade-in w-full max-w-2xl mx-auto py-12 px-4">

            <div className="text-center mb-10 animate-slide-up">
                <h2 className="text-3xl font-bold text-white mb-2">맞춤형 종목 포트폴리오</h2>
                <p className="text-gray-400">선택한 종목에 따라 위험도와 기대수익률이 실시간으로 변합니다.</p>
            </div>

            {/* 상단: Portfolio Analysis Gauges */}
            <div className="bg-gray-900/60 p-6 md:p-8 rounded-3xl border border-gray-700/50 backdrop-blur-md mb-8 shadow-xl animate-slide-up" style={{ animationDelay: '0.2s' }}>
                <PortfolioGauge
                    label="포트폴리오 위험성"
                    value={avgRiskDisplay}
                    minText="매우 안전 안정 추구"
                    maxText="고위험 초과수익 추구"
                    colorClass={{ text: 'text-red-400', bg: 'from-orange-500 to-red-500' }}
                />

                <PortfolioGauge
                    label="포트폴리오 수익성"
                    value={avgReturnDisplay}
                    minText="안정적인 예적금 수준"
                    maxText="시장 수익률 초과 알파"
                    colorClass={{ text: 'text-green-400', bg: 'from-emerald-500 to-green-400' }}
                />
            </div>

            {/* 하단: Stock List */}
            <div className="stock-list-container animate-slide-up" style={{ animationDelay: '0.4s' }}>
                <div className="flex justify-between items-center mb-4 px-2">
                    <h3 className="text-xl font-bold text-white">추천 종목 리스트</h3>
                    <span className="text-sm text-gray-400">{selectedStocks.length}개 선택됨</span>
                </div>

                <div className="flex flex-col gap-3">
                    {portfolioData.map((stock) => (
                        <StockListItem
                            key={stock.ticker}
                            stock={stock}
                            isSelected={selectedStocks.includes(stock.ticker)}
                            onToggle={handleToggle}
                        />
                    ))}

                    {portfolioData.length === 0 && (
                        <div className="py-12 text-center text-gray-400 bg-gray-800/20 rounded-xl border border-dashed border-gray-700">
                            추천받은 종목 데이터가 없습니다.
                        </div>
                    )}
                </div>
            </div>

            {/* 하단 액션 버튼 */}
            <div className="action-buttons mt-12 animate-slide-up" style={{ animationDelay: '0.6s' }}>
                <button
                    className="w-full py-4 rounded-full bg-gray-800 hover:bg-gray-700 text-white font-bold text-lg transition-colors shadow-lg"
                    onClick={onRestart}
                >
                    메인으로 돌아가기 🏠
                </button>
            </div>
        </div>
    );
}

export default Recommendation;

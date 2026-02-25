import React, { useState, useEffect } from 'react';
import MbtiBarChart from './MbtiBarChart';
import PortfolioGauge from './PortfolioGauge';
import StockListItem from './StockListItem';
import './Result.css';

function DashboardResult({ personaData, onRestart }) {
    if (!personaData) return null;

    const recommendedStocks = personaData.recommendedStocks || [];
    const mbtiType = personaData.features?.[0]?.split(': ')[1] || 'ENTJ';
    const rawAnswers = personaData.rawAnswers || [];

    // 1. 선택된 주식 종목 (체크박스 포트폴리오 산정용)
    const [selectedStocks, setSelectedStocks] = useState([]);

    // 2. 주석 표시를 위해 포커스(클릭)된 단일 주식
    const [focusedStock, setFocusedStock] = useState(null);

    // 초기화: 추천 종목 상위 4개 기본 선택 및 1위 종목 포커스
    useEffect(() => {
        if (recommendedStocks.length > 0) {
            const initialSelected = recommendedStocks.slice(0, 4).map(s => s.ticker);
            setSelectedStocks(initialSelected);
            setFocusedStock(recommendedStocks[0]);
        }
    }, [recommendedStocks]);

    // MBTI 점수 계산식 (Area 1)
    const scoreI = rawAnswers.length > 0 ? (rawAnswers.slice(0, 3).filter(a => a === 'A').length / 3) * 100 : 50;
    const scoreN = rawAnswers.length > 0 ? (rawAnswers.slice(3, 6).filter(a => a === 'B').length / 3) * 100 : 50;
    const scoreF = rawAnswers.length > 0 ? (rawAnswers.slice(6, 9).filter(a => a === 'B').length / 3) * 100 : 50;
    const scoreP = rawAnswers.length > 0 ? (rawAnswers.slice(9, 12).filter(a => a === 'B').length / 3) * 100 : 50;

    // 종목 체크박스 토글 핸들러
    const handleToggle = (ticker, stockObj) => {
        setSelectedStocks(prev =>
            prev.includes(ticker)
                ? prev.filter(t => t !== ticker)
                : [...prev, ticker]
        );
        // 체크박스 클릭 시 포커스도 해당 종목으로 이동
        setFocusedStock(stockObj);
    };

    // 포트폴리오(Area 2) 실시간 계산
    let avgReturnDisplay = 0;
    let avgRiskDisplay = 0;

    if (selectedStocks.length > 0) {
        const activeStocks = recommendedStocks.filter(s => selectedStocks.includes(s.ticker));
        const totalReturn = activeStocks.reduce((sum, s) => sum + (s.expectedReturn3M || 0), 0);
        const avgReturn = totalReturn / activeStocks.length;

        avgReturnDisplay = Math.min((avgReturn / 20) * 10, 10);
        if (avgReturnDisplay < 1.0) avgReturnDisplay = 1.0 + (avgReturnDisplay / 2);

        const volScoreMap = { 'Very Low': 1, 'Low': 3, 'Medium': 6, 'High': 9 };
        const totalRisk = activeStocks.reduce((sum, s) => sum + (volScoreMap[s.volatility] || 5), 0);
        avgRiskDisplay = totalRisk / activeStocks.length;
    }

    // 주석 코멘트 생성기 (Area 4)
    const generateCommentary = (stock) => {
        if (!stock) return "좌측 리스트에서 종목을 선택하시면 상세 투자 포인트가 표시됩니다.";

        let riskComment = "";
        if (stock.volatility === "High") riskComment = "시장 변동성이 높은 편이므로 단기적인 손실 리스크에 유의해야 하나, 폭발적인 상승 모멘텀을 기대할 수 있습니다.";
        else if (stock.volatility === "Medium") riskComment = "상대적으로 안정적인 변동성을 가지며, 균형 잡힌 포트폴리오 구성에 적합한 밸런스형 종목입니다.";
        else riskComment = "시장 충격에 강한 방어주 성격을 띠며, 예적금 대신 안전하게 자산을 파킹하기에 매우 유리한 종목입니다.";

        return `선택하신 [${stock.name}] 종목은 3개월 기대 수익률이 약 ${stock.expectedReturn3M}%로 평가되고 있습니다. ${riskComment} ${mbtiType.split(' ')[0]} 성향을 가진 분들의 리스크 허용치와 부합하는 훌륭한 전략적 자산 중 하나입니다.`;
    };

    return (
        <div className="w-full min-h-screen bg-gray-950 px-4 py-8 flex flex-col items-center">
            {/* 상단 타이틀 */}
            <div className="text-center mb-8 animate-fade-in">
                <h1 className="text-3xl lg:text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mb-2">
                    투자 MBTI 통합 분석 리포트
                </h1>
            </div>

            {/* 4분할 데스크톱 대시보드 Grid 컨테이너 */}
            <div className="w-full max-w-[1920px] grid grid-cols-1 lg:grid-cols-12 gap-8">

                {/* =======================================
                    Area 1 (좌측 패널): 페르소나 및 MBTI 차트
                ======================================= */}
                <div className="lg:col-span-4 lg:row-span-2 bg-gray-900/60 border border-gray-700/50 backdrop-blur-md rounded-3xl p-6 lg:p-8 flex flex-col items-center shadow-2xl animate-slide-up">
                    <div className="w-32 h-32 lg:w-40 lg:h-40 mb-6 rounded-full overflow-hidden shadow-[0_0_40px_rgba(139,92,246,0.2)] bg-gray-800 flex-shrink-0" style={{ animation: 'float 6s ease-in-out infinite' }}>
                        <img
                            src={`/images/${personaData.title.includes('거북이') ? 'turtle' : personaData.title.includes('강아지') ? 'dog' : personaData.title.includes('사자') ? 'lion' : 'eagle'}.png`}
                            alt="Persona Avatar"
                            className="w-full h-full object-cover"
                            onError={(e) => { e.target.style.display = 'none'; }}
                        />
                    </div>

                    <div className="text-center mb-6">
                        <p className="text-[#a39dd1] font-semibold text-sm mb-1">팩트로 보는 나는...</p>
                        <h2 className="text-2xl font-bold text-white mb-2">{personaData.title}</h2>
                        <p className="text-sm text-yellow-500 font-bold decoration-wavy underline underline-offset-4 decoration-yellow-500/50">
                            {personaData.description}
                        </p>
                    </div>

                    <div className="bg-gray-800/80 border border-gray-600/50 rounded-2xl p-5 w-full flex flex-col items-center mb-8 shadow-inner">
                        <p className="text-gray-400 font-bold text-xs mb-1">나의 투자 MBTI는?</p>
                        <h3 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-indigo-400 tracking-wider font-mono drop-shadow-lg mb-2">
                            {mbtiType.split(' ')[0]}
                        </h3>
                        {mbtiType.includes(' ') && (
                            <span className="text-xs font-bold text-gray-300 bg-black/40 px-3 py-1 rounded-full border border-gray-600/50">
                                {mbtiType.substring(mbtiType.indexOf(' ')).trim()}
                            </span>
                        )}
                    </div>

                    <div className="w-full flex-grow flex flex-col justify-center">
                        <h4 className="text-sm font-bold text-gray-400 mb-4 text-center">💡 성향 세부 분포 시각화</h4>
                        <div className="flex flex-col gap-2 w-full">
                            <MbtiBarChart dimension="시장 반응" score={scoreI} leftLabel="외향" rightLabel="内향" colorClass="bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500" />
                            <MbtiBarChart dimension="가치 판단" score={scoreN} leftLabel="감각" rightLabel="직관" colorClass="bg-gradient-to-r from-emerald-400 via-green-500 to-teal-500" />
                            <MbtiBarChart dimension="의사 결정" score={scoreF} leftLabel="사고" rightLabel="감정" colorClass="bg-gradient-to-r from-amber-400 via-orange-500 to-red-500" />
                            <MbtiBarChart dimension="대응 방식" score={scoreP} leftLabel="계획형" rightLabel="유연형" colorClass="bg-gradient-to-r from-purple-400 via-fuchsia-500 to-pink-500" />
                        </div>
                    </div>
                </div>

                {/* =======================================
                    Area 2 (우측 상단 패널): 맞춤형 포트폴리오 게이지
                ======================================= */}
                <div className="lg:col-span-8 bg-gray-900/60 border border-gray-700/50 backdrop-blur-md rounded-3xl p-6 lg:p-8 flex flex-col justify-center shadow-2xl animate-slide-up" style={{ animationDelay: '0.2s' }}>
                    <div className="mb-6 flex justify-between items-center bg-gray-800/50 p-4 rounded-xl border border-gray-700">
                        <div>
                            <h3 className="text-xl font-bold text-white mb-1">맞춤형 시각화 포트폴리오</h3>
                            <p className="text-sm text-gray-400">하단 범례에서 종목 체크를 해제하면 위험/수익 게이지가 실시간으로 재설정됩니다.</p>
                        </div>
                        <div className="hidden sm:block text-2xl font-black text-blue-500/30">PORTFOLIO</div>
                    </div>

                    <div className="flex flex-col sm:flex-row gap-6 w-full justify-around items-center h-full">
                        <div className="w-full sm:w-1/2 max-w-sm">
                            <PortfolioGauge
                                label="포트폴리오 위험성"
                                value={avgRiskDisplay}
                                minText="안정 추구"
                                maxText="고위험 추구"
                                colorClass={{ text: 'text-red-400', bg: 'from-orange-500 to-red-500' }}
                            />
                        </div>
                        <div className="w-full sm:w-1/2 max-w-sm mt-8 sm:mt-0">
                            <PortfolioGauge
                                label="포트폴리오 수익성"
                                value={avgReturnDisplay}
                                minText="예적금 보존"
                                maxText="알파 초과수익"
                                colorClass={{ text: 'text-green-400', bg: 'from-emerald-500 to-green-400' }}
                            />
                        </div>
                    </div>
                </div>

                {/* =======================================
                    Area 3 (우측 하단-좌 패널): 추천 종목 리스트
                ======================================= */}
                <div className="lg:col-span-4 bg-gray-900/60 border border-gray-700/50 backdrop-blur-md rounded-3xl p-6 shadow-2xl animate-slide-up flex flex-col" style={{ animationDelay: '0.4s' }}>
                    <h3 className="text-lg font-bold text-white mb-6 flex justify-between items-center">
                        <span>추천 자산 범례</span>
                        <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded-md">{selectedStocks.length}개 선택됨</span>
                    </h3>

                    <div className="flex-grow w-full">
                        <div className="grid grid-cols-2 gap-4">
                            {recommendedStocks.map((stock) => (
                                <div onClick={() => setFocusedStock(stock)} className="cursor-pointer transition-transform hover:scale-[1.02]" key={stock.ticker}>
                                    <StockListItem
                                        stock={stock}
                                        isSelected={selectedStocks.includes(stock.ticker)}
                                        onToggle={() => handleToggle(stock.ticker, stock)}
                                    />
                                </div>
                            ))}
                            {recommendedStocks.length === 0 && (
                                <div className="py-8 text-center text-gray-500 bg-gray-800/30 rounded-xl border border-dashed border-gray-700">
                                    추천 종목이 없습니다.
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* =======================================
                    Area 4 (우측 하단-우 패널): 종목 주석 및 코멘트
                ======================================= */}
                <div className="lg:col-span-4 bg-gray-900/60 border border-gray-700/50 backdrop-blur-md rounded-3xl p-6 shadow-2xl animate-slide-up flex flex-col relative overflow-hidden" style={{ animationDelay: '0.6s' }}>
                    {/* 장식용 배경 요소 */}
                    <div className="absolute top-0 right-0 w-32 h-32 bg-blue-600/10 rounded-full blur-3xl -mr-10 -mt-10 pointer-events-none"></div>
                    <div className="absolute bottom-0 left-0 w-40 h-40 bg-purple-600/10 rounded-full blur-3xl -ml-16 -mb-16 pointer-events-none"></div>

                    <h3 className="text-lg font-bold text-white mb-6 relative z-10 flex items-center">
                        <span className="bg-purple-500 w-2 h-6 rounded-sm mr-2 block"></span>
                        투자 인사이트 주석
                    </h3>

                    {focusedStock ? (
                        <div className="flex flex-col h-full relative z-10">
                            <div className="flex items-baseline mb-4 p-4 bg-gray-800/80 rounded-2xl border border-gray-700">
                                <span className="text-3xl font-black text-white mr-3">{focusedStock.ticker}</span>
                                <span className="text-sm font-semibold text-gray-400">{focusedStock.name}</span>
                            </div>

                            <div className="flex gap-4 mb-6">
                                <div className="bg-gray-800/50 rounded-xl p-3 flex-1 border border-gray-700/50 flex flex-col items-center justify-center">
                                    <span className="text-xs text-gray-400 mb-1">기대 수익률(3M)</span>
                                    <span className="text-xl font-bold text-green-400">+{focusedStock.expectedReturn3M}%</span>
                                </div>
                                <div className="bg-gray-800/50 rounded-xl p-3 flex-1 border border-gray-700/50 flex flex-col items-center justify-center">
                                    <span className="text-xs text-gray-400 mb-1">변동성 등급</span>
                                    <span className={`text-xl font-bold ${focusedStock.volatility === 'High' ? 'text-red-400' : focusedStock.volatility === 'Medium' ? 'text-yellow-400' : 'text-emerald-400'}`}>
                                        {focusedStock.volatility}
                                    </span>
                                </div>
                            </div>

                            <p className="text-gray-300 leading-relaxed text-sm bg-blue-900/10 p-5 rounded-2xl border border-blue-500/20 shadow-inner flex-grow">
                                {generateCommentary(focusedStock)}
                            </p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-full text-center p-6 relative z-10">
                            <span className="text-5xl mb-4 opacity-50">👀</span>
                            <p className="text-gray-400 font-semibold mb-2">종목을 선택해 주세요</p>
                            <p className="text-sm text-gray-500">좌측 범례에서 클릭 시 상세 주석과 분석 코멘터리가 여기에 비동기 렌더링 됩니다.</p>
                        </div>
                    )}
                </div>

            </div>

            {/* 메인으로 돌아가기 버튼 */}
            <div className="w-full max-w-[1920px] mt-8 animate-fade-in text-right">
                <button
                    onClick={onRestart}
                    className="px-8 py-3 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-full text-white font-bold transition-colors shadow-lg"
                >
                    테스트 초기화 및 다시하기 🔄
                </button>
            </div>
        </div>
    );
}

export default DashboardResult;

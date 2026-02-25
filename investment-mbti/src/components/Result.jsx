import React from 'react';
import { TrendingUp } from 'lucide-react';
import './Result.css';

function Result({ personaData, onRestart }) {
    if (!personaData) return null;

    const recommendedStocks = personaData.recommendedStocks || [];

    // MVO 최적 비중(weight) 기반으로 가중평균된 정확한 포트폴리오 3개월 기대수익률 산출
    const mvoExpectedReturn = recommendedStocks.reduce((acc, stock) =>
        acc + (stock.weight / 100) * stock.expectedReturn3M, 0
    ).toFixed(2);

    return (
        <div className="result-container animate-fade-in w-full">
            {/* 상단 헤더 영역 */}
            <div className="result-header animate-slide-up text-center mb-10" style={{ animationDelay: '0.1s' }}>
                <div className="result-badge mx-auto">투자 성향 분석 완료</div>
                <h1 className="persona-title text-4xl mt-4 mb-2 font-bold">{personaData.title}</h1>
                <p className="persona-subtitle text-lg text-gray-300">{personaData.description}</p>
            </div>

            {/* 메인 내용 영역 (데스크톱 2단 분리 Grid) */}
            <div className="lg:grid lg:grid-cols-12 lg:gap-8 items-start">

                {/* 좌측: 페르소나 설명 및 특징 */}
                <div className="lg:col-span-5 bg-[rgba(30,41,59,0.7)] p-6 rounded-2xl border border-[rgba(255,255,255,0.1)] backdrop-blur-md animate-slide-up mb-8 lg:mb-0 lg:sticky lg:top-8" style={{ animationDelay: '0.3s' }}>
                    <h3 className="section-title">💡 핵심 특징</h3>
                    <ul className="feature-list space-y-3">
                        {personaData.features.map((feat, idx) => (
                            <li key={idx} className="flex items-start">
                                <span className="text-blue-400 mr-2 mt-1">•</span>
                                <span>{feat}</span>
                            </li>
                        ))}
                    </ul>
                </div>

                {/* 우측: 포트폴리오 추천 */}
                <div className="lg:col-span-7 bg-[rgba(30,41,59,0.7)] p-6 rounded-2xl border border-[rgba(255,255,255,0.1)] backdrop-blur-md animate-slide-up h-full flex flex-col" style={{ animationDelay: '0.4s' }}>
                    <div className="portfolio-header">
                        <h3 className="section-title">📈 MVO 맞춤형 핵심 포트폴리오</h3>
                    </div>
                    <p className="section-desc mb-6">투자 성향과 손실 한도를 최적화한 인공지능 추천 종목과 비중입니다.</p>

                    <div className="return-badge animate-pulse mb-8 w-full justify-center">
                        <TrendingUp size={20} className="text-blue-400" />
                        3개월 후 포트폴리오 예상 기대수익률: <span className="text-green-400 font-bold text-xl ml-2">+{mvoExpectedReturn}%</span>
                    </div>

                    {/* 바둑판(다단) 그리드 뷰 전환 (모바일: 1열, 데스크톱: 2열) */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-grow">
                        {recommendedStocks.map((stock) => (
                            <div
                                key={stock.id || stock.ticker}
                                className="stock-item animate-fade-in hover:-translate-y-1 hover:shadow-lg hover:shadow-blue-500/10 transition-all duration-300 h-full flex flex-col justify-between"
                            >
                                <div className="flex justify-between items-start mb-4">
                                    <div className="stock-info text-left">
                                        <span className="stock-ticker text-lg font-bold block">{stock.ticker}</span>
                                        <span className="stock-name text-sm text-gray-400">{stock.name}</span>
                                    </div>
                                    <div className="stock-weight-badge text-lg ml-2 shrink-0">
                                        {stock.weight}%
                                    </div>
                                </div>
                                <div className="stock-volatility text-right pt-3 border-t border-gray-700/50 mt-auto">
                                    기대수익 <span className="text-green-400 font-bold ml-1">+{stock.expectedReturn3M}%</span>
                                </div>
                            </div>
                        ))}
                        {recommendedStocks.length === 0 && (
                            <div className="empty-portfolio-state col-span-full py-12 text-center text-gray-400 bg-gray-800/20 rounded-xl border border-dashed border-gray-700">
                                선택된 종목이 없습니다. 종목을 추가해보세요!
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* 하단 액션 버튼 */}
            <div className="action-buttons animate-slide-up mt-12 mb-8 flex flex-col sm:flex-row justify-center gap-4" style={{ animationDelay: '0.5s' }}>
                <button className="btn btn-primary share-btn w-full sm:w-auto hover:-translate-y-1 transition-transform" onClick={() => alert('공유 기능 준비 중입니다!')}>
                    결과 공유하기 🔗
                </button>
                <button className="btn restart-btn w-full sm:w-auto hover:-translate-y-1 transition-transform" onClick={onRestart}>
                    테스트 다시하기 🔄
                </button>
            </div>
        </div>
    );
}

export default Result;

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
        <div className="result-container animate-fade-in">
            <div className="result-header animate-slide-up" style={{ animationDelay: '0.1s' }}>
                <div className="result-badge">투자 성향 분석 완료</div>
                <h1 className="persona-title">{personaData.title}</h1>
                <p className="persona-subtitle">{personaData.description}</p>
            </div>

            <div className="persona-card animate-slide-up" style={{ animationDelay: '0.3s' }}>
                <h3 className="section-title">💡 핵심 특징</h3>
                <ul className="feature-list">
                    {personaData.features.map((feat, idx) => (
                        <li key={idx}>{feat}</li>
                    ))}
                </ul>

                <div className="divider"></div>

                <div className="portfolio-header">
                    <h3 className="section-title">📈 MVO 맞춤형 핵심 포트폴리오</h3>
                </div>
                <p className="section-desc">투자 성향과 손실 한도를 최적화한 인공지능 추천 비중입니다.</p>

                <div className="return-badge animate-pulse">
                    <TrendingUp size={20} className="text-blue-400" />
                    3개월 후 포트폴리오 예상 기대수익률: <span>+{mvoExpectedReturn}%</span>
                </div>

                <div className="stock-list-container wrap-list">
                    {recommendedStocks.map((stock) => (
                        <div
                            key={stock.id || stock.ticker}
                            className="stock-item animate-fade-in"
                        >
                            <div className="stock-weight-badge">
                                {stock.weight}%
                            </div>

                            <div className="stock-info">
                                <span className="stock-ticker">{stock.ticker}</span>
                                <span className="stock-name">{stock.name}</span>
                            </div>
                            <div className="stock-volatility">
                                기대수익 <span className="text-green-400">+{stock.expectedReturn3M}%</span>
                            </div>
                        </div>
                    ))}
                    {recommendedStocks.length === 0 && (
                        <div className="empty-portfolio-state">
                            선택된 종목이 없습니다. 종목을 추가해보세요!
                        </div>
                    )}
                </div>
            </div>

            <div className="action-buttons animate-slide-up" style={{ animationDelay: '0.5s' }}>
                <button className="btn btn-primary share-btn" onClick={() => alert('공유 기능 준비 중입니다!')}>
                    결과 공유하기 🔗
                </button>
                <button className="btn restart-btn" onClick={onRestart}>
                    테스트 다시하기 🔄
                </button>
            </div>
        </div>
    );
}

export default Result;

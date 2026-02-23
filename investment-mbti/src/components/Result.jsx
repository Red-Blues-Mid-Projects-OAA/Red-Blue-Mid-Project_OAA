import React, { useState, useEffect } from 'react';
import { Plus, X, Search, TrendingUp } from 'lucide-react';
import { STOCKS_DATA } from '../constants/stocks';
import './Result.css';

function Result({ personaData, onRestart }) {
    if (!personaData) return null;

    // Initialize selected stocks with the 10 recommended stocks by default
    const [selectedStocks, setSelectedStocks] = useState([]);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        if (personaData && personaData.recommendedStocks) {
            setSelectedStocks(personaData.recommendedStocks.map(s => s.ticker));
        }
    }, [personaData]);

    const toggleStock = (ticker) => {
        setSelectedStocks(prev =>
            prev.includes(ticker)
                ? prev.filter(t => t !== ticker)
                : [...prev, ticker]
        );
    };

    // Calculate average expected return using STOCKS_DATA (which contains expectedReturn3M)
    const currentSelectedStockObjects = STOCKS_DATA.filter(s =>
        selectedStocks.includes(s.ticker)
    );

    const avgExpectedReturn = currentSelectedStockObjects.length > 0
        ? (currentSelectedStockObjects.reduce((acc, stock) => acc + stock.expectedReturn3M, 0) / currentSelectedStockObjects.length).toFixed(2)
        : 0.00;

    // Filter ALL 50 stocks for the modal based on search query
    const filteredStocks = STOCKS_DATA.filter(stock =>
        stock.name.includes(searchQuery) || stock.ticker.toLowerCase().includes(searchQuery.toLowerCase())
    );

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
                    <h3 className="section-title">📈 나만의 맞춤 포트폴리오</h3>
                    <button className="btn-add-stock" onClick={() => setIsModalOpen(true)}>
                        <Plus size={16} /> 종목 추가하기
                    </button>
                </div>
                <p className="section-desc">체크박스를 통해 담고 싶은 종목만 선택해 보세요.</p>

                <div className="return-badge animate-pulse">
                    <TrendingUp size={20} className="text-blue-400" />
                    3개월 후 포트폴리오 기대수익률: <span>+{avgExpectedReturn}%</span>
                </div>

                <div className="stock-list-container wrap-list">
                    {currentSelectedStockObjects.map((stock) => (
                        <label
                            key={stock.id}
                            className="stock-item selectable selected"
                        >
                            <input
                                type="checkbox"
                                checked={true}
                                onChange={() => toggleStock(stock.ticker)}
                                className="hidden-checkbox"
                            />
                            <div className="custom-checkbox"></div>

                            <div className="stock-info">
                                <span className="stock-ticker">{stock.ticker}</span>
                                <span className="stock-name">{stock.name}</span>
                            </div>
                            <div className="stock-volatility">
                                수익률 <span>+{stock.expectedReturn3M}%</span>
                            </div>
                        </label>
                    ))}
                    {currentSelectedStockObjects.length === 0 && (
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

            {/* Portfolio Customization Modal */}
            {isModalOpen && (
                <div className="modal-overlay" onClick={() => setIsModalOpen(false)}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>전체 종목 리스트 (Top 50)</h3>
                            <button className="close-btn" onClick={() => setIsModalOpen(false)}>
                                <X size={24} />
                            </button>
                        </div>

                        <div className="search-bar">
                            <Search size={18} />
                            <input
                                type="text"
                                placeholder="종목명 또는 심볼 검색..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>

                        <div className="modal-stock-list">
                            {filteredStocks.map(stock => (
                                <label key={stock.id} className={`modal-stock-item ${selectedStocks.includes(stock.ticker) ? 'selected' : ''}`}>
                                    <div className="stock-info-left">
                                        <input
                                            type="checkbox"
                                            checked={selectedStocks.includes(stock.ticker)}
                                            onChange={() => toggleStock(stock.ticker)}
                                        />
                                        <div className="modal-stock-details">
                                            <span className="stock-ticker">{stock.ticker}</span>
                                            <span className="stock-name">{stock.name}</span>
                                        </div>
                                    </div>
                                    <div className="stock-info-right">
                                        <span className={`volatility-badge ${stock.volatility.toLowerCase()}`}>{stock.volatility}</span>
                                        <span className="stock-return">+{stock.expectedReturn3M}%</span>
                                    </div>
                                </label>
                            ))}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default Result;

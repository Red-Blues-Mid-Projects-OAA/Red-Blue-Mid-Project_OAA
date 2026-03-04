import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Search, ArrowUpDown, XCircle, Eye, RefreshCw, ArrowLeft, CheckCircle } from 'lucide-react';
import './PortfolioSelection.css';

const API_BASE = 'http://localhost:8000';

function PortfolioSelection({ recommendedStocks = [], finalLevel = 2, lambdaFinal = 7.7, onBack, onRestart, onConfirm }) {
    // ── 상태 관리 ──
    const [allStocks, setAllStocks] = useState([]);
    const [selectedTickers, setSelectedTickers] = useState(new Set());
    const [portfolioScores, setPortfolioScores] = useState({ return_pct: 0, risk_pct: 0, risk_pct_naive: 0, diversification_benefit: 0 });
    const [searchQuery, setSearchQuery] = useState('');
    const [sortKey, setSortKey] = useState('market_cap_asc');
    const [showSortDropdown, setShowSortDropdown] = useState(false);
    const [showSelectedOnly, setShowSelectedOnly] = useState(false);
    const [loading, setLoading] = useState(true);
    const [errorMsg, setErrorMsg] = useState(null);

    // 위험 타입별 최소 종목 수
    const MIN_COUNT_MAP = { 4: 10, 3: 7, 2: 4, 1: 1 };
    const LEVEL_NAME_MAP = { 4: '노예 개미', 3: '월급루팡 개미', 2: '파이어족 개미', 1: 'YOLO 개미' };
    const minCount = MIN_COUNT_MAP[finalLevel] || 4;
    const levelName = LEVEL_NAME_MAP[finalLevel] || '';

    // ── 초기화: 추천 종목을 기본 선택 + 전체 종목 로드 ──
    useEffect(() => {
        const initialTickers = new Set(recommendedStocks.map(s => String(s.ticker || '').toUpperCase()));
        setSelectedTickers(initialTickers);

        fetch(`${API_BASE}/api/all-stocks`)
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    setAllStocks(data.data);
                }
            })
            .catch(err => console.error('all-stocks fetch error:', err))
            .finally(() => setLoading(false));
    }, [recommendedStocks]);

    // ── 선택 변경 시 포트폴리오 스코어 재계산 ──
    useEffect(() => {
        const tickers = Array.from(selectedTickers);
        if (tickers.length === 0) {
            setPortfolioScores({ return_pct: 0, risk_pct: 0, risk_pct_naive: 0, diversification_benefit: 0 });
            return;
        }

        fetch(`${API_BASE}/api/portfolio-scores`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ selected_tickers: tickers }),
        })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    setPortfolioScores(data.data);
                }
            })
            .catch(err => console.error('portfolio-scores fetch error:', err));
    }, [selectedTickers]);

    // ── 토글 로직 ──
    const toggleTicker = useCallback((ticker) => {
        setSelectedTickers(prev => {
            const next = new Set(prev);
            if (next.has(ticker)) {
                next.delete(ticker);
            } else {
                next.add(ticker);
            }
            return next;
        });
    }, []);

    const clearAll = useCallback(() => {
        setSelectedTickers(new Set());
    }, []);

    // ── 추천 종목에 대한 스코어 매핑 ──
    const recStocksWithScores = useMemo(() => {
        const scoreMap = {};
        allStocks.forEach(s => { scoreMap[s.ticker] = s; });
        return recommendedStocks.map((s, i) => {
            const ticker = String(s.ticker || '').toUpperCase();
            const stockData = scoreMap[ticker] || {};
            return {
                ticker,
                name: s.name || stockData.name || ticker,
                return_rank: stockData.return_rank ?? 300,
                risk_rank: stockData.risk_rank ?? 300,
                rank: i + 1,
            };
        });
    }, [recommendedStocks, allStocks]);

    // ── 검색 + 정렬 + 필터 ──
    const filteredStocks = useMemo(() => {
        let list = [...allStocks];

        // 검색
        if (searchQuery.trim()) {
            const q = searchQuery.trim().toLowerCase();
            list = list.filter(s =>
                s.name.toLowerCase().includes(q) ||
                s.ticker.toLowerCase().includes(q)
            );
        }

        // 선택 항목만 보기
        if (showSelectedOnly) {
            list = list.filter(s => selectedTickers.has(s.ticker));
        }

        // 정렬 (rank 기반: 작은 숫자 = 더 좋음)
        switch (sortKey) {
            case 'market_cap_asc':
                list.sort((a, b) => a.market_cap_rank - b.market_cap_rank);
                break;
            case 'market_cap_desc':
                list.sort((a, b) => b.market_cap_rank - a.market_cap_rank);
                break;
            case 'return_desc':
                list.sort((a, b) => a.return_rank - b.return_rank);
                break;
            case 'return_asc':
                list.sort((a, b) => b.return_rank - a.return_rank);
                break;
            case 'risk_desc':
                list.sort((a, b) => b.risk_rank - a.risk_rank);
                break;
            case 'risk_asc':
                list.sort((a, b) => a.risk_rank - b.risk_rank);
                break;
            default:
                break;
        }
        return list;
    }, [allStocks, searchQuery, sortKey, showSelectedOnly, selectedTickers]);

    const SORT_OPTIONS = [
        { key: 'market_cap_asc', label: '시가총액 큰 순' },
        { key: 'market_cap_desc', label: '시가총액 작은 순' },
        { key: 'return_desc', label: '수익률 높은 순' },
        { key: 'return_asc', label: '수익률 낮은 순' },
        { key: 'risk_desc', label: '리스크 높은 순' },
        { key: 'risk_asc', label: '리스크 낮은 순' },
    ];

    if (loading) {
        return (
            <div className="ps-container">
                <div className="ps-loading">종목 데이터를 불러오는 중...</div>
            </div>
        );
    }

    return (
        <div className="ps-container">
            {/* ── 상단: 선호 추천 종목 ── */}
            <section className="ps-section reveal delay-2">
                <h2 className="ps-section-title">선호 추천 종목</h2>
                <div className="ps-rec-cards">
                    {recStocksWithScores.map((stock) => {
                        const isSelected = selectedTickers.has(stock.ticker);
                        return (
                            <button
                                key={stock.ticker}
                                type="button"
                                className={`ps-rec-card ${isSelected ? 'ps-rec-card--selected' : ''}`}
                                onClick={() => toggleTicker(stock.ticker)}
                            >
                                <div className="ps-rec-rank">{stock.rank}.</div>
                                <div className="ps-rec-info">
                                    <span className="ps-rec-name">{stock.name}</span>
                                    <span className="ps-rec-ticker"> / {stock.ticker}</span>
                                </div>
                                <div className="ps-rec-scores">
                                    <span className="ps-score-item">수익률: {stock.return_rank}위</span>
                                    <span className="ps-score-item">리스크: {stock.risk_rank}위</span>
                                </div>
                            </button>
                        );
                    })}
                </div>
            </section>

            {/* ── 중간: 스코어 바 + 컨트롤 ── */}
            <section className="ps-controls-section reveal delay-3">
                <div className="ps-bars-wrap">
                    {/* 수익률 바 */}
                    <div className="ps-bar-row">
                        <span className="ps-bar-label">(예상) 수익률 :</span>
                        <div className="ps-bar-track">
                            <div
                                className="ps-bar-fill ps-bar-fill--return"
                                style={{ width: `${portfolioScores.return_pct}%` }}
                            />
                        </div>
                        <span className="ps-bar-value">{portfolioScores.return_pct} / 100</span>
                    </div>
                    {/* 위험도 바 */}
                    <div className="ps-bar-row">
                        <span className="ps-bar-label">(예상) 위험도 :</span>
                        <div className="ps-bar-track">
                            {/* 실제 리스크 (파란-빨간 그라데이션) */}
                            <div
                                className="ps-bar-fill ps-bar-fill--risk"
                                style={{ width: `${portfolioScores.risk_pct_naive}%` }}
                            />
                            {/* 분산효과 감소분 (노란 줄무늬 오버레이) */}
                            {portfolioScores.diversification_benefit > 0 && (
                                <div
                                    className="ps-bar-diversification"
                                    style={{
                                        left: `${portfolioScores.risk_pct}%`,
                                        width: `${portfolioScores.diversification_benefit}%`,
                                    }}
                                />
                            )}
                        </div>
                        <span className="ps-bar-value">{portfolioScores.risk_pct} / 100</span>
                    </div>
                    {/* 위험도 산출 설명 (바 영역 우측 여백에 절대 위치) */}
                    {portfolioScores.risk_pct > 0 && (
                        <div style={{
                            position: 'absolute',
                            right: '-170px',
                            top: '50%',
                            transform: 'translateY(-50%)',
                            fontSize: '0.75rem',
                            color: '#64748b',
                            lineHeight: 1.45,
                            whiteSpace: 'nowrap',
                        }}>
                            <span style={{ fontWeight: 600, color: '#475569' }}>위험도 산출</span><br />
                            가중합 {portfolioScores.risk_pct_naive}
                            {portfolioScores.diversification_benefit > 0 && (
                                <> − 분산효과 {portfolioScores.diversification_benefit}</>
                            )}
                            {' '}= <strong style={{ color: '#0f172a' }}>{portfolioScores.risk_pct}</strong>
                        </div>
                    )}
                </div>

                <div className="ps-controls-right">
                    {/* 검색 */}
                    <div className="ps-search-wrap">
                        <Search size={15} className="ps-search-icon" />
                        <input
                            type="text"
                            className="ps-search-input"
                            placeholder="검색"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                    </div>

                    {/* 정렬 */}
                    <div className="ps-sort-wrap">
                        <button
                            type="button"
                            className="ps-icon-btn"
                            onClick={() => setShowSortDropdown(!showSortDropdown)}
                            title="정렬"
                        >
                            <ArrowUpDown size={16} />
                        </button>
                        {showSortDropdown && (
                            <div className="ps-sort-dropdown">
                                {SORT_OPTIONS.map(opt => (
                                    <button
                                        key={opt.key}
                                        type="button"
                                        className={`ps-sort-option ${sortKey === opt.key ? 'ps-sort-option--active' : ''}`}
                                        onClick={() => { setSortKey(opt.key); setShowSortDropdown(false); }}
                                    >
                                        {opt.label}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* 전체 해제 */}
                    <button type="button" className="ps-text-btn" onClick={clearAll}>
                        전체 해제
                    </button>

                    {/* 선택 항목 보기 */}
                    <button
                        type="button"
                        className={`ps-text-btn ${showSelectedOnly ? 'ps-text-btn--active' : ''}`}
                        onClick={() => setShowSelectedOnly(!showSelectedOnly)}
                    >
                        선택 항목 보기 ({selectedTickers.size})
                    </button>
                </div>
            </section>

            {/* ── 하단: 전체 300개 종목 ── */}
            <section className="ps-list-section reveal delay-4">
                <h2 className="ps-section-title">전체 종목 ({allStocks.length}개)</h2>

                {/* 테이블 헤더 */}
                <div className="ps-list-header">
                    <span className="ps-col-name">한글 종목 / 티커명</span>
                    <span className="ps-col-ret">1M</span>
                    <span className="ps-col-ret">3M</span>
                    <span className="ps-col-ret">6M</span>
                    <span className="ps-col-ret">12M</span>
                    <span className="ps-col-score">수익률 순위</span>
                    <span className="ps-col-score">리스크 순위</span>
                    <span className="ps-col-cap">시가총액</span>
                </div>

                {/* 종목 리스트 */}
                <div className="ps-list-body">
                    {filteredStocks.map((stock, idx) => {
                        const isSelected = selectedTickers.has(stock.ticker);
                        const r = stock.returns || {};
                        const safeReturn = (v) => { const n = Number(v); return Number.isFinite(n) ? n : 0; };
                        return (
                            <button
                                key={stock.ticker}
                                type="button"
                                className={`ps-list-row ${isSelected ? 'ps-list-row--selected' : ''}`}
                                onClick={() => toggleTicker(stock.ticker)}
                            >
                                <span className="ps-col-rank">{stock.market_cap_rank}.</span>
                                <span className="ps-col-name">
                                    <strong>{stock.name}</strong>
                                    <span className="ps-ticker-label"> / {stock.ticker}</span>
                                </span>
                                <span className={`ps-col-ret ps-return-badge ${safeReturn(r['1M']) >= 0 ? 'pos' : 'neg'}`}>
                                    {safeReturn(r['1M']) >= 0 ? '+' : ''}{safeReturn(r['1M']).toFixed(1)}%
                                </span>
                                <span className={`ps-col-ret ps-return-badge ${safeReturn(r['3M']) >= 0 ? 'pos' : 'neg'}`}>
                                    {safeReturn(r['3M']) >= 0 ? '+' : ''}{safeReturn(r['3M']).toFixed(1)}%
                                </span>
                                <span className={`ps-col-ret ps-return-badge ${safeReturn(r['6M']) >= 0 ? 'pos' : 'neg'}`}>
                                    {safeReturn(r['6M']) >= 0 ? '+' : ''}{safeReturn(r['6M']).toFixed(1)}%
                                </span>
                                <span className={`ps-col-ret ps-return-badge ${safeReturn(r['12M']) >= 0 ? 'pos' : 'neg'}`}>
                                    {safeReturn(r['12M']) >= 0 ? '+' : ''}{safeReturn(r['12M']).toFixed(1)}%
                                </span>
                                <span className="ps-col-score">{stock.return_rank}위</span>
                                <span className="ps-col-score">{stock.risk_rank}위</span>
                                <span className="ps-col-cap">{stock.market_cap_rank}위</span>
                            </button>
                        );
                    })}
                    {filteredStocks.length === 0 && (
                        <div className="ps-empty">검색 결과가 없습니다.</div>
                    )}
                </div>
            </section>

            {/* ── 하단 액션 버튼 ── */}
            <div className="ps-actions reveal delay-5">
                <button type="button" className="ps-action-btn ps-action-btn--restart" onClick={onRestart}>
                    <RefreshCw size={16} />
                    처음으로
                </button>
                <button
                    type="button"
                    className="ps-action-btn ps-action-btn--confirm"
                    onClick={() => {
                        if (selectedTickers.size < minCount) {
                            setErrorMsg(`${levelName} 타입은 최소 ${minCount}개 이상의 종목을 선택해야 합니다. (현재 ${selectedTickers.size}개 선택)`);
                            return;
                        }
                        onConfirm(Array.from(selectedTickers));
                    }}
                >
                    <CheckCircle size={16} />
                    포트폴리오 구성하기 ({selectedTickers.size}개)
                </button>
            </div>

            {/* ── 오류 모달 ── */}
            {errorMsg && (
                <div className="ps-error-overlay" onClick={() => setErrorMsg(null)}>
                    <div className="ps-error-modal" onClick={e => e.stopPropagation()}>
                        <p>{errorMsg}</p>
                        <button type="button" onClick={() => setErrorMsg(null)}>확인</button>
                    </div>
                </div>
            )}
        </div>
    );
}

export default PortfolioSelection;

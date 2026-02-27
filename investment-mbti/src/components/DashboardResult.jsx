import React from 'react';
import MbtiBarChart from './MbtiBarChart';
import PortfolioPieChart from './PortfolioPieChart';
import CumulativeReturnChart from './CumulativeReturnChart';
import StockCard from './StockCard';
import './Result.css';

/**
 * 통합 대시보드 결과 컴포넌트.
 *
 * 레이아웃: 좌측 사이드바(MBTI 프로필) + 우측 메인 패널(3단 구조)
 * - 우측 상단: 파이차트 + 포트폴리오 메트릭
 * - 우측 중앙: 누적수익률 차트 (과거 1년 + 예측 3개월)
 * - 우측 하단: 추천 종목 그리드 (읽기 전용)
 */
function DashboardResult({ personaData, onRestart }) {
    if (!personaData) return null;

    // ─── 데이터 추출 ───
    const recommendedStocks = personaData.recommendedStocks || [];
    const rawAnswers = personaData.rawAnswers || [];
    const portfolioAnalysis = personaData.portfolioAnalysis || {};
    const chartData = personaData.chartData || null;
    const forecastData = personaData.forecastData || null;
    const mbti = personaData.mbti || 'ENTJ';
    const mbtiNickname = personaData.mbtiNickname || '';

    // ─── MBTI 성향 바 점수 계산 ───
    const scoreI = rawAnswers.length > 0 ? (rawAnswers.slice(0, 3).filter(a => a === 'A').length / 3) * 100 : 50;
    const scoreN = rawAnswers.length > 0 ? (rawAnswers.slice(3, 6).filter(a => a === 'B').length / 3) * 100 : 50;
    const scoreF = rawAnswers.length > 0 ? (rawAnswers.slice(6, 9).filter(a => a === 'B').length / 3) * 100 : 50;
    const scoreP = rawAnswers.length > 0 ? (rawAnswers.slice(9, 12).filter(a => a === 'B').length / 3) * 100 : 50;

    // 페르소나 이미지 매핑
    const getPersonaImage = (title) => {
        if (title.includes('거북이')) return 'turtle';
        if (title.includes('강아지')) return 'dog';
        if (title.includes('사자')) return 'lion';
        return 'eagle';
    };

    // 수익률 포매팅 (양수 녹색/음수 적색 + 부호)
    const formatReturn = (val) => {
        if (val == null) return '—';
        const sign = val >= 0 ? '+' : '';
        return `${sign}${val.toFixed(2)}%`;
    };

    const returnColor = (val) => val >= 0 ? 'text-green-400' : 'text-red-400';

    return (
        <div className="w-full min-h-screen bg-gray-950 px-4 py-8 flex flex-col items-center">
            {/* 상단 타이틀 */}
            <div className="text-center mb-8 animate-fade-in">
                <h1 className="text-3xl lg:text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400 mb-2">
                    투자 MBTI 통합 분석 리포트
                </h1>
            </div>

            {/* ═══════════════════════════════════════════
                메인 대시보드: 좌측 사이드바 + 우측 3단 패널
            ═══════════════════════════════════════════ */}
            <div className="w-full max-w-[1600px] grid grid-cols-1 lg:grid-cols-12 gap-6">

                {/* ─── 좌측 사이드바: 페르소나 + MBTI 프로필 ─── */}
                <div className="lg:col-span-3 bg-gray-900/60 border border-gray-700/50 backdrop-blur-md rounded-3xl p-6 flex flex-col items-center shadow-2xl animate-slide-up">
                    {/* 페르소나 이미지 */}
                    <div
                        className="w-28 h-28 mb-5 rounded-full overflow-hidden shadow-[0_0_30px_rgba(139,92,246,0.15)] bg-gray-800 flex-shrink-0"
                        style={{ animation: 'float 6s ease-in-out infinite' }}
                    >
                        <img
                            src={`/images/${getPersonaImage(personaData.title)}.png`}
                            alt="Persona"
                            className="w-full h-full object-cover"
                            onError={(e) => { e.target.style.display = 'none'; }}
                        />
                    </div>

                    {/* MBTI 4글자 */}
                    <div className="bg-gray-800/80 border border-gray-600/50 rounded-2xl p-4 w-full text-center mb-4 shadow-inner">
                        <h3 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-indigo-400 tracking-wider font-mono mb-1">
                            {mbti}
                        </h3>
                        {mbtiNickname && (
                            <span className="text-xs font-bold text-gray-300 bg-black/40 px-3 py-1 rounded-full border border-gray-600/50">
                                {mbtiNickname}
                            </span>
                        )}
                    </div>

                    {/* 페르소나명 + 한줄 설명 */}
                    <div className="text-center mb-6">
                        <h2 className="text-xl font-bold text-white mb-1">{personaData.title}</h2>
                        <p className="text-sm text-yellow-500 font-semibold">{personaData.description}</p>
                    </div>

                    {/* MBTI 4차원 바 */}
                    <div className="w-full flex flex-col gap-2">
                        <h4 className="text-xs font-bold text-gray-500 mb-1 text-center">성향 세부 분포</h4>
                        <MbtiBarChart dimension="에너지-집중" score={scoreI} leftLabel="외향" rightLabel="내향" colorClass="bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500" />
                        <MbtiBarChart dimension="직관-현실" score={scoreN} leftLabel="감각" rightLabel="직관" colorClass="bg-gradient-to-r from-emerald-400 via-green-500 to-teal-500" />
                        <MbtiBarChart dimension="논리-가치" score={scoreF} leftLabel="사고" rightLabel="감정" colorClass="bg-gradient-to-r from-amber-400 via-orange-500 to-red-500" />
                        <MbtiBarChart dimension="계획-탐색" score={scoreP} leftLabel="계획형" rightLabel="유연형" colorClass="bg-gradient-to-r from-purple-400 via-fuchsia-500 to-pink-500" />
                    </div>
                </div>

                {/* ─── 우측 메인 패널 (3단 구조) ─── */}
                <div className="lg:col-span-9 flex flex-col gap-6">

                    {/* ═══ 우측 상단: 포트폴리오 구성 (파이차트 + 메트릭) ═══ */}
                    <div className="bg-gray-900/60 border border-gray-700/50 backdrop-blur-md rounded-3xl p-6 shadow-2xl animate-slide-up" style={{ animationDelay: '0.1s' }}>
                        <h3 className="text-lg font-bold text-white mb-5 flex items-center">
                            <span className="bg-blue-500 w-1.5 h-5 rounded-sm mr-2 block" />
                            추천 포트폴리오 구성
                        </h3>

                        <div className="flex flex-col lg:flex-row items-center lg:items-start gap-8">
                            {/* 파이차트 */}
                            <div className="shrink-0">
                                <PortfolioPieChart stocks={recommendedStocks} size={200} />
                            </div>

                            {/* 포트폴리오 메트릭 4칸 */}
                            <div className="grid grid-cols-2 gap-4 flex-1">
                                {/* 과거 3개월 수익률 */}
                                <div className="bg-gray-800/60 rounded-2xl p-4 border border-gray-700/30">
                                    <span className="text-xs text-gray-500 block mb-1">과거 3개월 수익률</span>
                                    <span className={`text-2xl font-black ${returnColor(portfolioAnalysis.expected_return_simple)}`}>
                                        {formatReturn(portfolioAnalysis.expected_return_simple)}
                                    </span>
                                </div>

                                {/* 예측 3개월 수익률 */}
                                <div className="bg-gray-800/60 rounded-2xl p-4 border border-gray-700/30">
                                    <span className="text-xs text-gray-500 block mb-1">예측 3개월 수익률</span>
                                    <span className={`text-2xl font-black ${returnColor(portfolioAnalysis.expected_return_simple)}`}>
                                        {formatReturn(portfolioAnalysis.expected_return_simple)}
                                    </span>
                                </div>

                                {/* 변동성 */}
                                <div className="bg-gray-800/60 rounded-2xl p-4 border border-gray-700/30">
                                    <span className="text-xs text-gray-500 block mb-1">변동성 (60일 σ)</span>
                                    <span className="text-2xl font-black text-blue-400">
                                        {portfolioAnalysis.volatility_60d != null
                                            ? `${(portfolioAnalysis.volatility_60d * 100).toFixed(2)}%`
                                            : '—'}
                                    </span>
                                </div>

                                {/* VaR 5% */}
                                <div className="bg-gray-800/60 rounded-2xl p-4 border border-gray-700/30">
                                    <span className="text-xs text-gray-500 block mb-1">VaR 5%</span>
                                    <span className="text-2xl font-black text-red-400">
                                        {portfolioAnalysis.var_5 != null
                                            ? `${portfolioAnalysis.var_5.toFixed(2)}%`
                                            : '준비중'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ═══ 우측 중앙: 누적수익률 차트 ═══ */}
                    <div className="bg-gray-900/60 border border-gray-700/50 backdrop-blur-md rounded-3xl p-6 shadow-2xl animate-slide-up" style={{ animationDelay: '0.2s' }}>
                        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                            <span className="bg-violet-500 w-1.5 h-5 rounded-sm mr-2 block" />
                            포트폴리오 성과 분석
                            <span className="ml-auto text-xs text-gray-500 font-normal">포트폴리오(파랑) vs S&P 500(회색) | 예측(보라 점선)</span>
                        </h3>
                        <CumulativeReturnChart
                            chartData={chartData}
                            forecastData={forecastData}
                            warnings={chartData?.warnings || []}
                        />
                    </div>

                    {/* ═══ 우측 하단: 추천 종목 그리드 (읽기 전용) ═══ */}
                    <div className="bg-gray-900/60 border border-gray-700/50 backdrop-blur-md rounded-3xl p-6 shadow-2xl animate-slide-up" style={{ animationDelay: '0.3s' }}>
                        <h3 className="text-lg font-bold text-white mb-4 flex items-center">
                            <span className="bg-emerald-500 w-1.5 h-5 rounded-sm mr-2 block" />
                            추천 종목
                            <span className="ml-2 text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded-md">
                                {recommendedStocks.length}개
                            </span>
                        </h3>

                        {/* 종목 수에 따라 유동적 그리드 */}
                        <div className={`grid gap-3 ${recommendedStocks.length <= 2 ? 'grid-cols-1 sm:grid-cols-2' :
                            recommendedStocks.length <= 4 ? 'grid-cols-2' :
                                recommendedStocks.length <= 7 ? 'grid-cols-2 lg:grid-cols-3 xl:grid-cols-4' :
                                    'grid-cols-2 lg:grid-cols-3 xl:grid-cols-5'
                            }`}>
                            {recommendedStocks.map((stock) => (
                                <StockCard key={stock.ticker} stock={stock} />
                            ))}
                        </div>

                        {recommendedStocks.length === 0 && (
                            <div className="py-8 text-center text-gray-500 bg-gray-800/30 rounded-xl border border-dashed border-gray-700">
                                추천 종목이 없습니다.
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* 메인으로 돌아가기 버튼 */}
            <div className="w-full max-w-[1600px] mt-8 animate-fade-in text-right">
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

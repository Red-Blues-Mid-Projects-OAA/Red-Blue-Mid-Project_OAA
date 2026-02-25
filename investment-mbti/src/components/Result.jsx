import React from 'react';
import MbtiBarChart from './MbtiBarChart';
import './Result.css';

function Result({ personaData, onRestart, onShowRecommendation }) {
    if (!personaData) return null;

    const recommendedStocks = personaData.recommendedStocks || [];
    const mbtiType = personaData.features?.[0]?.split(': ')[1] || 'ENTJ';
    const rawAnswers = personaData.rawAnswers || [];

    // Score calculation logic based on 12 questions (3 per dimension)
    // Left label is 0%, Right label is 100% capacity
    // Q1~Q3: I(A) vs E(B) -> Left=E, Right=I -> RightScore = A count
    const scoreI = rawAnswers.length > 0 ? (rawAnswers.slice(0, 3).filter(a => a === 'A').length / 3) * 100 : 50;
    // Q4~Q6: S(A) vs N(B) -> Left=S, Right=N -> RightScore = B count
    const scoreN = rawAnswers.length > 0 ? (rawAnswers.slice(3, 6).filter(a => a === 'B').length / 3) * 100 : 50;
    // Q7~Q9: T(A) vs F(B) -> Left=T, Right=F -> RightScore = B count
    const scoreF = rawAnswers.length > 0 ? (rawAnswers.slice(6, 9).filter(a => a === 'B').length / 3) * 100 : 50;
    // Q10~Q12: J(A) vs P(B) -> Left=J, Right=P -> RightScore = B count
    const scoreP = rawAnswers.length > 0 ? (rawAnswers.slice(9, 12).filter(a => a === 'B').length / 3) * 100 : 50;

    return (
        <div className="result-container animate-fade-in w-full max-w-2xl mx-auto py-12 px-4">
            {/* 상단 헤더 영역 */}
            <div className="result-header animate-slide-up text-center mb-8 flex flex-col items-center" style={{ animationDelay: '0.1s' }}>
                <div className="w-40 h-40 mb-6 rounded-full overflow-hidden shadow-[0_0_40px_rgba(139,92,246,0.3)] bg-gray-800" style={{ animation: 'float 6s ease-in-out infinite' }}>
                    <img
                        src={`/images/${personaData.title.includes('거북이') ? 'turtle' : personaData.title.includes('강아지') ? 'dog' : personaData.title.includes('사자') ? 'lion' : 'eagle'}.png`}
                        alt="Persona Avatar"
                        className="w-full h-full object-cover"
                        onError={(e) => { e.target.style.display = 'none'; }}
                    />
                </div>

                <p className="text-[#a39dd1] font-semibold text-sm mb-2">팩트로 보는 나는...</p>
                <h1 className="text-3xl font-bold text-white mb-4 whitespace-pre-line">{personaData.title}</h1>

                {/* 캐치프레이즈 (예: 무조건 인생은 짧고 굵게 레쭈고) */}
                <p className="text-xl text-yellow-400 font-extrabold mb-8 decoration-wavy underline-offset-8 underline decoration-yellow-400/50">
                    {personaData.description}
                </p>

                {/* MBTI 박스 */}
                <div className="bg-gray-900/40 border border-gray-700/50 backdrop-blur-md rounded-2xl px-8 py-6 shadow-2xl w-full max-w-sm mb-12 transform hover:scale-105 transition-transform flex flex-col items-center justify-center">
                    <p className="text-gray-300 font-bold mb-2">나의 투자 MBTI는?</p>
                    <h2 className="text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-indigo-400 tracking-wider font-mono mb-3 drop-shadow-lg">
                        {mbtiType.split(' ')[0]}
                    </h2>
                    {mbtiType.includes(' ') && (
                        <p className="text-base font-bold text-gray-400 bg-black/20 px-4 py-1.5 rounded-full border border-gray-700/50">
                            {mbtiType.substring(mbtiType.indexOf(' ')).trim()}
                        </p>
                    )}
                </div>
            </div>

            {/* MBTI 바 차트 영역 */}
            <div className="animate-slide-up bg-gray-900/50 p-6 md:p-8 rounded-3xl border border-gray-700/50 backdrop-blur-sm mb-12 shadow-2xl" style={{ animationDelay: '0.3s' }}>
                <h3 className="text-center text-xl font-bold text-white mb-8">💡 나의 성향 세부 분포</h3>
                <div className="flex flex-col w-full max-w-md mx-auto gap-2">
                    <MbtiBarChart dimension="시장 반응" score={scoreI} leftLabel="외향" rightLabel="내향" colorClass="bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500" />
                    <MbtiBarChart dimension="가치 판단" score={scoreN} leftLabel="감각" rightLabel="직관" colorClass="bg-gradient-to-r from-emerald-400 via-green-500 to-teal-500" />
                    <MbtiBarChart dimension="의사 결정" score={scoreF} leftLabel="사고" rightLabel="감정" colorClass="bg-gradient-to-r from-amber-400 via-orange-500 to-red-500" />
                    <MbtiBarChart dimension="대응 방식" score={scoreP} leftLabel="계획형" rightLabel="유연형" colorClass="bg-gradient-to-r from-purple-400 via-fuchsia-500 to-pink-500" />
                </div>
            </div>

            {/* 하단 액션 버튼 (포트폴리오 페이지로 이동) */}
            <div className="action-buttons animate-slide-up flex flex-col items-center gap-4" style={{ animationDelay: '0.5s' }}>
                <button
                    className="w-full max-w-sm py-4 rounded-full bg-gradient-to-r from-violet-600 to-indigo-600 text-white font-bold text-lg shadow-[0_0_20px_rgba(139,92,246,0.5)] hover:-translate-y-1 hover:shadow-[0_0_30px_rgba(139,92,246,0.7)] transition-all"
                    onClick={onShowRecommendation}
                >
                    내 추천 종목 확인하기 👉
                </button>

                <button
                    className="w-full max-w-sm py-3 rounded-full bg-gray-800 text-gray-300 font-semibold hover:bg-gray-700 transition-colors"
                    onClick={onRestart}
                >
                    테스트 다시하기 🔄
                </button>
            </div>
        </div>
    );
}

export default Result;

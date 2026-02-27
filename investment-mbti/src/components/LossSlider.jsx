import React, { useState } from 'react';
import './LossSlider.css';

// 손실 한도 슬라이더 컴포넌트 (범위: 1% ~ 40%, step: 1%)
function LossSlider({ onComplete }) {
    const [lossValue, setLossValue] = useState(1);
    // 슬라이더 건드렸는지 여부 추적 (Bonus UX requirement)
    const [isSliderTouched, setIsSliderTouched] = useState(false);
    const [showWarning, setShowWarning] = useState(false);

    // 슬라이더 구간별 피드백 (10% 단위로 4구간)
    const getFeedback = (value) => {
        if (value <= 10) return { label: '초보수적', color: '#10b981', desc: '은행 예적금과 원금 보장을 사랑하는 당신' };
        if (value <= 20) return { label: '안전제일', color: '#3b82f6', desc: '약간의 수익을 위해 작은 변동성은 참는 타입' };
        if (value <= 30) return { label: '중도성향', color: '#f59e0b', desc: '시장 수익률만큼은 먹어야 직성이 풀리는 밸런스형' };
        return { label: '공격적', color: '#ef4444', desc: '야수의 심장. 하락장은 바겐세일일 뿐!' };
    };

    // 1000만원 기준 손실 후 남은 금액 계산
    const calculateMoney = (value) => {
        const currentMoney = 10000000;
        const lossPercentage = value / 100;
        const remainingMoney = currentMoney * (1 - lossPercentage);
        return Math.floor(remainingMoney / 10000).toLocaleString() + '만 원';
    };

    const handleSliderChange = (e) => {
        setIsSliderTouched(true);
        if (showWarning) setShowWarning(false);
        setLossValue(parseInt(e.target.value, 10));
    };

    const submitLoss = () => {
        if (!isSliderTouched) {
            setShowWarning(true);
            setTimeout(() => setShowWarning(false), 3000);
            return;
        }
        // 백엔드로 전송할 때는 음수 변환
        onComplete(-lossValue);
    };

    const feedback = getFeedback(lossValue);
    // 그래디언트 비율 계산 (1~40 범위: (value - 1) / 39 * 100)
    const gradientPercent = ((lossValue - 1) / 39) * 100;

    return (
        <div className="slider-container animate-fade-in max-w-xl mx-auto">
            <div className="slider-header animate-slide-up" style={{ animationDelay: '0.1s' }}>
                <h2>마지막으로 묻습니다.</h2>
                <p>당신의 소중한 투자금 <strong>1,000만 원</strong>.</p>
                <p>어디까지 잃어도 밤에 푹 주무실 수 있나요?</p>
            </div>

            <div className="slider-card animate-slide-up" style={{ animationDelay: '0.2s' }}>
                <div className="loss-value">
                    <span className="percent">-{lossValue}%</span>
                </div>

                <div className="money-indicator">
                    <svg className="wallet-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" /></svg>
                    <span className="remains">내 수중에 남은 돈: {calculateMoney(lossValue)}</span>
                </div>

                <input
                    type="range"
                    min="1"
                    max="40"
                    step="1"
                    value={lossValue}
                    onChange={handleSliderChange}
                    className="range-slider cursor-pointer"
                    style={{
                        background: `linear-gradient(to right, ${feedback.color} ${gradientPercent}%, rgba(255,255,255,0.1) ${gradientPercent}%)`
                    }}
                />

                <div className="slider-labels text-sm mt-2 text-gray-400">
                    <span>안전 (-1%)</span>
                    <span>위험 (-40%)</span>
                </div>

            </div>

            {showWarning && (
                <div className="text-red-400 text-sm font-bold mt-4 animate-bounce bg-red-500/10 border border-red-500/20 py-2 px-4 rounded-xl">
                    ⚠️ 위험한도를 슬라이더로 직접 조절해주세요!
                </div>
            )}

            <button
                className="btn btn-primary submit-btn animate-slide-up mt-6 hover:-translate-y-1 hover:shadow-lg transition-all"
                style={{ animationDelay: '0.3s' }}
                onClick={submitLoss}
            >
                마지막 답변 제출 🚀
            </button>
        </div>
    );
}

export default LossSlider;

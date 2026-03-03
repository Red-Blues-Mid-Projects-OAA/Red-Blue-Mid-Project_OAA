import { useState } from 'react';

// 투자 예정 금액 입력 페이지 컴포넌트
function InvestmentAmount({ onComplete, onBack }) {
    const [value, setValue] = useState('');
    const [showAlert, setShowAlert] = useState(false);

    // 숫자만 허용, 앞자리 0 제거
    const handleChange = (e) => {
        const numericOnly = e.target.value.replace(/[^0-9]/g, '');
        setValue(numericOnly === '0' ? '' : numericOnly);
        if (showAlert) setShowAlert(false);
    };

    const handleNext = () => {
        const amount = parseInt(value, 10);
        // 미입력 또는 10만원 미만이면 알림 표시
        if (!value || isNaN(amount) || amount < 10) {
            setShowAlert(true);
            return;
        }
        onComplete(amount);
    };

    // 엔터키로 다음 페이지 이동
    const handleKeyDown = (e) => {
        if (e.key === 'Enter') handleNext();
    };

    return (
        <div className="animate-fade-in flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-6 py-10">
            <div className="w-full max-w-xl flex flex-col gap-10 animate-slide-up" style={{ animationDelay: '0.1s' }}>

                {/* 상단: 제목 + 설명 */}
                <div className="text-center">
                    <h2 className="text-3xl font-bold" style={{ color: 'var(--text-primary)' }}>
                        얼마까지 투자할 계획이신가요?
                    </h2>
                </div>

                {/* 중단: 입력 카드 — min-h 고정으로 입력 시 높이 변동 방지 */}
                <div className="bg-white border rounded-2xl py-16 px-10 shadow-sm flex flex-col items-center justify-center gap-4"
                    style={{ borderColor: 'var(--card-border)', minHeight: '220px' }}>

                    <div className="flex items-center justify-center gap-3">
                        <input
                            type="text"
                            inputMode="numeric"
                            pattern="[0-9]*"
                            value={value}
                            onChange={handleChange}
                            onKeyDown={handleKeyDown}
                            placeholder="0"
                            className="text-center text-4xl font-bold outline-none border-b-2 pb-2 transition-colors duration-200"
                            style={{
                                width: '180px',
                                borderColor: showAlert ? '#ef4444' : 'var(--accent-primary)',
                                color: 'var(--text-primary)',
                                background: 'transparent',
                            }}
                            autoFocus
                        />
                        <span className="text-xl font-semibold whitespace-nowrap" style={{ color: 'var(--text-secondary)' }}>
                            만 원
                        </span>
                    </div>

                    {/* 변환 금액 — 자리 항상 차지해 카드 높이 고정 */}
                    <p className="text-xl font-bold" style={{
                        color: 'var(--accent-primary)',
                        visibility: (value && parseInt(value, 10) >= 10) ? 'visible' : 'hidden'
                    }}>
                        {value && parseInt(value, 10) >= 10
                            ? (parseInt(value, 10) * 10000).toLocaleString()
                            : '0'}원
                    </p>
                </div>

                {/* 하단: 버튼 영역 */}
                <div>
                    {showAlert && (
                        <div className="mb-4 text-red-500 text-sm font-bold text-center animate-bounce
                                        bg-red-50 border border-red-200 py-2 px-4 rounded-xl">
                            ⚠️ 최소 입력 가능 금액은 10만원입니다
                        </div>
                    )}

                    <button
                        onClick={handleNext}
                        className="btn btn-primary btn-3d w-4/5 mx-auto block text-lg font-bold py-4 rounded-2xl"
                    >
                        다음 →
                    </button>

                    <div className="mt-10 flex justify-center">
                        <button
                            onClick={onBack}
                            className="flex items-center gap-1.5 text-gray-400 hover:text-gray-700 transition-colors duration-200 group"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="16" height="16"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="group-hover:-translate-x-0.5 transition-transform duration-200"
                            >
                                <path d="m15 18-6-6 6-6" />
                            </svg>
                            <span className="text-sm font-semibold">이전으로</span>
                        </button>
                    </div>
                </div>

            </div>
        </div>
    );
}

export default InvestmentAmount;

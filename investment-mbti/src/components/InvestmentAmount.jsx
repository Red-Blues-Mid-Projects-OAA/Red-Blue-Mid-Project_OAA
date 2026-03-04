import { useEffect, useState } from 'react';

// 투자 예정 금액 입력 페이지 컴포넌트
function InvestmentAmount({ onComplete, onBack }) {
    const [value, setValue] = useState('');
    const [showAlert, setShowAlert] = useState(false);
    const [alertNonce, setAlertNonce] = useState(0);

    useEffect(() => {
        if (!showAlert) return undefined;
        const timer = setTimeout(() => setShowAlert(false), 3000);
        return () => clearTimeout(timer);
    }, [showAlert, alertNonce]);

    // 숫자만 허용, 앞자리 0 제거
    const handleChange = (e) => {
        const numericOnly = e.target.value.replace(/[^0-9]/g, '');
        setValue(numericOnly === '0' ? '' : numericOnly);
        if (showAlert) setShowAlert(false);
    };

    const handleNext = () => {
        const amount = parseInt(value, 10);
        // 미입력 또는 10만원 미만이면 경고 표시
        if (!value || isNaN(amount) || amount < 10) {
            setAlertNonce((n) => n + 1);
            setShowAlert(true);
            return;
        }
        onComplete(amount);
    };

    // 엔터로 다음 페이지 이동
    const handleKeyDown = (e) => {
        if (e.key === 'Enter') handleNext();
    };

    return (
        <div className="animate-fade-in flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-6 py-10">
            <div className="w-full max-w-xl flex flex-col gap-10 animate-slide-up" style={{ animationDelay: '0.1s' }}>

                {/* 상단: 제목 */}
                <div className="text-center">
                    <h2 className="text-3xl font-bold" style={{ color: 'var(--text-primary)' }}>
                        얼마까지 투자할 계획이신가요?
                    </h2>
                </div>

                {/* 중단: 입력 카드 */}
                <div
                    className="bg-white border rounded-2xl py-16 px-10 shadow-sm flex flex-col items-center justify-center gap-4"
                    style={{ borderColor: 'var(--card-border)', minHeight: '220px' }}
                >
                    <div className="flex items-center justify-center gap-3">
                        <input
                            type="text"
                            inputMode="numeric"
                            pattern="[0-9]*"
                            value={value}
                            onChange={handleChange}
                            onKeyDown={handleKeyDown}
                            placeholder=""
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

                    <p
                        className="text-xl font-bold"
                        style={{
                            color: 'var(--accent-primary)',
                            visibility: value && parseInt(value, 10) >= 10 ? 'visible' : 'hidden',
                        }}
                    >
                        {value && parseInt(value, 10) >= 10 ? (parseInt(value, 10) * 10000).toLocaleString() : '0'}원
                    </p>
                </div>

                {/* 하단: 버튼 영역 */}
                <div>
                    {/* 경고 슬롯 고정: 표시/숨김 시 다른 요소가 움직이지 않음 */}
                    <div className="-mt-1 h-[40px] mb-10 relative">
                        <div
                            key={alertNonce}
                            className={`absolute inset-x-0 top-0 w-full mx-auto text-red-500 text-sm font-bold text-center
                                        bg-red-50 border border-red-200 rounded-xl px-3 py-1
                                        flex items-center justify-center leading-tight transition-all duration-200
                                        ${showAlert
                                            ? 'opacity-100 translate-y-0 animate-[bounce_0.45s_ease-in-out_3]'
                                            : 'opacity-0 -translate-y-1 pointer-events-none'
                                        }`}
                        >
                            ⚠️ 최소 입력 가능 금액은 10만원입니다
                        </div>
                    </div>

                    <button
                        onClick={handleNext}
                        className="btn btn-primary btn-3d w-4/5 mx-auto block text-lg font-bold py-4 rounded-2xl"
                    >
                        다음 →
                    </button>

                    <div className="flow-back-action flex justify-center">
                        <button
                            onClick={onBack}
                            className="flex items-center gap-1.5 text-gray-400 hover:text-gray-700 transition-colors duration-200 group"
                        >
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="16"
                                height="16"
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

/*
 * 이 파일은 질문 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import { useEffect, useState } from 'react';
import './Question.css';

/**
 * 질문 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function Question({ data, step, totalSteps, onAnswer, onBack }) {
    // 화면에서 계속 바뀌는 값을 state로 보관합니다.
    const [animationClass, setAnimationClass] = useState('animate-fade-in');

    // 화면이 열리거나 특정 값이 바뀔 때 필요한 부수 작업을 실행합니다.
    useEffect(() => {
        setAnimationClass('');
        const timer = setTimeout(() => {
            setAnimationClass('animate-slide-up');
        }, 10);
        return () => clearTimeout(timer);
    }, [data.id]);

    // 사용자 입력, 버튼 클릭, API 요청을 처리하는 함수들입니다.
    const handleSelect = (value, optionText, optionKey) => {
        onAnswer({
            questionId: data.id,
            type: data.type,
            value,
            selectedText: optionText,
            optionKey,
        });
    };

    const progress = (step / totalSteps) * 100;

    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
    return (
        <div className={`question-container ${animationClass}`}>
            <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
            </div>

            <div className="question-meta">
                <div className="step-indicator">
                    {step} <span className="step-total">/ {totalSteps}</span>
                </div>
            </div>

            <section className={`question-main-layout ${!data.image ? 'no-image' : ''}`}>
                {data.image && (
                    <div className="question-image-panel">
                        <div className="question-image-container">
                            <img
                                src={data.image}
                                alt="question illustration"
                                className="question-image"
                            />
                        </div>
                    </div>
                )}

                <div className={`question-content-panel ${!data.image ? 'full-width' : ''}`}>
                    <header className="question-header">
                        <h2 className="question-text">{data.question}</h2>
                    </header>

                    <div className="options-container">
                        <button
                            className="btn option-btn hover:-translate-y-1 hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-200"
                            onClick={() => handleSelect(data.optionA.value, data.optionA.text, 'A')}
                        >
                            {data.optionA.text}
                        </button>
                        <button
                            className="btn option-btn hover:-translate-y-1 hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-200"
                            onClick={() => handleSelect(data.optionB.value, data.optionB.text, 'B')}
                        >
                            {data.optionB.text}
                        </button>
                    </div>

                    <div className="question-back-action">
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
            </section>
        </div>
    );
}

export default Question;

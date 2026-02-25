import React, { useState, useEffect } from 'react';
import './Question.css';

function Question({ data, step, totalSteps, onAnswer, onBack }) {
    const [animationClass, setAnimationClass] = useState('animate-fade-in');

    // 문항이 바뀔 때마다 애니메이션 재등록
    useEffect(() => {
        setAnimationClass('');
        const timer = setTimeout(() => {
            setAnimationClass('animate-slide-up');
        }, 10);
        return () => clearTimeout(timer);
    }, [data.id]);

    const handleSelect = (value, optionText, optionKey) => {
        onAnswer({
            questionId: data.id,
            type: data.type,
            value: value,
            selectedText: optionText,
            optionKey: optionKey
        });
    };

    const progress = (step / totalSteps) * 100;

    return (
        <div className={`question-container ${animationClass}`}>
            {/* Progress Bar */}
            <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            </div>

            <div className="flex justify-between items-center mb-4">
                {/* 1단계에서는 뒤로 가기 버튼 숨김 또는 비활성화 처리 */}
                <button
                    onClick={onBack}
                    className={`flex items-center gap-1 px-4 py-2 rounded-full text-sm font-semibold transition-all duration-200 border ${step === 1
                            ? 'opacity-0 cursor-default pointer-events-none'
                            : 'opacity-100 text-violet-300 border-violet-500/30 bg-violet-500/5 hover:bg-violet-500/20 hover:border-violet-400 hover:text-white hover:-translate-y-1 hover:shadow-lg hover:shadow-violet-500/20'
                        }`}
                    disabled={step === 1}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6" /></svg>
                    이전으로
                </button>
                <div className="step-indicator m-0">
                    {step} <span className="step-total">/ {totalSteps}</span>
                </div>
            </div>

            <div className="question-header mt-2">
                <h3 className="part-title">{data.partTitle}</h3>
                <p className="subtitle">{data.subtitle}</p>
                <h2 className="question-text">{data.question}</h2>
            </div>

            <div className="options-container gap-4 mt-8 flex flex-col">
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
        </div>
    );
}

export default Question;

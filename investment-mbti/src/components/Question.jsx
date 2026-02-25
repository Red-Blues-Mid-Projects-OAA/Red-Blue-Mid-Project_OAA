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

            <div className="flex justify-end items-center mb-4">
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

            {/* 1단계에서는 뒤로 가기 버튼 숨김 또는 비활성화 처리 */}
            <div className={`mt-8 flex justify-center transition-opacity duration-300 ${step === 1 ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
                <button
                    onClick={onBack}
                    disabled={step === 1}
                    className="flex items-center gap-2 px-6 py-3 rounded-xl border border-gray-600 bg-gray-800/50 hover:bg-gray-700 hover:border-gray-400 text-gray-300 hover:text-white transition-all 
                               shadow-sm hover:shadow-md active:scale-95"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6" /></svg>
                    <span className="font-bold tracking-wider">뒤로가기</span>
                </button>
            </div>
        </div>
    );
}

export default Question;

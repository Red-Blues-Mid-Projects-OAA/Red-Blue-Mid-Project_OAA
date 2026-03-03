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
                <h2 className="question-text">{data.question}</h2>
            </div>

            {data.image && (
                <div className="question-image-container my-6 flex justify-center">
                    <img
                        src={data.image}
                        alt="question illustration"
                        className="question-image"
                    />
                </div>
            )}

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

            {/* 모든 단계에서 뒤로가기 표시 (1단계는 이전 페이지로 이동) */}
            <div className="mt-20 flex justify-center">
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
    );
}

export default Question;

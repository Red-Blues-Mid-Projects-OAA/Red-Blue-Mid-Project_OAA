import React, { useState, useEffect } from 'react';
import './Question.css';

function Question({ data, step, totalSteps, onAnswer }) {
    const [animationClass, setAnimationClass] = useState('animate-fade-in');

    // 문항이 바뀔 때마다 애니메이션 재등록
    useEffect(() => {
        setAnimationClass('');
        const timer = setTimeout(() => {
            setAnimationClass('animate-slide-up');
        }, 10);
        return () => clearTimeout(timer);
    }, [data.id]);

    const handleSelect = (value, optionText) => {
        onAnswer({
            questionId: data.id,
            type: data.type,
            value: value,
            selectedText: optionText
        });
    };

    const progress = (step / totalSteps) * 100;

    return (
        <div className={`question-container ${animationClass}`}>
            {/* Progress Bar */}
            <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${progress}%` }}></div>
            </div>
            <div className="step-indicator">
                {step} <span className="step-total">/ {totalSteps}</span>
            </div>

            <div className="question-header">
                <h3 className="part-title">{data.partTitle}</h3>
                <p className="subtitle">{data.subtitle}</p>
                <h2 className="question-text">{data.question}</h2>
            </div>

            <div className="options-container">
                <button
                    className="btn option-btn"
                    onClick={() => handleSelect(data.optionA.value, data.optionA.text)}
                >
                    {data.optionA.text}
                </button>
                <button
                    className="btn option-btn"
                    onClick={() => handleSelect(data.optionB.value, data.optionB.text)}
                >
                    {data.optionB.text}
                </button>
            </div>
        </div>
    );
}

export default Question;

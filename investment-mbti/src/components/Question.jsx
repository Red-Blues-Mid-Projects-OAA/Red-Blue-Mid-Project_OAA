import { useEffect, useState } from 'react';
import './Question.css';

function Question({ data, step, totalSteps, onAnswer, onBack }) {
    const [animationClass, setAnimationClass] = useState('animate-fade-in');

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
            value,
            selectedText: optionText,
            optionKey,
        });
    };

    const progress = (step / totalSteps) * 100;

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

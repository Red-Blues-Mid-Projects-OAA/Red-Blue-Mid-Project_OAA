import React from 'react';
import './Intro.css';

function Intro({ onStart }) {
    return (
        <div className="intro-container animate-fade-in flex-column flex-center">
            <div className="intro-content">
                <div className="badge animate-slide-up" style={{ animationDelay: '0.1s' }}>
                    금융 초보자를 위한
                </div>
                <h1 className="title animate-slide-up" style={{ animationDelay: '0.2s' }}>
                    <span className="title-label">투자 성향 MBTI 테스트:</span><br />
                    나는 어떤 유형의 개미일까? 🐜
                </h1>
                <p className="subtitle animate-slide-up" style={{ animationDelay: '0.3s' }}>
                    개미라고 다 같은 개미가 아니다!<br />
                    작은 흔들림에도 불안한 개미인가, 폭락장에서도 버티는 개미인가?
                </p>

                <div className="btn-wrapper animate-slide-up" style={{ animationDelay: '0.4s' }}>
                    <button className="btn btn-primary start-btn" onClick={onStart}>
                        테스트 시작하기 🚀
                    </button>
                </div>
            </div>
        </div>
    );
}

export default Intro;

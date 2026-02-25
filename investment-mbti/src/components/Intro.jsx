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
                    투자 MBTI :<br />
                    당신의 <span className="highlight">'돈 그릇'</span> 유형 테스트
                </h1>
                <p className="subtitle animate-slide-up" style={{ animationDelay: '0.3s' }}>
                    위험을 즐기는 사냥꾼인가, 안전을 쫓는 거북이인가?<br />
                    간단한 12문항으로 당신의 투자 페르소나를 확인하세요.
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

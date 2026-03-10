/*
 * 이 파일은 소개 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React from 'react';
import './Intro.css';

/**
 * 소개 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function Intro({ onStart }) {
    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
    return (
        <div className="intro-container animate-fade-in flex-column flex-center">
            <div className="intro-content">
                <div className="intro-main">
                    <div className="intro-logo-wrap animate-slide-up" style={{ animationDelay: '0.05s' }}>
                        <div className="intro-logo-shell">
                            <img
                                className="intro-logo"
                                src="/images/rogomain-transparent.png"
                                alt="ROGO logo"
                            />
                        </div>
                    </div>

                    <div className="intro-copy">
                        <div className="badge animate-slide-up" style={{ animationDelay: '0.1s' }}>
                            금융 초보자를 위한
                        </div>

                        <div className="intro-text-group">
                            <h1 className="title animate-slide-up" style={{ animationDelay: '0.2s' }}>
                                <span className="title-label">투자 성향 MBTI 테스트:</span><br />
                                <span className="title-main title-main--desktop">나는 어떤 유형의 개미일까? 🐜</span>
                                <span className="title-main title-main--mobile">나는 어떤 유형의 개미일까? 🐜</span>
                            </h1>

                            <p className="subtitle animate-slide-up" style={{ animationDelay: '0.3s' }}>
                                개미라고 다 같은 개미가 아니다!<br />
                                <span className="subtitle-line subtitle-line--desktop">작은 흔들림에도 불안한 개미인가, 폭락장에서도 버티는 개미인가?</span>
                                <span className="subtitle-line subtitle-line--mobile">
                                    작은 흔들림에도 불안한 개미인가,<br />
                                    폭락장에서도 버티는 개미인가?
                                </span>
                            </p>
                        </div>
                    </div>
                </div>

                <div className="btn-wrapper animate-slide-up" style={{ animationDelay: '0.4s' }}>
                    <button className="btn btn-primary btn-3d start-btn" onClick={onStart}>
                        테스트 시작하기 🚀
                    </button>
                </div>
            </div>
        </div>
    );
}

export default Intro;

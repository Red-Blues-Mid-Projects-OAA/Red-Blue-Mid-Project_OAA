/*
 * 이 파일은 로딩 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React from 'react';
import './Loading.css';

/**
 * 로딩 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function Loading() {
    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
    return (
        <div className="loading-container animate-fade-in flex-column flex-center">
            <div className="cube-wrapper">
                <div className="cube-face front"></div>
                <div className="cube-face back"></div>
                <div className="cube-face right"></div>
                <div className="cube-face left"></div>
                <div className="cube-face top"></div>
                <div className="cube-face bottom"></div>
            </div>

            <h2 className="loading-text">
                당신의 투자 페르소나를<br />
                <span className="highlight-load">분석하고 있습니다...</span>
            </h2>
            <p className="loading-sub">*잠시만 기다려 주세요*</p>
        </div>
    );
}

export default Loading;

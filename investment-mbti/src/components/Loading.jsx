import React from 'react';
import './Loading.css';

function Loading() {
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

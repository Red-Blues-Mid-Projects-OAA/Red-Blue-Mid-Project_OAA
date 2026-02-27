import React, { useEffect, useState } from 'react';
import './DashboardResult.css';

function MbtiBarChart({ dimension, score, leftLabel, rightLabel, gradient, colorClass = '' }) {
    const [animatedScore, setAnimatedScore] = useState(50);
    const [isHovered, setIsHovered] = useState(false);
    const safeScore = Math.max(0, Math.min(100, Number(score) || 0));

    useEffect(() => {
        const timer = setTimeout(() => {
            setAnimatedScore(safeScore);
        }, 90);
        return () => clearTimeout(timer);
    }, [safeScore]);

    const fillStyle = gradient ? { background: gradient } : undefined;
    const leftPct = Math.max(0, Math.min(100, 100 - animatedScore));
    const rightPct = Math.max(0, Math.min(100, animatedScore));

    return (
        <div
            className={`mbti-row ${isHovered ? 'mbti-row-hovered' : ''}`}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            {/* 타이틀: 바 위 중앙 */}
            <div className="mbti-row-title">
                <h4>{dimension}</h4>
            </div>

            {/* 바 + 라벨: 라벨은 바 양측 끝 외부 */}
            <div className="mbti-bar-outer">
                <span className="mbti-outer-label mbti-outer-label-left">{leftLabel}</span>

                <div className="mbti-track-wrap">
                    <div className="mbti-track">
                        <div
                            className={`mbti-fill ${colorClass}`.trim()}
                            style={{ ...fillStyle, width: `${rightPct}%` }}
                        />
                    </div>
                    {/* 호버 시 퍼센트 표시 */}
                    {isHovered && (
                        <div className="mbti-hover-tooltip" style={{ left: `${rightPct}%` }}>
                            {rightPct.toFixed(0)}%
                        </div>
                    )}
                </div>

                <span className="mbti-outer-label mbti-outer-label-right">{rightLabel}</span>
            </div>

            {/* 하단 퍼센트 */}
            <div className="mbti-row-foot">
                <span>{leftPct.toFixed(0)}%</span>
                <span>{rightPct.toFixed(0)}%</span>
            </div>
        </div>
    );
}

export default MbtiBarChart;

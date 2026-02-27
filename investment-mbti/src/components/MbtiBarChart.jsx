import React, { useEffect, useState } from 'react';
import './DashboardResult.css';

function MbtiBarChart({ dimension, score, leftLabel, rightLabel, gradient, colorClass = '' }) {
    const [animatedScore, setAnimatedScore] = useState(50);
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
        <div className="mbti-row">
            <div className="mbti-row-head">
                <h4>{dimension}</h4>
                <span className="mbti-right-score">{rightPct.toFixed(0)}%</span>
            </div>

            <div className="mbti-track">
                <div
                    className={`mbti-fill ${colorClass}`.trim()}
                    style={{ ...fillStyle, width: `${rightPct}%` }}
                />
            </div>

            <div className="mbti-row-foot">
                <span>{leftLabel} {leftPct.toFixed(0)}%</span>
                <span>{rightLabel} {rightPct.toFixed(0)}%</span>
            </div>
        </div>
    );
}

export default MbtiBarChart;


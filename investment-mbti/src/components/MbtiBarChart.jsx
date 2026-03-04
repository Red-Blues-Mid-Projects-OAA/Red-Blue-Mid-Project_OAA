import React, { useEffect, useState } from 'react';
import './DashboardResult.css';

function MbtiBarChart({ dimension, score, leftLabel, rightLabel, gradient, colorClass = '' }) {
    const [animatedScore, setAnimatedScore] = useState(50);

    // 최종 수치로 방향 결정 (애니메이션 도중 방향 전환 방지)
    const safeScore = Math.max(0, Math.min(100, Number(score) || 0));
    const leftDominant = (100 - safeScore) >= safeScore;

    useEffect(() => {
        const timer = setTimeout(() => setAnimatedScore(safeScore), 90);
        return () => clearTimeout(timer);
    }, [safeScore]);

    const leftPct = Math.round(100 - animatedScore);
    const rightPct = Math.round(animatedScore);

    // 우세한 쪽 채움 너비와 팁 위치
    const fillPct = leftDominant ? leftPct : rightPct;
    const tipLeft = leftDominant ? fillPct : (100 - fillPct);

    // RTL 방향은 그라디언트 반전 (wall=밝음, tip=진함 유지)
    const fillGradient = leftDominant
        ? gradient
        : gradient?.replace('90deg', '270deg');

    return (
        <div className="mbti-row">
            {/* 중앙 타이틀 */}
            <div className="mbti-row-title">
                <h4>{dimension}</h4>
            </div>

            <div className="mbti-bar-outer">
                {/* 왼쪽: 라벨 + % (우세하면 강조색) */}
                <div className={`mbti-label-group mbti-label-left${leftDominant ? ' mbti-dominant' : ''}`}>
                    <span className="mbti-label-name">{leftLabel}</span>
                    <span className="mbti-label-pct">{leftPct}%</span>
                </div>

                {/* 바 트랙 + 끝 지점 마커 */}
                <div className="mbti-track-wrap">
                    <div className="mbti-track">
                        <div
                            className={`mbti-fill${!leftDominant ? ' mbti-fill-rtl' : ''} ${colorClass}`.trim()}
                            style={{ background: fillGradient, width: `${fillPct}%` }}
                        />
                    </div>
                    {/* 팁 마커: 채움 끝 지점에 위치 */}
                    <span className="mbti-tip" style={{ left: `${tipLeft}%` }} />
                </div>

                {/* 오른쪽: 라벨 + % (우세하면 강조색) */}
                <div className={`mbti-label-group mbti-label-right${!leftDominant ? ' mbti-dominant' : ''}`}>
                    <span className="mbti-label-name">{rightLabel}</span>
                    <span className="mbti-label-pct">{rightPct}%</span>
                </div>
            </div>
        </div>
    );
}

export default MbtiBarChart;

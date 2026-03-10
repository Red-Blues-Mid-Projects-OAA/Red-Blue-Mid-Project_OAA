/*
 * 이 파일은 MBTI 막대 차트 관련 프론트엔드 로직을 담고 있습니다.
 * 상단 상수와 보조 함수가 표시용 값을 만들고, 상태와 props에서 파생한 값이 마지막 JSX에 연결되므로 데이터가 화면 요소로 바뀌는 흐름을 위에서 아래로 따라가면 됩니다.
 */

import React, { useEffect, useState } from 'react';
import './DashboardResult.css';

/**
 * MBTI 막대 차트 컴포넌트가 화면 상태와 렌더링을 담당합니다.
 */

function MbtiBarChart({ dimension, score, leftLabel, rightLabel, gradient, colorClass = '' }) {
    // 화면에서 계속 바뀌는 값을 state로 보관합니다.
    const [animatedScore, setAnimatedScore] = useState(50);

    // 최종 수치로 방향 결정 (애니메이션 도중 방향 전환 방지)
    const safeScore = Math.max(0, Math.min(100, Number(score) || 0));
    const leftDominant = (100 - safeScore) >= safeScore;

    // 화면이 열리거나 특정 값이 바뀔 때 필요한 부수 작업을 실행합니다.
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

    // 마지막에 현재 상태를 반영한 화면 구조를 JSX로 반환합니다.
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

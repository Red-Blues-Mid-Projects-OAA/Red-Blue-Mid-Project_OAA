# 📈 Red & Blue: 최적 자산 배분 알고리즘을 활용한 퀀트 투자 기법

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.14+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB">
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">
</div>

---

## 📋 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [팀원 구성 및 역할](#2-팀원-구성-및-역할)
3. [기술 스택](#3-기술-스택)
4. [시스템 아키텍처 및 파이프라인](#4-시스템-아키텍처-및-파이프라인)
5. [주요 기능](#5-주요-기능)
6. [화면 구성](#6-화면-구성)
7. [디렉토리 구조](#7-디렉토리-구조)
8. [브랜치 전략 및 협업 방식](#8-브랜치-전략-및-협업-방식)
9. [설치 및 실행 방법](#9-설치-및-실행-방법)

---

## 1. 📝 프로젝트 소개
**Red & Blue**는 개인의 심리적 위험 감수 성향(투자 MBTI)과 머신러닝 예측 모델을 융합하여, 사용자 맞춤형 S&P 500 주식 포트폴리오를 제안하는 풀스택 웹 서비스입니다. 
단순히 직관에 의존하는 투자가 아닌, 과거 금융 데이터와 거시경제 지표에 기반한 알고리즘(Ensemble ML & CAPM)을 통해 효율적이고 체계적인 투자 전략을 제공하는 것을 목표로 합니다.

---

## 2. 👥 팀원 구성 및 역할
| 이름 | 역할 | 담당 업무 | Github |
|:---:|:---|:---|:---|
| **윤재정(조장)** | Data & ML(Main) | S&P 500 데이터 크롤링, 파생 피처 엔지니어링, 4종 ML 개별 모델 및 앙상블 훈련 파이프라인 구축 | [@githubID](https://github.com/) |
| **이대한** | Data & ML(support) | 사용자 Risk Profiling 로직 작성, CAPM 기반 포트폴리오 최적화 비중 연산, 데이터 서빙 API 구현 | [@githubID](https://github.com/) |
| **이윤원** | Database | React/Vite 기반 웹 SPA 개발, MBTI 설문 동적 UI 및 Recharts 기반 데이터 시각화(PieChart, Gauge) | [@githubID](https://github.com/) |
| **이우정** | Backend | 사용자 Risk Profiling 로직 작성, CAPM 기반 포트폴리오 최적화 비중 연산, 데이터 서빙 API 구현 | [@githubID](https://github.com/) |
| **남기혁** | Frontend | React/Vite 기반 웹 SPA 개발, MBTI 설문 동적 UI 및 Recharts 기반 데이터 시각화(PieChart, Gauge) | [@githubID](https://github.com/) |

---

## 3. 🛠 상세 기술 스택 및 알고리즘 (Tech Stack & Algorithms)
### 🧠 Machine Learning & Data Science (머신러닝 및 데이터 사이언스)
가장 핵심이 되는 예측 파이프라인으로, 단일 모델이 아닌 교차 검증 및 앙상블 기법을 활용했습니다.

■ **분류 알고리즘 (Classification Models):**

- **Logistic Regression (로지스틱 회귀)**: 선형 기반의 베이스라인 이진 분류 모델
- **Random Forest (랜덤 포레스트)**: 의사결정나무(Decision Tree) 기반의 배깅(Bagging) 앙상블 모델
- **SVM (Support Vector Machine):** 비선형 데이터 경계면(Hyperplane)을 찾는 강력한 분류 모델
- **XGBoost (eXtreme Gradient Boosting)**: 트리 기반의 그래디언트 부스팅(Boosting) 최신 알고리즘

■ **고급 ML 기법 (Advanced ML Techniques):**

- **Ensemble Learning (앙상블 학습)**: 4가지 개별 모델의 예측 결과를 결합(Voting 또는 Stacking)하여 최종 예측의 안정성(Robustness)과 정확도를 극대화 (ensemble.py)
- **Hyperparameter Optimization (하이퍼파라미터 최적화)**: 각 종목별로 모델이 최상의 성능을 내도록 파라미터를 자동 탐색 및 튜닝 (결과물 _best_params.json 활용)
- **Feature Importance (피처 중요도 분석)**: 모델의 예측에 가장 큰 영향을 미친 변수(Macro, Momentum 등)를 추출하여 블랙박스 모델의 설명력(XAI) 부여 (importance.py)

### 📈 Financial Engineering & Optimization (금융 공학 및 포트폴리오 알고리즘)
단순한 예측을 넘어, 실제 투자 가능한 비율로 변환하기 위한 수학적/금융 알고리즘입니다.

- **CAPM (Capital Asset Pricing Model, 자본자산가격결정모형)**: 무위험 수익률과 시장 수익률을 바탕으로 개별 자산의 기대 수익률을 산출하는 핵심 금융 알고리즘 (capm.py)
- **Mean-Variance Portfolio Optimization (마코위츠의 평균-분산 최적화)**: 사용자의 위험 감수 성향에 맞춰, 기대 수익률은 극대화하고 리스크(변동성)는 최소화하는 최적의 종목 비중(Weight)을 산출하는 알고리즘 (portfolio_optimizer.py)
- **Risk Profiling Scoring Algorithm**: 12가지 다면적 질문을 통해 사용자의 위험 회피도(Risk Aversion)를 수치화하고 엠블럼을 부여하는 자체 평가 알고리즘 (risk_profile.py)

### 📊 Data Engineering & Preprocessing (데이터 엔지니어링 알고리즘)
원시 데이터를 모델이 학습할 수 있는 파생 변수로 변환하는 고도화된 전처리 기술입니다.

- **Market Regime Detection (시장 국면 판별)**: 상승장, 하락장, 횡보장 등 현재 시장의 상태(Regime)를 수학적으로 감지하여 모델에 반영 (adjust_regime.py)
- **EWMA (Exponentially Weighted Moving Average, 지수이동평균)**: 최근 데이터에 더 큰 가중치를 두어 주가의 추세를 부드럽게 추적하는 알고리즘 (calculate_ewma.py)
- **Logarithmic Returns (로그 수익률 계산)**: 금융 시계열 데이터의 정규성을 확보하기 위한 연속 복리 수익률 변환 (calculate_log_returns.py)
- **Feature Engineering Domain**: 거시경제(Macro), 모멘텀(Momentum), 변동성(Volatility), 거래량(Volume) 기반의 파생 지표 생성 기술

### ⚙️ Backend & API (백엔드)
Framework: Python 기반의 빠르고 비동기 처리가 가능한 웹 프레임워크 (FastAPI 또는 Flask)

- **Data Serving**: 계산된 최적 포트폴리오 배열과 수익률 차트 데이터를 프론트엔드 형식에 맞춰 가공하는 로직 (chart_data_provider.py, real_data_provider.py)

### 🎨 Frontend & UI/UX (프론트엔드)
- **Core**: React.js (v18+), JSX, JavaScript (ES6+)
- **Build Tool**: Vite (빠른 HMR 및 빌드 제공)
- **Data Visualization**: Recharts 또는 Chart.js 기반의 동적 데이터 시각화 (도넛 차트, 누적 수익률 라인 차트, 게이지 차트 구현)
- **Styling**: CSS3 (반응형 웹 디자인 및 컴포넌트별 모듈화)
- **Linting**: ESLint (코드 컨벤션 유지)

### 🛠 Infra & DevOps (인프라 및 자동화)
- **CI/CD Automation**: GitHub Actions 기반의 Crontab 스케줄링. 매일 장 마감 후 자동으로 파이프라인(daily-db-load.yml)을 돌려 주가를 스크래핑하고 DB를 최신화
- **Data Source APIs**: yfinance API (S&P 500 개별 종목 주가 및 거래량), FRED API (거시경제 지표)
- **Batch Processing**: Shell Scripting (run_pipeline_300.sh)을 통한 대용량 데이터 일괄 자동 학습 시스템
- **Version Control**: Git & GitHub (Feature Branch 병렬 협업)

---

## 4. ⚙️ 시스템 아키텍처 및 파이프라인

프로젝트는 데이터 수집부터 웹 서빙까지 4단계의 유기적인 파이프라인으로 설계되었습니다.

```mermaid
flowchart TD
    subgraph Data Pipeline
        A1[yfinance API<br>S&P 500 Data] --> B(Feature Engineering<br>Macro/Momentum/Volatility/Volume)
        A2[FRED API<br>Macroeconomic Data] --> B
        B --> C[(Master DB)]
    end
    
    subgraph ML Pipeline
        C --> D{Ensemble Modeling}
        D -->|LogReg| E
        D -->|RandomForest| E
        D -->|SVM| E
        D -->|XGBoost| E
        E[Artifacts<br>Metrics & Expected Returns]
    end
    
    subgraph Backend
        E --> F[API Server]
        F --> G[Portfolio Optimizer<br>CAPM]
        F --> H[Risk Profiler]
    end
    
    subgraph Frontend
        H --> I[MBTI Survey UI]
        I --> G
        G --> J[Dashboard<br>Visualization]
    end
```

---

## 5. 💡 주요 기능
### ■ 일일 데이터 자동 갱신 (GitHub Actions)
- 매일 워크플로우(daily-db-load.yml)를 트리거하여 최신 S&P 500 주가 데이터를 스크래핑하고 마스터 데이터셋을 최신화합니다.

### ■ 머신러닝 앙상블 예측 (ML)
- 로지스틱 회귀, 랜덤 포레스트, SVM, XGBoost 4개의 모델을 학습 및 결합(ensemble.py)하여 각 종목의 상승/하락 확률을 예측합니다. 수백 개 종목은 쉘 스크립트(run_pipeline_300.sh)를 통해 일괄 처리됩니다.

### ■ 투자 MBTI 분석 (Risk Profiling)
- 12문항의 설문을 통해 사용자의 투자 성향을 분석하고, 어울리는 캐릭터(Yolo, Worker 등)를 부여합니다.

### 포트폴리오 최적화 (CAPM)
- 사용자의 위험 성향과 모델이 예측한 종목의 기대 수익률을 융합하여, 가장 안정적이고 효율적인 투자 비중(Weight)을 계산하여 추천합니다.

---

## 6. 🖥 화면 구성 --> 이미지 교체 관련 찾아야 함

| 메인 및 설문조사 | 투자 성향 결과 대시보드 | 포트폴리오 추천 차트 |
| :---: | :---: | :---: |
| <img src="https://via.placeholder.com/250x150.png?text=Intro+UI" width="250"> | <img src="https://via.placeholder.com/250x150.png?text=MBTI+Result" width="250"> | <img src="https://via.placeholder.com/250x150.png?text=Portfolio+Chart" width="250"> |
| 12가지 문항을 통한<br>투자 성향 진단 진행 | MBTI에 매칭되는 캐릭터 및<br>위험 감수도(Loss Slider) 조정 | 개인화된 종목 비중<br>파이 차트 시각화 |

---

## 7. 📂 디렉토리 구조
MSA(Microservices Architecture) 형태를 지향하여 각 역할을 완벽히 분리했습니다.

```text
📦 Red-Blue-Mid-Project_OAA
 ┣ 📂 .github/workflows       # [자동화] 일일 데이터 수집 및 DB 갱신 CI/CD
 ┣ 📂 DB/                     # [데이터] 기초 자산 데이터 수집 및 병합 파이프라인
 ┃ ┣ 📜 build_master_dataset.py
 ┃ ┣ 📜 calculate_ewma.py
 ┃ ┗ 📂 utils/sp500_scraper.py
 ┣ 📂 Classification/         # [머신러닝] 피처 생성, 개별 모델 훈련 및 앙상블
 ┃ ┣ 📂 Preprocessing/        # 매크로, 모멘텀, 변동성 등 학습 변수 생성
 ┃ ┣ 📂 models/               # LogReg, RF, SVM, XGB 개별 모델 정의
 ┃ ┣ 📂 ensemble/             # 모델 결과 결합 알고리즘
 ┃ ┗ 📂 artifacts/            # [산출물] 각 종목별 학습 결과 및 매핑 요약 데이터
 ┣ 📂 investment-mbti-back/   # [백엔드] FastAPI 서빙 및 포트폴리오 산출 로직
 ┃ ┣ 📜 main.py
 ┃ ┣ 📜 portfolio_optimizer.py# 포트폴리오 비중 최적화 로직
 ┃ ┗ 📜 risk_profile.py       # 투자 MBTI 분류 로직
 ┣ 📂 investment-mbti/        # [프론트엔드] React + Vite 웹 애플리케이션
 ┃ ┣ 📂 public/images/        # 시각화 리소스 (YOLO, Slave 등 캐릭터 에셋)
 ┃ ┣ 📂 src/components/       # 설문조사, 차트 시각화 대시보드 컴포넌트
 ┃ ┗ 📂 src/constants/        # 설문 문항 및 주식 기초 상수 데이터
 ┗ 📂 scripts/                # 전체 파이프라인 통합 일괄 실행 스크립트 모음
```
---

## 8. 🔀 브랜치 전략 및 협업 방식

모놀리식 환경에서의 충돌을 막기 위해 GitHub Flow 기반의 Feature 브랜치 전략을 사용했습니다.
gitGraph
    commit id: "Init"
    branch dev
    checkout dev
    commit id: "Set skeleton"
    
    branch feat/ml-pipeline
    checkout feat/ml-pipeline
    commit id: "Add RF, XGB models"
    checkout dev
    merge feat/ml-pipeline
    
    branch feat/backend
    checkout feat/backend
    commit id: "Build CAPM optimizer"
    checkout dev
    merge feat/backend
    
    branch feat/frontend
    checkout feat/frontend
    commit id: "Create PieChart & Dashboard"
    checkout dev
    merge feat/frontend
    
    checkout main
    merge dev id: "Release v1.0"

- **main**: 배포 가능한 안정적인 코드가 유지되는 브랜치
- **dev**: 다음 출시 버전을 위해 개발 중인 코드가 모이는 브랜치
- **feat/OOO**: 기능 개발 브랜치. 개발 완료 후 PR(Pull Request) 리뷰를 거쳐 dev로 병합

---

## 9. 🚀 설치 및 실행 방법
### ■ 클라이언트 (Frontend) 실행
cd investment-mbti
npm install
npm run dev

### ■ 서버 (Backend) 실행
cd investment-mbti-back
pip install -r requirements.txt
python main.py

### ■ 머신러닝 파이프라인 (재학습 시)
cd scripts
bash run_pipeline_300.sh

---

### 💡 활용 가이드:
1. **GitHub 복사/붙여넣기:** 위 코드를 처음부터 끝까지 그대로 긁어서 `README.md`에 붙여넣으세요.
2. **내용 수정:** * **[팀원 구성]** 섹션의 이름과 깃허브 링크를 본인과 팀원에 맞게 수정하세요.
   * **[화면 구성]** 섹션에서 `https://via.placeholder.com/...` 이라고 되어 있는 부분의 주소를 **실제 프로젝트 캡쳐본 이미지 링크**(깃허브 이슈나 리드미 에디터에 이미지를 드래그 앤 드롭하면 나오는 URL)로 교체하시면 완벽해집니다.

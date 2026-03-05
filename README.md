# 📈 Red & Blue: 투자 MBTI 기반 S&P 500 AI 포트폴리오 추천 서비스

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB">
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">
</div>

<br>

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

<br>

## 1. 📝 프로젝트 소개
**Red & Blue**는 개인의 심리적 위험 감수 성향(투자 MBTI)과 머신러닝 예측 모델을 융합하여, 사용자 맞춤형 S&P 500 주식 포트폴리오를 제안하는 풀스택 웹 서비스입니다. 
단순히 직관에 의존하는 투자가 아닌, 과거 금융 데이터와 거시경제 지표에 기반한 알고리즘(Ensemble ML & CAPM)을 통해 효율적이고 체계적인 투자 전략을 제공하는 것을 목표로 합니다.

<br>

## 2. 👥 팀원 구성 및 역할
| 이름 | 역할 | 담당 업무 | Github |
|:---:|:---|:---|:---|
| **윤재정(조장)** | Data & ML(Main) | S&P 500 데이터 크롤링, 파생 피처 엔지니어링, 4종 ML 개별 모델 및 앙상블 훈련 파이프라인 구축 | [@githubID](https://github.com/) |
| **이대한** | Data & ML(support) | 사용자 Risk Profiling 로직 작성, CAPM 기반 포트폴리오 최적화 비중 연산, 데이터 서빙 API 구현 | [@githubID](https://github.com/) |
| **이윤원** | Database | React/Vite 기반 웹 SPA 개발, MBTI 설문 동적 UI 및 Recharts 기반 데이터 시각화(PieChart, Gauge) | [@githubID](https://github.com/) |
| **이우정** | Backend | 사용자 Risk Profiling 로직 작성, CAPM 기반 포트폴리오 최적화 비중 연산, 데이터 서빙 API 구현 | [@githubID](https://github.com/) |
| **남기혁** | Frontend | React/Vite 기반 웹 SPA 개발, MBTI 설문 동적 UI 및 Recharts 기반 데이터 시각화(PieChart, Gauge) | [@githubID](https://github.com/) |

<br>

## 3. 🛠 기술 스택
### Frontend
- **Framework:** React.js (Vite)
- **Styling & UI:** CSS3, Recharts (데이터 시각화)
- **Asset:** Custom Character Images (Yolo, Slave, Fire 등)

### Backend
- **Framework:** Python API Framework (FastAPI / Flask)
- **Algorithm:** CAPM (자본자산가격결정모형), Markowitz Portfolio Optimization

### Machine Learning & Data
- **Library:** Scikit-learn, XGBoost, Pandas, Numpy
- **Data Source:** yfinance (주가 및 VIX, 거시경제 지표)

### Infra & DevOps
- **CI/CD:** GitHub Actions (일일 DB 자동 업데이트 배치)
- **Version Control:** Git, GitHub

<br>

## 4. ⚙️ 시스템 아키텍처 및 파이프라인

프로젝트는 데이터 수집부터 웹 서빙까지 4단계의 유기적인 파이프라인으로 설계되었습니다.

```mermaid
flowchart TD
    subgraph Data Pipeline
        A[yfinance API<br>S&P 500 Data] --> B(Feature Engineering<br>Macro/Momentum/Volatility/Volume)
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
        G --> J[Dashboard<br>Visulization]
    end

5. 💡 주요 기능
일일 데이터 자동 갱신 (GitHub Actions): 매일 워크플로우(daily-db-load.yml)를 트리거하여 최신 S&P 500 주가 데이터를 스크래핑하고 마스터 데이터셋을 최신화합니다.

머신러닝 앙상블 예측 (ML): 로지스틱 회귀, 랜덤 포레스트, SVM, XGBoost 4개의 모델을 학습 및 결합(ensemble.py)하여 각 종목의 상승/하락 확률을 예측합니다. 수백 개 종목은 쉘 스크립트(run_pipeline_300.sh)를 통해 일괄 처리됩니다.

투자 MBTI 분석 (Risk Profiling): 12문항의 설문을 통해 사용자의 투자 성향을 분석하고, 어울리는 캐릭터(Yolo, Worker 등)를 부여합니다.

포트폴리오 최적화 (CAPM): 사용자의 위험 성향과 모델이 예측한 종목의 기대 수익률을 융합하여, 가장 안정적이고 효율적인 투자 비중(Weight)을 계산하여 추천합니다.

6. 🖥 화면 구성
메인 및 설문조사,투자 성향 결과 대시보드,포트폴리오 추천 차트
"<img src=""https://www.google.com/search?q=https://via.placeholder.com/250x150.png%3Ftext%3DIntro%2BUI"" width=""250"">","<img src=""https://www.google.com/search?q=https://via.placeholder.com/250x150.png%3Ftext%3DMBTI%2BResult"" width=""250"">","<img src=""https://www.google.com/search?q=https://via.placeholder.com/250x150.png%3Ftext%3DPortfolio%2BChart"" width=""250"">"
12가지 문항을 통한투자 성향 진단 진행,MBTI에 매칭되는 캐릭터 및위험 감수도(Loss Slider) 조정,개인화된 종목 비중파이 차트 시각화

7. 📂 디렉토리 구조
MSA(Microservices Architecture) 형태를 지향하여 각 역할을 완벽히 분리했습니다.
Red-Blue-Mid-Project_OAA/
├── .github/workflows/         # 일일 데이터 수집 CI/CD 배치
├── DB/                        # 데이터 수집 및 병합 로직
│   ├── build_master_dataset.py
│   └── utils/sp500_scraper.py
├── Classification/            # ML 모델링 및 아티팩트
│   ├── Preprocessing/         # 거시, 모멘텀 등 파생 변수 생성
│   ├── models/                # LogReg, RF, SVM, XGB 개별 모델
│   ├── ensemble/              # 모델 결합 최적화
│   └── artifacts/             # [산출물] 종목별 예측 결과 파일
├── investment-mbti-back/      # API 및 포트폴리오 연산 서버
│   ├── main.py
│   ├── portfolio_optimizer.py
│   └── risk_profile.py
├── investment-mbti/           # 프론트엔드 React/Vite 앱
│   ├── src/components/        # Dashboard, PieChart, Question 등
│   └── public/images/         # 캐릭터 및 에셋 이미지
└── scripts/                   # 전체 시스템 일괄 구동 쉘 스크립트

8. 🔀 브랜치 전략 및 협업 방식
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


Gemini의 응답
제시해주신 티스토리 블로그의 가이드(개발자 포트폴리오용 표준 README 목차)는 채용 담당자나 다른 개발자가 프로젝트를 파악하기에 훨씬 체계적이고 가독성이 높은 훌륭한 구조입니다.

해당 블로그에서 강조하는 **'목적 - 아키텍처 - 주요 기능 - 기술 스택 - 개발 및 협업 방식'**의 흐름을 완벽하게 반영하여, 업로드해주신 Red-Blue 프로젝트 코드를 바탕으로 한 최종 README.md 파일 텍스트를 작성해 드립니다.

복사해서 바로 깃허브 README.md에 붙여넣으시면, 구조화된 다이어그램과 뱃지들이 아름답게 렌더링됩니다.

Markdown
# 📈 Red & Blue: 투자 MBTI 기반 S&P 500 AI 포트폴리오 추천 서비스

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB">
  <img src="https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white">
</div>

<br>

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

<br>

## 1. 📝 프로젝트 소개
**Red & Blue**는 개인의 심리적 위험 감수 성향(투자 MBTI)과 머신러닝 예측 모델을 융합하여, 사용자 맞춤형 S&P 500 주식 포트폴리오를 제안하는 풀스택 웹 서비스입니다. 
단순히 직관에 의존하는 투자가 아닌, 과거 금융 데이터와 거시경제 지표에 기반한 알고리즘(Ensemble ML & CAPM)을 통해 효율적이고 체계적인 투자 전략을 제공하는 것을 목표로 합니다.

<br>

## 2. 👥 팀원 구성 및 역할
| 이름 | 역할 | 담당 업무 | Github |
|:---:|:---|:---|:---|
| **윤재정(조장)** | Data & ML(main) | S&P 500 데이터 크롤링, 파생 피처 엔지니어링, 4종 ML 개별 모델 및 앙상블 훈련 파이프라인 구축 | [@githubID](https://github.com/) |
| **이대한** | Data & ML(suport) | 사용자 Risk Profiling 로직 작성, CAPM 기반 포트폴리오 최적화 비중 연산, 데이터 서빙 API 구현 | [@githubID](https://github.com/) |
| **이윤원** | Database | React/Vite 기반 웹 SPA 개발, MBTI 설문 동적 UI 및 Recharts 기반 데이터 시각화(PieChart, Gauge) | [@githubID](https://github.com/) |
| **이우정** | Backend | React/Vite 기반 웹 SPA 개발, MBTI 설문 동적 UI 및 Recharts 기반 데이터 시각화(PieChart, Gauge) | [@githubID](https://github.com/) |
| **남기혁** | Frontend | React/Vite 기반 웹 SPA 개발, MBTI 설문 동적 UI 및 Recharts 기반 데이터 시각화(PieChart, Gauge) | [@kiekuu](https://github.com/) |

<br>

## 3. 🛠 기술 스택
### Frontend
- **Framework:** React.js (Vite)
- **Styling & UI:** CSS3, Recharts (데이터 시각화)
- **Asset:** Custom Character Images (Yolo, Slave, Fire 등)

### Backend
- **Framework:** Python API Framework (FastAPI / Flask)
- **Algorithm:** CAPM (자본자산가격결정모형), Markowitz Portfolio Optimization

### Machine Learning & Data
- **Library:** Scikit-learn, XGBoost, Pandas, Numpy
- **Data Source:** yfinance (주가 및 VIX, 거시경제 지표)

### Infra & DevOps
- **CI/CD:** GitHub Actions (일일 DB 자동 업데이트 배치)
- **Version Control:** Git, GitHub

<br>

## 4. ⚙️ 시스템 아키텍처 및 파이프라인

프로젝트는 데이터 수집부터 웹 서빙까지 4단계의 유기적인 파이프라인으로 설계되었습니다.

```mermaid
flowchart TD
    subgraph Data Pipeline
        A[yfinance API<br>S&P 500 Data] --> B(Feature Engineering<br>Macro/Momentum/Volatility/Volume)
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
        G --> J[Dashboard<br>Visulization]
    end
5. 💡 주요 기능
일일 데이터 자동 갱신 (GitHub Actions): 매일 워크플로우(daily-db-load.yml)를 트리거하여 최신 S&P 500 주가 데이터를 스크래핑하고 마스터 데이터셋을 최신화합니다.

머신러닝 앙상블 예측 (ML): 로지스틱 회귀, 랜덤 포레스트, SVM, XGBoost 4개의 모델을 학습 및 결합(ensemble.py)하여 각 종목의 상승/하락 확률을 예측합니다. 수백 개 종목은 쉘 스크립트(run_pipeline_300.sh)를 통해 일괄 처리됩니다.

투자 MBTI 분석 (Risk Profiling): 12문항의 설문을 통해 사용자의 투자 성향을 분석하고, 어울리는 캐릭터(Yolo, Worker 등)를 부여합니다.

포트폴리오 최적화 (CAPM): 사용자의 위험 성향과 모델이 예측한 종목의 기대 수익률을 융합하여, 가장 안정적이고 효율적인 투자 비중(Weight)을 계산하여 추천합니다.

6. 🖥 화면 구성
메인 및 설문조사	투자 성향 결과 대시보드	포트폴리오 추천 차트
<img src="https://www.google.com/search?q=https://via.placeholder.com/250x150.png%3Ftext%3DIntro%2BUI" width="250">	<img src="https://www.google.com/search?q=https://via.placeholder.com/250x150.png%3Ftext%3DMBTI%2BResult" width="250">	<img src="https://www.google.com/search?q=https://via.placeholder.com/250x150.png%3Ftext%3DPortfolio%2BChart" width="250">
12가지 문항을 통한


투자 성향 진단 진행

MBTI에 매칭되는 캐릭터 및


위험 감수도(Loss Slider) 조정

개인화된 종목 비중


파이 차트 시각화

(※ 위 Placeholder 이미지를 실제 프로젝트 화면 캡처본으로 교체하세요.)

7. 📂 디렉토리 구조
MSA(Microservices Architecture) 형태를 지향하여 각 역할을 완벽히 분리했습니다.

Plaintext
Red-Blue-Mid-Project_OAA/
├── .github/workflows/         # 일일 데이터 수집 CI/CD 배치
├── DB/                        # 데이터 수집 및 병합 로직
│   ├── build_master_dataset.py
│   └── utils/sp500_scraper.py
├── Classification/            # ML 모델링 및 아티팩트
│   ├── Preprocessing/         # 거시, 모멘텀 등 파생 변수 생성
│   ├── models/                # LogReg, RF, SVM, XGB 개별 모델
│   ├── ensemble/              # 모델 결합 최적화
│   └── artifacts/             # [산출물] 종목별 예측 결과 파일
├── investment-mbti-back/      # API 및 포트폴리오 연산 서버
│   ├── main.py
│   ├── portfolio_optimizer.py
│   └── risk_profile.py
├── investment-mbti/           # 프론트엔드 React/Vite 앱
│   ├── src/components/        # Dashboard, PieChart, Question 등
│   └── public/images/         # 캐릭터 및 에셋 이미지
└── scripts/                   # 전체 시스템 일괄 구동 쉘 스크립트
8. 🔀 브랜치 전략 및 협업 방식
모놀리식 환경에서의 충돌을 막기 위해 GitHub Flow 기반의 Feature 브랜치 전략을 사용했습니다.

코드 스니펫
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
main: 배포 가능한 안정적인 코드가 유지되는 브랜치

dev: 다음 출시 버전을 위해 개발 중인 코드가 모이는 브랜치

feat/OOO: 기능 개발 브랜치. 개발 완료 후 PR(Pull Request) 리뷰를 거쳐 dev로 병합

9. 🚀 설치 및 실행 방법
클라이언트 (Frontend) 실행
cd investment-mbti
npm install
npm run dev

서버 (Backend) 실행
cd investment-mbti-back
pip install -r requirements.txt
python main.py

머신러닝 파이프라인 (재학습 시)
cd scripts
bash run_pipeline_300.sh

---

### 💡 활용 가이드:
1. **GitHub 복사/붙여넣기:** 위 코드를 처음부터 끝까지 그대로 긁어서 `README.md`에 붙여넣으세요.
2. **내용 수정:** * **[팀원 구성]** 섹션의 이름과 깃허브 링크를 본인과 팀원에 맞게 수정하세요.
   * **[화면 구성]** 섹션에서 `https://via.placeholder.com/...` 이라고 되어 있는 부분의 주소를 **실제 프로젝트 캡쳐본 이미지 링크**(깃허브 이슈나 리드미 에디터에 이미지를 드래그 앤 드롭하면 나오는 URL)로 교체하시면 완벽해집니다.

# investment-mbti (Frontend)

Vite + React 기반 프론트엔드입니다.

## API URL 설정

프론트는 `VITE_API_URL` 환경 변수를 사용해 백엔드 주소를 결정합니다.

1. 로컬 개발 (`localhost`):
`VITE_API_URL`이 없으면 자동으로 `http://localhost:8000`을 사용합니다.
2. 배포 환경 (예: Vercel):
`VITE_API_URL`을 반드시 설정해야 합니다.

예시:

```env
VITE_API_URL=https://your-backend.example.com
```

## Vercel 설정

1. Vercel 프로젝트 `Settings > Environment Variables`로 이동
2. `VITE_API_URL` 추가
3. 값에 백엔드 공개 URL 입력
4. Redeploy 실행

## 실행

```bash
npm install
npm run dev
```

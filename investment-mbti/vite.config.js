/*
 * 이 파일은 프론트엔드 개발 서버와 빌드 동작을 정하는 Vite 설정 파일입니다.
 * 설정값과 고정 데이터가 한곳에 모여 있어 개발 서버, API 주소, 린트 규칙, 화면 공통 상수를 바꾸더라도 호출부 수정 범위를 최소화할 수 있습니다.
 */

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
})

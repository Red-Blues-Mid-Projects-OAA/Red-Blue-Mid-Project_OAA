const rawApiBase = import.meta.env.VITE_API_URL?.trim() || '';
const localApiPattern = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?(\/|$)/i;

const isLocalHost =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
const isRemoteHost = !isLocalHost;
const isLocalApiConfigured = localApiPattern.test(rawApiBase);

const fallbackApiBase = isLocalHost ? 'http://localhost:8000' : '';
const resolvedApiBase = isRemoteHost && isLocalApiConfigured ? '' : (rawApiBase || fallbackApiBase);

export const API_BASE = resolvedApiBase.replace(/\/+$/, '');
export const hasApiBase = API_BASE.length > 0;

export function getApiConfigErrorMessage() {
  if (isRemoteHost && isLocalApiConfigured) {
    return '배포 환경에서는 localhost 백엔드를 사용할 수 없습니다. VITE_API_URL을 공개 HTTPS 백엔드 주소로 설정해주세요.';
  }

  if (!hasApiBase) {
    return '서버 주소가 설정되지 않았습니다. Vercel 환경 변수 VITE_API_URL을 설정해주세요.';
  }

  return '';
}

export function buildApiUrl(path) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${normalizedPath}`;
}

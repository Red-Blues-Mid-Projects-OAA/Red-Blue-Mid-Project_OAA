const rawApiBase = import.meta.env.VITE_API_URL?.trim() || '';
const localApiPattern =
  /^https?:\/\/((localhost|127\.0\.0\.1|::1)|(10\.\d{1,3}\.\d{1,3}\.\d{1,3})|(192\.168\.\d{1,3}\.\d{1,3})|(172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})|([a-z0-9-]+\.local))(?:\:\d+)?(\/|$)/i;

function isLocalNetworkHost(hostname) {
  const target = String(hostname || '').trim().toLowerCase();
  if (!target) return false;

  if (target === 'localhost' || target === '127.0.0.1' || target === '::1') {
    return true;
  }

  if (/^[a-z0-9-]+\.local$/i.test(target)) {
    return true;
  }

  return /^(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})$/i.test(target);
}

const browserHostname =
  typeof window !== 'undefined' ? String(window.location.hostname || '').trim() : '';
const isLocalHost = isLocalNetworkHost(browserHostname);
const isRemoteHost = !isLocalHost;
const isLocalApiConfigured = localApiPattern.test(rawApiBase);

let fallbackApiBase = '';
if (isLocalHost) {
  if (browserHostname === 'localhost' || browserHostname === '127.0.0.1' || browserHostname === '::1') {
    fallbackApiBase = 'http://localhost:8000';
  } else {
    fallbackApiBase = `http://${browserHostname}:8000`;
  }
}

const resolvedApiBase = isRemoteHost && isLocalApiConfigured ? '' : rawApiBase || fallbackApiBase;

export const API_BASE = resolvedApiBase.replace(/\/+$/, '');
export const hasApiBase = API_BASE.length > 0;

export function getApiConfigErrorMessage() {
  if (isRemoteHost && isLocalApiConfigured) {
    return '배포 환경에서는 localhost/사설망 백엔드를 사용할 수 없습니다. VITE_API_URL을 공개 HTTPS 백엔드 주소로 설정해주세요.';
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

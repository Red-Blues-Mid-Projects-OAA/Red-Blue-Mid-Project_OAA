const rawApiBase = import.meta.env.VITE_API_URL?.trim();
const isLocalHost =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');

const fallbackApiBase = isLocalHost ? 'http://localhost:8000' : '';

export const API_BASE = (rawApiBase || fallbackApiBase).replace(/\/+$/, '');
export const hasApiBase = API_BASE.length > 0;

export function buildApiUrl(path) {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${normalizedPath}`;
}

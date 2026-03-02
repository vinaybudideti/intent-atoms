// In dev mode (Vite), proxy rewrites /api/* -> localhost:8000/*
// In production (served by FastAPI), API is on the same origin at root
const isDev = import.meta.env.DEV;
const BASE = isDev ? "/api" : "";

export async function queryEngine(queryText) {
  const res = await fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: queryText }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status}`);
  return res.json();
}

export async function fetchStats() {
  const res = await fetch(`${BASE}/stats`);
  if (!res.ok) throw new Error(`Stats failed: ${res.status}`);
  return res.json();
}

export async function fetchAtoms(skip = 0, limit = 50) {
  const res = await fetch(`${BASE}/atoms?skip=${skip}&limit=${limit}`);
  if (!res.ok) throw new Error(`Atoms failed: ${res.status}`);
  return res.json();
}

export async function clearCache() {
  const res = await fetch(`${BASE}/clear`, { method: "POST" });
  if (!res.ok) throw new Error(`Clear failed: ${res.status}`);
  return res.json();
}

export async function healthCheck() {
  try {
    const res = await fetch(`${BASE}/health`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.engine_ready === true;
  } catch {
    return false;
  }
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function fetchRecommendations(query, topK = 10) {
  const url = `${API_URL}/api/recommend?q=${encodeURIComponent(query)}&top_k=${topK}`
  const res = await fetch(url)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

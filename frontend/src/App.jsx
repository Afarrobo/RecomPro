import { useState, useRef } from 'react'
import { fetchRecommendations } from './api'
import './App.css'

const EXAMPLE_QUERIES = [
  'wireless earbuds with good bass and mic for calling',
  'budget keyboard with good build quality',
  'kom dame valo powerbank',
  'smartwatch with long battery backup',
]

const CATEGORIES_ROW_1 = [
  'PowerBank', 'CellPhone', 'HDMI Cable', 'Microphone', 'Charging Cable',
  'Charging Adapter', 'SmartPhone', 'WebCam', 'SmartWatch', 'Mouse',
]
const CATEGORIES_ROW_2 = [
  'Router', 'Pendrive', 'Memory Cards', 'Keyboard', 'Earbuds',
  'Earphones', 'Mini Printer', 'CC Camera', 'Speaker',
]

function Blobs() {
  return (
    <div className="blobs" aria-hidden="true">
      <span className="blob blob--coral" />
      <span className="blob blob--mint" />
      <span className="blob blob--gold" />
      <span className="tag tag--1">৳</span>
      <span className="tag tag--2">★</span>
      <span className="tag tag--3">✓</span>
      <span className="tag tag--2">★</span>
    </div>
  )
}

function CategoryMarquee({ onPick }) {
  return (
    <div className="marquee">
      <div className="marquee__row marquee__row--left">
        {[...CATEGORIES_ROW_1, ...CATEGORIES_ROW_1].map((c, i) => (
          <button key={`${c}-${i}`} type="button" className="marquee__pill" onClick={() => onPick(c)}>
            {c}
          </button>
        ))}
      </div>
      <div className="marquee__row marquee__row--right">
        {[...CATEGORIES_ROW_2, ...CATEGORIES_ROW_2].map((c, i) => (
          <button key={`${c}-${i}`} type="button" className="marquee__pill" onClick={() => onPick(c)}>
            {c}
          </button>
        ))}
      </div>
    </div>
  )
}

function ScoreBar({ score }) {
  const pct = Math.max(4, Math.min(100, Math.round(score * 100)))
  return (
    <div className="score-bar" title={`Match score: ${pct}%`}>
      <div className="score-bar__fill" style={{ width: `${pct}%` }} />
    </div>
  )
}

function ProductCard({ product }) {
  return (
    <li className="card">
      <div className="card__top">
        <span className="card__rank">#{product.rank}</span>
        <span className="card__category">{product.sheet_name}</span>
      </div>
      <h3 className="card__title">{product.product_name}</h3>
      {product.product_link ? <a className="card__link" href={product.product_link} target="_blank" rel="noopener noreferrer">View product →</a> : null}
      <div className="card__meta">
        {product.brand && <span>{product.brand}</span>}
        {product.price_bdt != null && <span>৳ {Math.round(product.price_bdt).toLocaleString()}</span>}
        {product.overall_rating != null && <span>★ {product.overall_rating.toFixed(1)}</span>}
        <span>{product.review_count} reviews</span>
      </div>
      {product.top_strengths && (
        <p className="card__row"><strong>Loved for</strong> {product.top_strengths}</p>
      )}
      {product.top_weaknesses && (
        <p className="card__row card__row--weak"><strong>Watch out for</strong> {product.top_weaknesses}</p>
      )}
      <ScoreBar score={product.score} />
    </li>
  )
}

export default function App() {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [data, setData] = useState(null)
  const inputRef = useRef(null)

  async function runSearch(q) {
    const trimmed = (q ?? query).trim()
    if (!trimmed) return
    setLoading(true)
    setError(null)
    try {
      const result = await fetchRecommendations(trimmed, 10)
      setData(result)
    } catch (err) {
      setError(err.message || 'Something went wrong')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    runSearch()
  }

  function handleExample(q) {
    setQuery(q)
    runSearch(q)
    inputRef.current?.focus()
  }

  return (
    <div className="page">
      <Blobs />

      <header className="hero">
        <span className="hero__eyebrow">Aspect-aware product recommendations</span>
        <h1 className="hero__title">RecomPro</h1>
        <p className="hero__subtitle">
          Tell it what you actually want — a category, a budget, a must-have feature —
          in English or Bangla. It reads real customer reviews to find the closest match.
        </p>

        <CategoryMarquee onPick={handleExample} />

        <form className="search" onSubmit={handleSubmit}>
          <input
            ref={inputRef}
            className="search__input"
            type="text"
            placeholder="e.g. budget earbuds with strong bass under 2000 taka"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button className="search__button" type="submit" disabled={loading}>
            {loading ? 'Searching…' : 'Find products'}
          </button>
        </form>

        <div className="examples">
          {EXAMPLE_QUERIES.map((q) => (
            <button key={q} className="examples__chip" onClick={() => handleExample(q)} type="button">
              {q}
            </button>
          ))}
        </div>
      </header>

      <main className="results">
        {error && <p className="results__error">{error}</p>}

        {data && !error && (
          <>
            <div className="results__meta">
              <span>Results for "{data.query}"</span>
              {data.matched_categories?.length > 0 && (
                <span className="results__pill">category: {data.matched_categories.join(', ')}</span>
              )}
              {data.matched_aspects?.length > 0 && (
                <span className="results__pill">focus: {data.matched_aspects.join(', ').replaceAll('_', ' ')}</span>
              )}
            </div>

            {data.results.length === 0 ? (
              <p className="results__empty">No products matched that query yet — try loosening the budget or category.</p>
            ) : (
              <ul className="card-grid">
                {data.results.map((p) => (
                  <ProductCard key={p.product_id} product={p} />
                ))}
              </ul>
            )}
          </>
        )}
      </main>
    </div>
  )
}
"""
Ports the scoring logic from product_review.ipynb into a service that:

  1. reads reviews from Postgres (loaded there from your Excel file)
  2. builds the same aspect-group product catalog the notebook builds
  3. encodes products/queries with a multilingual E5 model (dense) + TF-IDF (sparse)
  4. blends dense + sparse + aspect + category + price signals into a final ranking

"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text as sql_text
from transformers import AutoModel, AutoTokenizer

from .config import ARTIFACT_DIR, BASE_MODEL, DENSE_MODEL_DIR
from .database import engine


# Aspect / category vocab (from the notebook)


ASPECT_GROUPS = {
    'sound': {'raw': ['sound', 'sound_quality', 'audio'], 'aliases': ['sound', 'audio', 'clear sound', 'speaker sound', 'সাউন্ড', 'শব্দ']},
    'bass': {'raw': ['bass', 'bass_and_loudness'], 'aliases': ['bass', 'loud', 'বেস', 'বাস']},
    'microphone': {'raw': ['mic', 'microphone'], 'aliases': ['mic', 'microphone', 'call quality', 'voice call', 'মাইক', 'কল']},
    'noise_control': {'raw': ['noise', 'noise_cancellation', 'anc_enc'], 'aliases': ['noise', 'anc', 'enc', 'noise cancellation', 'নয়েজ']},
    'battery': {'raw': ['battery'], 'aliases': ['battery', 'battery backup', 'backup', 'long battery', 'ব্যাটারি', 'ব্যাকআপ']},
    'charging_speed': {'raw': ['charging', 'charger', 'wattage', 'speed'], 'aliases': ['charging', 'fast charging', 'quick charge', 'watt', 'speed', 'চার্জ', 'ফাস্ট চার্জ']},
    'heating': {'raw': ['heat', 'heating'], 'aliases': ['heat', 'heating', 'overheat', 'hot', 'গরম', 'হিট']},
    'connectivity': {'raw': ['connectivity', 'bluetooth', 'network', 'stability', 'range', 'ports', 'connector', 'compatibility', 'data', 'app', 'software', 'firmware'], 'aliases': ['connect', 'bluetooth', 'wifi', 'network', 'range', 'stable', 'compatibility', 'usb', 'type c', 'কানেক্ট', 'ওয়াইফাই']},
    'build_quality': {'raw': ['quality', 'build', 'build_design', 'build_quality', 'build_quality_durability', 'durability', 'durability_and_quality', 'design', 'strap', 'case_accessories', 'accessories'], 'aliases': ['quality', 'build', 'durable', 'strong', 'premium', 'design', 'টেকসই', 'কোয়ালিটি', 'বিল্ড']},
    'comfort': {'raw': ['comfort', 'comfort_design', 'comfort_fit'], 'aliases': ['comfort', 'comfortable', 'fit', 'আরাম', 'কমফোর্ট']},
    'price_value': {'raw': ['price', 'price_value'], 'aliases': ['price', 'budget', 'cheap', 'affordable', 'value for money', 'কম দাম', 'কম দামে', 'বাজেট', 'সাশ্রয়ী']},
    'delivery_service': {'raw': ['delivery', 'delivery_service', 'packaging'], 'aliases': ['delivery', 'packaging', 'box', 'ডেলিভারি', 'প্যাকেজিং']},
    'seller_authenticity': {'raw': ['seller', 'authenticity', 'warranty', 'reliability', 'corruption'], 'aliases': ['seller', 'original', 'authentic', 'genuine', 'warranty', 'reliable', 'সেলার', 'অরিজিনাল', 'ওয়ারেন্টি']},
    'display_video': {'raw': ['display', 'video', 'video_quality', 'view', 'resolution', 'clarity', 'lowlight', 'night_view', 'wideangle', 'photo', 'print', 'text'], 'aliases': ['display', 'screen', 'video', 'camera', 'resolution', 'clear view', 'night vision', 'print quality', 'ডিসপ্লে', 'ক্যামেরা']},
    'performance': {'raw': ['performance', 'features', 'setup', 'security', 'sensors', 'controls', 'keys', 'click_buttons', 'scroll_wheel', 'layout', 'lighting', 'rgb_lighting', 'indicator', 'motion'], 'aliases': ['performance', 'smooth', 'feature', 'gaming', 'setup', 'sensor', 'keys', 'lighting', 'পারফরম্যান্স', 'ফিচার']},
    'storage_capacity': {'raw': ['capacity', 'size', 'paper', 'cable', 'length'], 'aliases': ['capacity', 'storage', 'memory', 'size', 'length', 'ক্যাপাসিটি', 'স্টোরেজ', 'মেমোরি']},
}
RAW2GROUP = {raw: g for g, d in ASPECT_GROUPS.items() for raw in d['raw']}

CATEGORY_ALIASES = {
    'powerbank': ['powerbank', 'power bank', 'পাওয়ার ব্যাংক'],
    'cellphone': ['feature phone', 'button phone', 'cell phone', 'cellphone', 'বাটন ফোন'],
    'hdmi_cable': ['hdmi', 'hdmi cable'],
    'microphone': ['microphone', 'standalone mic', 'recording mic'],
    'chargingcable': ['charging cable', 'data cable', 'usb cable', 'type c cable', 'চার্জিং ক্যাবল'],
    'chargingadapter': ['adapter', 'charging adapter', 'wall charger', 'fast charger', 'চার্জার', 'অ্যাডাপ্টার'],
    'smartphone': ['smartphone', 'smart phone', 'android phone', 'স্মার্টফোন'],
    'webcam': ['webcam', 'web cam', 'ওয়েবক্যাম'],
    'smartwatch': ['smartwatch', 'smart watch', 'watch', 'ঘড়ি'],
    'mouse': ['mouse', 'মাউস'],
    'router': ['router', 'wifi router', 'wi-fi router', 'রাউটার'],
    'pendrive': ['pendrive', 'pen drive', 'usb drive', 'flash drive', 'পেনড্রাইভ'],
    'memorycards': ['memory card', 'sd card', 'micro sd', 'মেমোরি কার্ড'],
    'keyboard': ['keyboard', 'কিবোর্ড'],
    'earbuds': ['wireless earbuds', 'earbuds', 'earbud', 'tws', 'airpods', 'ইয়ারবাড'],
    'earphones': ['headphone', 'headphones', 'earphone', 'earphones', 'headset', 'neckband', 'হেডফোন', 'ইয়ারফোন'],
    'mini_printer': ['mini printer', 'thermal printer', 'printer', 'প্রিন্টার'],
    'cc_camera': ['cctv', 'cc tv', 'security camera', 'ip camera', 'সিসিটিভি'],
    'speaker': ['pc speaker', 'computer speaker', 'speaker', 'স্পিকার'],
}
GROUP_PHRASES = {
    'sound': 'good sound quality', 'bass': 'strong bass', 'microphone': 'good mic for calls',
    'noise_control': 'low noise', 'battery': 'good battery backup', 'charging_speed': 'fast charging',
    'heating': 'no heating problem', 'connectivity': 'stable connection', 'build_quality': 'good build quality',
    'comfort': 'comfortable fit', 'price_value': 'good value for money', 'delivery_service': 'good delivery service',
    'seller_authenticity': 'original and reliable seller', 'display_video': 'clear display and video',
    'performance': 'smooth performance', 'storage_capacity': 'good capacity',
}
SHEET_DISPLAY = {k: k.replace('_', ' ') for k in CATEGORY_ALIASES}
BANGLA_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

FEATURE_NAMES = ['dense', 'sparse', 'aspect', 'category', 'rating', 'positive_rate', 'price_penalty', 'negative_rate', 'confidence']
# Hand-set weights (dense/sparse/aspect/category matches push scores up, price_penalty/negative_rate pull them down).
# Swap these for manual_weights.joblib from your Kaggle run if you have it (see load_optional_artifacts).
DEFAULT_MANUAL_WEIGHTS = np.array([0.32, 0.14, 0.20, 0.14, 0.05, 0.05, -0.18, -0.12, 0.05], dtype='float32')


def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    x = unicodedata.normalize('NFKC', str(x)).replace('\u200c', ' ').replace('\xa0', ' ')
    x = re.sub(r'\s+', ' ', x).strip()
    return '' if x.lower() in {'nan', 'none', 'null'} else x


def token_count(text: str) -> int:
    return len(re.findall(r'[A-Za-z0-9\u0980-\u09FF]+', normalize_text(text)))


def category_text(sheet_name: str) -> str:
    return SHEET_DISPLAY.get(sheet_name.lower(), sheet_name.replace('_', ' '))


def contains_alias(q: str, alias: str) -> bool:
    alias = alias.lower()
    if re.search(r'[A-Za-z0-9]', alias):
        return re.search(r'(?<![a-z0-9])' + re.escape(alias) + r'(?![a-z0-9])', q) is not None
    return alias in q


def extract_query_aspects(query: str):
    q = normalize_text(query).lower()
    return sorted({g for g, d in ASPECT_GROUPS.items() if any(contains_alias(q, a) for a in d['aliases'])})


def detect_query_categories(query: str):
    q = normalize_text(query).lower()
    return sorted({cat for cat, aliases in CATEGORY_ALIASES.items() if any(contains_alias(q, a) for a in aliases)})


def parse_price_query(query: str):
    q = normalize_text(query).lower().translate(BANGLA_DIGITS)
    nums = [int(x) for x in re.findall(r'\d{2,7}', q)]
    has_budget = any(k in q for k in ['budget', 'cheap', 'affordable', 'low price', 'kom dam', 'kom dame', 'কম দাম', 'কম দামে', 'বাজেট', 'সাশ্রয়ী'])
    info = {'max_price': None, 'min_price': None, 'budget_query': has_budget}
    if nums and any(k in q for k in ['under', 'below', 'within', 'max', 'less than', 'টাকার মধ্যে', 'এর মধ্যে', '৳', 'tk', 'taka', 'bdt']):
        info['max_price'] = nums[0]
    if nums and any(k in q for k in ['above', 'over', 'minimum', 'at least', 'more than']):
        info['min_price'] = nums[0]
    return info


# Words that mean "sort cheapest first" / "sort priciest first" — distinct from parse_price_query's
# budget filter above, this is about *ordering* the results, not filtering them.
CHEAP_SORT_WORDS = ['cheapest', 'lowest price', 'lowest to highest', 'low to high', 'sort by price low',
                     'kom dam', 'kom dame', 'সবচেয়ে কম দাম', 'কম দাম আগে']
EXPENSIVE_SORT_WORDS = ['expensive', 'highest price', 'high to low', 'highest to lowest', 'sort by price high',
                         'premium', 'best quality', 'high end', 'beshi dam', 'বেশি দাম', 'সবচেয়ে দামি', 'সেরা']


def detect_sort_intent(query: str):
    q = normalize_text(query).lower()
    if any(k in q for k in CHEAP_SORT_WORDS):
        return 'price_asc'
    if any(k in q for k in EXPENSIVE_SORT_WORDS):
        return 'price_desc'
    return None


def row_minmax(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return ((x - mn) / (mx - mn + 1e-8)).astype('float32')


# ---------------------------------------------------------------------------
# Catalog building (mirrors build_product_catalog in the notebook)
# ---------------------------------------------------------------------------

def load_reviews_from_db() -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql(sql_text("SELECT * FROM reviews"), conn)
    if df.empty:
        raise RuntimeError(
            "No rows in the `reviews` table yet. Run scripts/load_excel_to_postgres.py first."
        )
    df['review_text'] = df['review_text'].fillna('').map(normalize_text)
    df['product_name'] = df['product_name'].fillna('').map(normalize_text)
    df['aspects'] = df['aspects'].apply(lambda a: a if isinstance(a, dict) else {})
    return df


def build_product_catalog(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby('product_id'):
        meta = g.iloc[0]
        group_counts = defaultdict(lambda: {'pos': 0, 'neg': 0, 'mention': 0})
        for _, r in g.iterrows():
            for raw_col, val in (r['aspects'] or {}).items():
                try:
                    v = int(val)
                except (TypeError, ValueError):
                    continue
                if v not in (-1, 0, 1) or v == 0:
                    continue
                group = RAW2GROUP.get(raw_col, raw_col)
                group_counts[group]['pos' if v == 1 else 'neg'] += 1
                group_counts[group]['mention'] += 1

        group_stats = {}
        for group, c in group_counts.items():
            m = max(c['mention'], 0)
            pos, neg = c['pos'], c['neg']
            quality = (pos + 1.0) / (pos + neg + 2.0)
            confidence = min(1.0, math.log1p(m) / math.log1p(10))
            group_stats[group] = {
                **c,
                'quality': float(quality),
                'neg_rate': float(neg / m) if m else 0.0,
                'confidence': float(confidence),
                'aspect_score': float(quality * confidence),
            }

        strengths = [g_ for g_, s in sorted(group_stats.items(), key=lambda kv: (kv[1]['aspect_score'], kv[1]['mention']), reverse=True)
                     if s['mention'] >= 2 and s['quality'] >= 0.65 and s['neg_rate'] <= 0.45][:8]
        weaknesses = [g_ for g_, s in sorted(group_stats.items(), key=lambda kv: (kv[1]['neg_rate'], kv[1]['confidence'], kv[1]['mention']), reverse=True)
                      if s['mention'] >= 2 and s['neg_rate'] >= 0.40][:6]

        pos_snips = g[(g['sentiment'] == 1)]['review_text'].drop_duplicates().head(4).tolist()
        rating = pd.to_numeric(g['overall_rating'], errors='coerce').dropna()
        price = pd.to_numeric(g['price_bdt'], errors='coerce').dropna()
        review_confidence = min(1.0, math.log1p(len(g)) / math.log1p(50))
        strength_text = ', '.join(GROUP_PHRASES.get(x, x.replace('_', ' ')) for x in strengths)

        parts = [
            f"product name: {meta['product_name']}",
            f"category: {meta['category']}",
            f"brand: {meta['brand']}" if meta['brand'] else '',
            f"price bdt: {float(price.median()):.2f}" if len(price) else '',
            f"average rating: {float(rating.mean()):.2f}" if len(rating) else '',
            f"review count: {len(g)}",
            f"strengths: {strength_text}" if strengths else '',
            'positive reviews: ' + ' || '.join(pos_snips) if pos_snips else '',
        ]

        rows.append({
            'product_id': pid,
            'sheet_name': meta['category'],
            'sheet_key': str(meta['category']).lower(),
            'product_name': meta['product_name'],
            'brand': meta['brand'],
            'product_link': meta.get('product_link'), 
            'price_bdt': float(price.median()) if len(price) else np.nan,
            'overall_rating': float(rating.mean()) if len(rating) else np.nan,
            'review_count': int(len(g)),
            'review_confidence': float(review_confidence),
            'positive_review_rate': float((g['sentiment'] == 1).mean()),
            'negative_review_rate': float((g['sentiment'] == -1).mean()),
            'aspect_group_stats': group_stats,
            'top_strengths': ', '.join(x.replace('_', ' ') for x in strengths),
            'top_weaknesses': ', '.join(x.replace('_', ' ') for x in weaknesses),
            'profile_text': '\n'.join(p for p in parts if p),
        })
    return pd.DataFrame(rows).sort_values(['sheet_name', 'review_count', 'overall_rating'], ascending=[True, False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dense encoder
# ---------------------------------------------------------------------------

class E5Encoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=1)


class RecommenderService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_source = DENSE_MODEL_DIR or BASE_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = E5Encoder(model_source).to(self.device).eval()

        self.catalog: pd.DataFrame | None = None
        self.product_embeddings: np.ndarray | None = None
        self.sparse_vectorizer: TfidfVectorizer | None = None
        self.product_sparse_matrix = None
        self.product_prices = None
        self.price_norm = None
        self.rating_feature = None
        self.positive_rate_feature = None
        self.review_confidence_feature = None
        self.weights = DEFAULT_MANUAL_WEIGHTS

    @torch.no_grad()
    def _encode(self, texts, is_query: bool, batch_size: int = 64) -> np.ndarray:
        prefix = 'query: ' if is_query else 'passage: '
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = [prefix + normalize_text(t) for t in texts[i:i + batch_size]]
            tok = self.tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
            tok = {k: v.to(self.device) for k, v in tok.items()}
            outs.append(self.model(**tok).cpu().numpy())
        return np.vstack(outs).astype('float32') if outs else np.zeros((0, 768), dtype='float32')

    def build(self, cache: bool = True):
        cache_pkl = ARTIFACT_DIR / 'product_catalog.pkl'
        cache_npy = ARTIFACT_DIR / 'product_embeddings.npy'
        cache_tfidf = ARTIFACT_DIR / 'sparse_vectorizer.joblib'

        df = load_reviews_from_db()
        self.catalog = build_product_catalog(df)
        if self.catalog.empty:
            raise RuntimeError("Catalog is empty after grouping — check the `reviews` table has data.")

        if cache and cache_npy.exists() and cache_pkl.exists():
            cached = pd.read_pickle(cache_pkl)
            if len(cached) == len(self.catalog) and (cached['product_id'] == self.catalog['product_id']).all():
                self.product_embeddings = np.load(cache_npy)
                self.sparse_vectorizer = joblib.load(cache_tfidf)
            else:
                self._embed_and_cache()
        else:
            self._embed_and_cache()

        self.product_sparse_matrix = self.sparse_vectorizer.transform(self.catalog['profile_text'].tolist())
        self.product_prices = self.catalog['price_bdt'].fillna(self.catalog['price_bdt'].median()).to_numpy()
        self.price_norm = (self.product_prices - self.product_prices.min()) / (np.ptp(self.product_prices) + 1e-8)
        self.rating_feature = self.catalog['overall_rating'].fillna(self.catalog['overall_rating'].mean()).to_numpy() / 5.0
        self.positive_rate_feature = self.catalog['positive_review_rate'].fillna(0).to_numpy()
        self.review_confidence_feature = self.catalog['review_confidence'].fillna(0).to_numpy()

    def _embed_and_cache(self):
        self.product_embeddings = self._encode(self.catalog['profile_text'].tolist(), is_query=False)
        self.sparse_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=1, max_features=50000)
        self.sparse_vectorizer.fit(self.catalog['profile_text'].tolist())
        self.catalog.to_pickle(ARTIFACT_DIR / 'product_catalog.pkl')
        np.save(ARTIFACT_DIR / 'product_embeddings.npy', self.product_embeddings)
        joblib.dump(self.sparse_vectorizer, ARTIFACT_DIR / 'sparse_vectorizer.joblib')

    def _aspect_and_category_features(self, query: str):
        n = len(self.catalog)
        groups = extract_query_aspects(query)
        aspect = np.zeros(n, dtype='float32')
        neg = np.zeros(n, dtype='float32')
        if groups:
            for j, stats in enumerate(self.catalog['aspect_group_stats']):
                vals, negs = [], []
                for g in groups:
                    s = stats.get(g)
                    if s and s['mention'] > 0:
                        vals.append(s['quality'] * s['confidence'])
                        negs.append(s['neg_rate'] * s['confidence'])
                aspect[j] = float(np.mean(vals)) if vals else 0.0
                neg[j] = float(np.mean(negs)) if negs else 0.0

        cats = detect_query_categories(query)
        cat = np.zeros(n, dtype='float32')
        if cats:
            cat = self.catalog['sheet_key'].isin(cats).to_numpy().astype('float32')

        info = parse_price_query(query)
        price_pen = np.zeros(n, dtype='float32')
        if info['max_price']:
            m = max(info['max_price'], 1)
            price_pen += np.where(self.product_prices > m, np.minimum(1.0, (self.product_prices - m) / m), 0)
        if info['min_price']:
            m = max(info['min_price'], 1)
            price_pen += np.where(self.product_prices < m, np.minimum(1.0, (m - self.product_prices) / m), 0)
        if info['budget_query'] and info['max_price'] is None:
            price_pen += self.price_norm

        return aspect, neg, cat, price_pen, groups, cats, info

    def recommend(self, query: str, top_k: int = 10):
        if self.catalog is None:
            raise RuntimeError("RecommenderService.build() must be called before recommend().")
        q = normalize_text(query)
        n = len(self.catalog)

        q_dense = self._encode([q], is_query=True)
        raw_dense_sims = (q_dense @ self.product_embeddings.T)[0]
        raw_sparse_sims = cosine_similarity(self.sparse_vectorizer.transform([q]), self.product_sparse_matrix).astype('float32')[0]
        max_dense_score = float(raw_dense_sims.max())
        max_sparse_score = float(raw_sparse_sims.max())

        dense = row_minmax(raw_dense_sims.reshape(1, -1))[0]
        sparse = row_minmax(raw_sparse_sims.reshape(1, -1))[0]
        aspect, neg, cat, price_pen, groups, cats, price_info = self._aspect_and_category_features(q)

        # Out-of-domain guard (ports the notebook's OOD check): reject queries whose best raw
        # similarity to anything in the catalog is weak, using the *unscaled* similarity — row_minmax
        # always stretches to 0-1 relative to this one query, so it can't be used to detect "no match".
        is_ood = False
        if not groups and not cats:
            if max_dense_score < 0.75 and max_sparse_score < 0.08:
                is_ood = True
        elif max_dense_score < 0.60 and max_sparse_score < 0.02:
            is_ood = True

        X = np.stack([dense, sparse, aspect, cat, self.rating_feature, self.positive_rate_feature,
                       price_pen, neg, self.review_confidence_feature], axis=1).astype('float32')
        manual = (X @ self.weights).astype('float32')
        scores = row_minmax(manual.reshape(1, -1))[0]

        has_aspect = bool(groups)
        scores = (0.65 * scores + 0.35 * aspect) if has_aspect else (0.80 * scores + 0.20 * cat if cats else scores)

        mask = np.ones(n, dtype=bool)
        if cats:
            cat_mask = self.catalog['sheet_key'].isin(cats).to_numpy()
            if cat_mask.any():
                mask &= cat_mask
        if price_info['max_price'] is not None:
            pm = self.product_prices <= price_info['max_price']
            if (mask & pm).any():
                mask &= pm
        if price_info['min_price'] is not None:
            pm = self.product_prices >= price_info['min_price']
            if (mask & pm).any():
                mask &= pm

        idx = np.where(mask)[0]
        sort_order = detect_sort_intent(q)

        if sort_order == 'price_asc':
            idx = idx[np.argsort(self.product_prices[idx])]
        elif sort_order == 'price_desc':
            idx = idx[np.argsort(-self.product_prices[idx])]
        else:
            idx = idx[np.argsort(-scores[idx])]

        # Out-of-domain guard wins regardless of category/aspect/price/sort matches — a query
        # that doesn't meaningfully resemble anything in the catalog returns no results.
        if is_ood:
            ranked = np.array([], dtype=int)
        else:
            ranked = idx[:top_k]

        out = self.catalog.iloc[ranked][[
            'product_id', 'sheet_name', 'product_name', 'brand',  'product_link' ,'price_bdt', 'overall_rating',
            'review_count', 'positive_review_rate', 'negative_review_rate', 'top_strengths', 'top_weaknesses',
        ]].copy()
        out.insert(0, 'rank', range(1, len(out) + 1))
        out['score'] = scores[ranked]
        results = out.replace({np.nan: None}).to_dict(orient='records')
        return {
            'query': query,
            'matched_categories': cats,
            'matched_aspects': groups,
            'results': results,
        }


_service: RecommenderService | None = None


def get_service() -> RecommenderService:
    global _service
    if _service is None:
        _service = RecommenderService()
        _service.build()
    return _service
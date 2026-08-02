"""
Loads every sheet of products.xlsx into the Postgres `reviews` table.

Usage (from backend/ folder, with your virtualenv active and .env configured):
    python -m scripts.load_excel_to_postgres

What it does, per sheet (e.g. "PowerBank", "Keyboard", ...):
  - reads all rows
  - pulls out the known meta columns (Marketplace, Category, Product_Link, Product_Name,
    Brand, Price_BDT, Overall_Rating, Reviews, Sentiment)
  - every other column (Quality, Battery, Mic, Heat, ...) is treated as an aspect column
    and stored as JSON: {"quality": 1, "battery": -1, ...}
  - builds product_id = "<sheet>::<product name>" (lowercased), same scheme as the notebook

If you see `relation "reviews" does not exist` when the API runs, it almost always means
this script was never run successfully against the SAME database the API connects to
(check DATABASE_URL in backend/.env vs whatever you're browsing in pgAdmin). This version
prints exactly which database/schema it's writing to, so that's easy to catch.
"""
import math
import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text as sql_text
from sqlalchemy.exc import ProgrammingError

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.config import EXCEL_PATH, DATABASE_URL  # noqa: E402
from app.database import Base, engine, SessionLocal  # noqa: E402
from app.db_models import Review  # noqa: E402

BANGLA_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')

META_COLUMN_MAP = {
    'marketplace': 'marketplace',
    'category': 'category',
    'product_link': 'product_link',
    'product_name': 'product_name',
    'brand': 'brand',
    'price_bdt': 'price_bdt',
    'overall_rating': 'overall_rating',
    'reviews': 'review_text',
    'sentiment': 'sentiment',
}


def normalize_col(x: str) -> str:
    x = str(x).strip().lower().replace('&', ' and ')
    x = re.sub(r'[()\[\]{}]', '', x)
    x = re.sub(r'[/\-]+', '_', x)
    x = re.sub(r'[^a-z0-9_]+', '_', x)
    return re.sub(r'_+', '_', x).strip('_')


def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    x = unicodedata.normalize('NFKC', str(x)).replace('\u200c', ' ').replace('\xa0', ' ')
    x = re.sub(r'\s+', ' ', x).strip()
    return '' if x.lower() in {'nan', 'none', 'null'} else x


def parse_price(x):
    s = normalize_text(x).translate(BANGLA_DIGITS)
    if not s:
        return None
    s = re.sub(r'[^0-9,.-]', '', s).replace(',', '')
    try:
        return float(s) if s else None
    except ValueError:
        return None


def parse_rating(x):
    s = normalize_text(x).translate(BANGLA_DIGITS).replace(',', '.')
    s = re.sub(r'[^0-9.-]', '', s)
    try:
        v = float(s) if s else None
        return v if v is not None and 0 <= v <= 5 else None
    except ValueError:
        return None


def parse_sentiment(x):
    try:
        v = float(str(x).strip().replace(',', '.'))
        return int(v) if v in (-1, 0, 1) else 0
    except (ValueError, TypeError):
        return 0


def _describe_target_db() -> None:
    """Print which host/database this run is actually writing to (password masked)."""
    parsed = urlparse(DATABASE_URL)
    host = parsed.hostname or "?"
    port = parsed.port or "?"
    dbname = (parsed.path or "").lstrip("/") or "?"
    user = parsed.username or "?"
    print(f"Target DB -> host={host} port={port} db={dbname} user={user}")
    print("If this doesn't match the database/server you're browsing in pgAdmin, "
          "fix DATABASE_URL in backend/.env before continuing.")


def _ensure_reviews_table() -> None:
    """Create the `reviews` table (and any other missing tables) if they don't exist yet,
    and verify it actually exists afterwards instead of assuming create_all worked."""
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    if not inspector.has_table("reviews"):
        raise RuntimeError(
            "Tried to create the `reviews` table but it still doesn't exist after "
            "create_all(). This usually means the DB user lacks CREATE privileges, "
            "or DATABASE_URL points at a database/schema you don't expect — check "
            "the 'Target DB' line printed above against pgAdmin."
        )
    print("Confirmed `reviews` table exists.")


def main():
    excel_path = Path(EXCEL_PATH)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found at {excel_path} — set EXCEL_PATH in backend/.env")

    _describe_target_db()

    print(f"Reading {excel_path} ...")
    xls = pd.ExcelFile(excel_path)

    _ensure_reviews_table()

    session = SessionLocal()
    try:
        session.execute(sql_text("TRUNCATE TABLE reviews RESTART IDENTITY"))
        session.commit()
    except ProgrammingError as exc:
        session.rollback()
        raise RuntimeError(
            "TRUNCATE failed even though the table was just confirmed to exist. "
            "Double-check you're not pointing at a different schema (e.g. not `public`)."
        ) from exc

    total = 0
    for sheet in xls.sheet_names:
        df = xls.parse(sheet_name=sheet, dtype=object)
        df.columns = [normalize_col(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.dropna(axis=1, how='all')

        aspect_cols = [c for c in df.columns if c not in META_COLUMN_MAP and c]
        rows = []
        for _, r in df.iterrows():
            review_text = normalize_text(r.get('reviews'))
            product_name = normalize_text(r.get('product_name'))
            if not review_text or not product_name:
                continue

            aspects = {}
            for c in aspect_cols:
                val = r.get(c)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                try:
                    v = int(float(val))
                except (ValueError, TypeError):
                    continue
                if v in (-1, 0, 1):
                    aspects[c] = v

            rows.append(Review(
                category=sheet,
                marketplace=normalize_text(r.get('marketplace')) or 'Daraz',
                product_link=normalize_text(r.get('product_link')),
                product_name=product_name,
                brand=normalize_text(r.get('brand')),
                price_bdt=parse_price(r.get('price_bdt')),
                overall_rating=parse_rating(r.get('overall_rating')),
                review_text=review_text,
                sentiment=parse_sentiment(r.get('sentiment')),
                aspects=aspects,
                product_id=f"{sheet.lower()}::{product_name.lower()}",
            ))

        if rows:
            session.bulk_save_objects(rows)
            session.commit()
            total += len(rows)
            print(f"  {sheet}: inserted {len(rows)} review rows")

    print(f"Done. Inserted {total} rows across {len(xls.sheet_names)} sheets into `reviews`.")
    session.close()


if __name__ == "__main__":
    main()
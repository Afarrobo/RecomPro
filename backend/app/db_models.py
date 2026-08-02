from sqlalchemy import Column, Integer, String, Float, Text, SmallInteger
from sqlalchemy.dialects.postgresql import JSONB

from .database import Base


class Review(Base):
    """
    One row per review, one table for the whole workbook.
    Each Excel sheet (PowerBank, Keyboard, Earbuds, ...) becomes a value in `category`.
    Aspect columns differ sheet to sheet (Quality, Capacity, Battery, Mic, ...), so instead of a
    fixed column per aspect we store them as a JSON dict: {"quality": 1, "battery": -1, ...}.
    This mirrors the aspect_cols handling in your notebook's load_reviews().
    """

    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String, index=True, nullable=False)       # sheet name, e.g. "PowerBank"
    marketplace = Column(String)
    product_link = Column(Text)
    product_name = Column(Text, index=True, nullable=False)
    brand = Column(String)
    price_bdt = Column(Float)
    overall_rating = Column(Float)
    review_text = Column(Text)
    sentiment = Column(SmallInteger)          # -1, 0, 1
    aspects = Column(JSONB, nullable=False, default=dict)  # {"quality": 1, "battery": -1, ...}
    product_id = Column(String, index=True, nullable=False)  # category::product_name (lowercased)

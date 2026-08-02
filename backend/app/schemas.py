from typing import Optional
from pydantic import BaseModel


class ProductResult(BaseModel):
    rank: int
    product_id: str
    sheet_name: str
    product_name: str
    brand: Optional[str] = None
    product_link: Optional[str] = None 
    price_bdt: Optional[float] = None
    overall_rating: Optional[float] = None
    review_count: int
    positive_review_rate: Optional[float] = None
    negative_review_rate: Optional[float] = None
    top_strengths: Optional[str] = None
    top_weaknesses: Optional[str] = None
    score: float


class RecommendResponse(BaseModel):
    query: str
    matched_categories: list[str]
    matched_aspects: list[str]
    results: list[ProductResult]

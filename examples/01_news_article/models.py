from dataclasses import dataclass


@dataclass
class NewsArticle:
    """Represents a finalized news article."""

    title: str
    summary: str
    sources: list[str]


@dataclass
class ArticleRequest:
    """Represents a clarified request for a news article."""

    topic: str
    angle: str
    audience: str
    requirements: list[str]

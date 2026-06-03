from textwrap import dedent

from .models import ArticleRequest, NewsArticle


def render_article_request(article_request: ArticleRequest) -> str:
    return (
        dedent(
            """
            Topic: {topic}
            Angle: {angle}
            Audience: {audience}
            Requirements:
            {requirements}
            """
        )
        .strip()
        .format(
            topic=article_request.topic,
            angle=article_request.angle,
            audience=article_request.audience,
            requirements="\n".join(
                f"- {requirement}" for requirement in article_request.requirements
            )
            or "- (none)",
        )
    )


def render_news_article(article: NewsArticle) -> str:
    return (
        dedent(
            """
            ## Title
            {title}

            ## Summary
            {summary}

            ## Sources
            {sources}
            """
        )
        .strip()
        .format(
            title=article.title,
            summary=article.summary,
            sources="\n".join(f"- {source}" for source in article.sources)
            or "- (none)",
        )
    )

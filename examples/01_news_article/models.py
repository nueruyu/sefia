from dataclasses import dataclass
from textwrap import dedent


@dataclass
class NewsArticle:
    """Represents a finalized news article."""

    title: str
    summary: str
    sources: list[str]

    def to_markdown(self):
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
                title=self.title,
                summary=self.summary,
                sources="\n".join(f"- {source}" for source in self.sources)
                or "- (none)",
            )
        )


@dataclass
class ArticleRequest:
    """Represents a clarified request for a news article."""

    topic: str
    angle: str
    audience: str
    requirements: list[str]

    def to_markdown(self) -> str:
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
                topic=self.topic,
                angle=self.angle,
                audience=self.audience,
                requirements="\n".join(
                    f"- {requirement}" for requirement in self.requirements
                )
                or "- (none)",
            )
        )

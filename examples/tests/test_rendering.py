from importlib import import_module

# Example packages are prefixed with digits (e.g. ``01_news_article``), which a
# plain ``import`` statement cannot reference, so load them via importlib.
news_models = import_module("01_news_article.models")
news_rendering = import_module("01_news_article.rendering")
code_models = import_module("02_code_quality.models")
code_rendering = import_module("02_code_quality.rendering")


class TestRenderArticleRequest:
    def test_includes_all_fields(self):
        request = news_models.ArticleRequest(
            topic="Generative AI",
            angle="Impact on developers",
            audience="Engineering managers",
            requirements=["Cite sources", "Keep it concise"],
        )

        rendered = news_rendering.render_article_request(request)

        assert "Topic: Generative AI" in rendered
        assert "Angle: Impact on developers" in rendered
        assert "Audience: Engineering managers" in rendered
        assert "- Cite sources" in rendered
        assert "- Keep it concise" in rendered

    def test_empty_requirements_render_placeholder(self):
        request = news_models.ArticleRequest(
            topic="t", angle="a", audience="aud", requirements=[]
        )

        rendered = news_rendering.render_article_request(request)

        assert "- (none)" in rendered


class TestRenderNewsArticle:
    def test_includes_title_summary_and_sources(self):
        article = news_models.NewsArticle(
            title="Big News",
            summary="Something happened.",
            sources=["https://example.com/a", "https://example.com/b"],
        )

        rendered = news_rendering.render_news_article(article)

        assert "## Title" in rendered
        assert "Big News" in rendered
        assert "## Summary" in rendered
        assert "Something happened." in rendered
        assert "- https://example.com/a" in rendered
        assert "- https://example.com/b" in rendered

    def test_empty_sources_render_placeholder(self):
        article = news_models.NewsArticle(title="t", summary="s", sources=[])

        rendered = news_rendering.render_news_article(article)

        assert "- (none)" in rendered


class TestRenderQualityReport:
    def _issue(self, perspective: str):
        return code_models.CodeIssue(
            file_path="src/app.py",
            line_number=42,
            perspective=perspective,
            description="Magic number used.",
            suggestion="Extract a named constant.",
        )

    def test_renders_summary_and_issue_details(self):
        report = code_models.QualityReport(
            overall_summary="Overall solid.",
            issues_by_perspective={"Coding Style": [self._issue("Coding Style")]},
        )

        rendered = code_rendering.render_quality_report(report)

        assert "# Code Quality Report" in rendered
        assert "**Summary:** Overall solid." in rendered
        assert "## Coding Style" in rendered
        assert "`src/app.py` (Line: 42)" in rendered
        assert "Magic number used." in rendered
        assert "Extract a named constant." in rendered

    def test_perspectives_without_issues_are_skipped(self):
        report = code_models.QualityReport(
            overall_summary="Nothing found.",
            issues_by_perspective={"Coding Style": [], "Design": []},
        )

        rendered = code_rendering.render_quality_report(report)

        assert "## Coding Style" not in rendered
        assert "## Design" not in rendered

    def test_report_with_no_issues(self):
        report = code_models.QualityReport(overall_summary="Clean.")

        rendered = code_rendering.render_quality_report(report)

        assert "# Code Quality Report" in rendered
        assert "**Summary:** Clean." in rendered

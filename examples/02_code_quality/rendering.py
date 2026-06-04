from textwrap import dedent

from .models import QualityReport


def render_quality_report(report: QualityReport) -> str:
    """Render the quality report as Markdown."""
    output = [f"# Code Quality Report\n\n**Summary:** {report.overall_summary}\n"]

    for perspective, issues in report.issues_by_perspective.items():
        if not issues:
            continue
        output.append(f"\n## {perspective}\n")
        for issue in issues:
            output.append(
                dedent(
                    f"""
                    - **File:** `{issue.file_path}` (Line: {issue.line_number})
                      - **Issue:** {issue.description}
                      - **Suggestion:** {issue.suggestion}
                    """
                ).strip()
            )

    return "\n".join(output)

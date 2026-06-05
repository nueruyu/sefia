from .models import QualityReport


def render_quality_report(report: QualityReport) -> str:
    """Render the quality report as Markdown."""
    output = [
        "# Code Quality Report",
        f"**Summary:** {report.overall_summary}",
    ]

    for perspective, issues in report.issues_by_perspective.items():
        if not issues:
            continue
        output.append(f"## {perspective}")
        for issue in issues:
            output.append(
                "\n".join(
                    [
                        f"### `{issue.file_path}` (Line: {issue.line_number})",
                        "",
                        "**Issue**",
                        "",
                        issue.description,
                        "",
                        "**Suggestion**",
                        "",
                        issue.suggestion,
                    ]
                )
            )

    return "\n\n".join(output)

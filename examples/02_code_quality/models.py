from dataclasses import dataclass, field
from enum import Enum

from sefios import state


@dataclass
class ProjectScope:
    """Defines the scope of the code review."""

    project_path: str
    focus_areas: list[str] = field(default_factory=list)
    excluded_files: list[str] = field(default_factory=list)


@state(key="code_quality.project_understanding")
@dataclass
class ProjectUnderstanding:
    """Represents the evolving understanding of the project."""

    summary: str = "Not yet analyzed."
    tech_stack: list[str] = field(default_factory=list)
    key_components: dict[str, str] = field(default_factory=dict)
    read_files: list[str] = field(default_factory=list)

    def copy(self) -> "ProjectUnderstanding":
        return ProjectUnderstanding(
            summary=self.summary,
            tech_stack=list(self.tech_stack),
            key_components=dict(self.key_components),
            read_files=list(self.read_files),
        )


class ReviewPerspective(Enum):
    """Defines the different perspectives for code review."""

    CODING_STYLE = "Coding Style and Conventions"
    DESIGN_PRINCIPLES = "Software Design Principles (SOLID, DRY, etc.)"
    MAINTAINABILITY = "Code Maintainability and Readability"
    DEPENDENCIES = "External Library Dependencies and Usage"


@dataclass
class CodeIssue:
    """Represents a single issue found in the code."""

    file_path: str
    line_number: int
    perspective: str
    description: str
    suggestion: str


@dataclass
class QualityReport:
    """Represents the final code quality report."""

    overall_summary: str
    issues_by_perspective: dict[str, list[CodeIssue]] = field(default_factory=dict)

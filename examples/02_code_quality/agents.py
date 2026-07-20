from pathlib import Path
from typing import Annotated

from sefia import AsRawText, Tools
from sefios import infer
from sefios.tools import Input

from .models import (
    CodeIssue,
    ProjectScope,
    ProjectUnderstanding,
    QualityReport,
)
from .tools import Files

RawCode = Annotated[str, AsRawText]


class ScopingAgent:
    _input: Tools[Input]

    def __init__(self, input_tool: Input):
        self._input = input_tool

    @infer
    async def define_scope(self) -> ProjectScope:
        """
        Defines a concrete code-review scope from the user's review request.

        The resulting ProjectScope contains the target Git repository path,
        review focus areas, and excluded files or patterns. Explicit paths
        provided by the user are preserved exactly. Critical missing details may
        be resolved through a focused Input tool question.
        """
        ...


class UnderstandingAgent:
    _file_tool: Tools[Files]

    def __init__(self, file_tool: Files):
        self._file_tool = file_tool

    @infer
    async def _prioritize_files_to_read(
        self,
        current_understanding: ProjectUnderstanding,
        unread_file_paths: list[str],
    ) -> list[str]:
        """
        Selects up to three unread files that are most useful for improving the
        current project understanding.

        Useful files are typically README files, package and build metadata,
        public API entry points, central orchestration code, core domain models,
        and representative tests. The returned paths are members of
        unread_file_paths. An empty list represents that the current
        understanding is already sufficient for review-file selection.
        """
        ...

    @infer
    async def _update_understanding(
        self,
        current_understanding: ProjectUnderstanding,
        new_file_contents: dict[str, str],
    ) -> ProjectUnderstanding:
        """
        Updates the project understanding from newly read file contents.

        The returned ProjectUnderstanding keeps a concise summary, detected
        technology stack, and key components. It preserves useful existing
        facts and incorporates only facts supported by the supplied files.
        """
        ...

    async def deepen_understanding(
        self,
        current_understanding: ProjectUnderstanding,
        all_file_paths: list[str],
        project_root: str,
    ) -> ProjectUnderstanding:
        read_files_set = set(current_understanding.read_files)
        unread_files = [path for path in all_file_paths if path not in read_files_set]
        if not unread_files:
            return current_understanding

        files_to_read = await self._prioritize_files_to_read(
            current_understanding, unread_files
        )
        unread_file_set = set(unread_files)
        files_to_read = list(
            dict.fromkeys(path for path in files_to_read if path in unread_file_set)
        )[:3]
        if not files_to_read:
            return current_understanding

        full_paths = [str(Path(project_root) / path) for path in files_to_read]
        contents = await self._file_tool.read_files(full_paths)
        updated_understanding = await self._update_understanding(
            current_understanding, contents
        )
        updated_understanding.read_files = sorted(
            set(current_understanding.read_files + files_to_read)
        )
        return updated_understanding


class ReviewScopingAgent:
    _input: Tools[Input]

    def __init__(self, input_tool: Input):
        self._input = input_tool

    @infer
    async def propose_and_confirm_review_files(
        self,
        understanding: ProjectUnderstanding,
        scope: ProjectScope,
        all_file_paths: list[str],
    ) -> list[str]:
        """
        Returns the final tracked files selected for code review.

        The initial proposal is based on the project understanding, requested
        focus areas, exclusions, public interfaces, central implementation
        files, configuration files relevant to the requested focus, and
        representative tests. The final list reflects the user's confirmation
        or adjustment through the Input tool. Returned paths are unique members
        of all_file_paths.
        """
        ...


class CodingStyleAuditor:
    @infer
    async def review(self, file_contents: dict[str, RawCode]) -> list[CodeIssue]:
        """
        Returns coding-style issues found in the supplied files.

        A coding-style issue is a concrete naming, formatting, comment, or local
        consistency problem that reduces readability in the supplied code. The
        result excludes subjective preferences, ordinary string literals,
        ordinary numeric values, punctuation used only for rendering, parameter
        names required by language convention, and constants whose extraction
        would make the code less clear.

        A duplicated literal or magic value is treated as an issue only when it
        represents shared domain meaning, configuration, protocol text, storage
        keys, external API names, or a value likely to change in multiple places.
        The returned issues use supplied file paths and specific line numbers.
        CodeIssue.perspective is assigned by the caller and may be empty here.
        """
        ...


class DesignPrincipleArchitect:
    @infer
    async def review(self, file_contents: dict[str, RawCode]) -> list[CodeIssue]:
        """
        Returns software-design issues found in the supplied files.

        A design issue is a concrete responsibility, coupling, cohesion,
        abstraction, or boundary problem whose improvement would make the code
        easier to evolve. The result excludes thin functions that intentionally
        define workflow stages, persistence boundaries, logging boundaries,
        resumable execution boundaries, or agent/orchestration steps.

        A wrapper is treated as a design issue only when it hides important
        behavior, duplicates nontrivial logic, or creates a misleading
        abstraction. The returned issues use supplied file paths and specific
        line numbers. CodeIssue.perspective is assigned by the caller and may be
        empty here.
        """
        ...


class MaintainabilityAssessor:
    @infer
    async def review(self, file_contents: dict[str, RawCode]) -> list[CodeIssue]:
        """
        Returns maintainability and readability issues found in the supplied
        files.

        A maintainability issue is a concrete problem involving control-flow
        complexity, unclear state transitions, weak error handling, missing
        validation at a meaningful boundary, hard-to-test behavior, or
        misleading structure. The result excludes speculative scalability
        concerns, production-hardening suggestions for intentionally small
        examples, and refactors whose main benefit is stylistic.

        File-based persistence, simple session files, and small CLI examples are
        treated as acceptable unless the supplied code itself shows a concrete
        correctness, concurrency, or data-loss risk. The returned issues use
        supplied file paths and specific line numbers. CodeIssue.perspective is
        assigned by the caller and may be empty here.
        """
        ...


class DependencySpecialist:
    @infer
    async def review(self, file_contents: dict[str, RawCode]) -> list[CodeIssue]:
        """
        Returns external-dependency issues found in the supplied files.

        A dependency issue is a concrete problem involving dependency
        declarations, optional dependency boundaries, version constraints,
        unnecessary runtime imports, unsafe library use, or mismatches between
        declared packages and imported packages. Version-constraint issues are
        based on dependency metadata files supplied in file_contents, such as
        pyproject.toml, requirements files, lock files, or package manifests.

        The result excludes claims inferred only from an import statement when
        the relevant dependency declaration file is not supplied. Conditional
        imports are treated as acceptable when they support optional features and
        produce clear errors for missing extras. The returned issues use supplied
        file paths and specific line numbers. CodeIssue.perspective is assigned
        by the caller and may be empty here.
        """
        ...


class ReportingAgent:
    @infer
    async def create_report(
        self,
        all_issues: list[CodeIssue],
        understanding: ProjectUnderstanding,
    ) -> QualityReport:
        """
        Creates the final code-quality report from reviewed issues and project
        understanding.

        The report contains a concise overall summary and issues grouped by
        perspective. Duplicate issues are merged. Low-value nits are omitted
        when their suggested change would not clearly improve readability,
        maintainability, correctness, or design clarity. Issue descriptions and
        suggestions remain faithful to the supplied issue details and do not add
        unsupported findings.
        """
        ...

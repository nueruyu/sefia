from pathlib import Path

from glyff import identify
from sefia import infer

from ..common.human_input import HumanInputTool
from .models import CodeIssue, ProjectScope, ProjectUnderstanding, QualityReport
from .tools import FileTool


@identify("ScopingAgent")
class ScopingAgent:
    def __init__(self, human_input: HumanInputTool):
        self._human_input = human_input

    @infer()
    async def define_scope(self, user_request: str) -> ProjectScope:
        """
        Clarify the user's request and define a concrete code-review scope.

        Determine the Git repository path, review focus areas, and files or
        patterns that should be excluded. Use the HumanInputTool to ask one
        focused question at a time when critical details are missing. Preserve
        explicit paths exactly as the user provides them.
        """
        ...


@identify("UnderstandingAgent")
class UnderstandingAgent:
    def __init__(self, file_tool: FileTool):
        self._file_tool = file_tool

    @infer()
    async def _prioritize_files_to_read(
        self,
        current_understanding: ProjectUnderstanding,
        unread_file_paths: list[str],
    ) -> list[str]:
        """
        Select up to three unread files that will most improve the current
        project understanding.

        Prioritize README files, dependency and build configuration, main entry
        points, public interfaces, and representative tests. Return only paths
        from unread_file_paths. Return an empty list when the understanding is
        already sufficient for selecting files to review.
        """
        ...

    @infer()
    async def _update_understanding(
        self,
        current_understanding: ProjectUnderstanding,
        new_file_contents: dict[str, str],
    ) -> ProjectUnderstanding:
        """
        Update the project understanding using the newly read file contents.

        Refine the concise project summary, detected technology stack, and key
        components. Preserve useful facts from the current understanding and do
        not claim details that are unsupported by the provided files.
        """
        ...

    async def deepen_understanding(
        self,
        current_understanding: ProjectUnderstanding,
        all_file_paths: list[str],
        project_root: str,
    ) -> ProjectUnderstanding:
        unread_files = [
            path
            for path in all_file_paths
            if path not in current_understanding.read_files
        ]
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


@identify("ReviewScopingAgent")
class ReviewScopingAgent:
    def __init__(self, human_input: HumanInputTool):
        self._human_input = human_input

    @infer()
    async def propose_and_confirm_review_files(
        self,
        understanding: ProjectUnderstanding,
        scope: ProjectScope,
        all_file_paths: list[str],
    ) -> list[str]:
        """
        Propose the most relevant tracked files for review, then use the
        HumanInputTool to ask the user to confirm or adjust the selection.

        Respect the requested focus areas and exclusions. Return only paths from
        all_file_paths, without duplicates, after the user has approved the
        final selection.
        """
        ...


@identify("CodingStyleAuditor")
class CodingStyleAuditor:
    @infer()
    async def review(self, file_contents: dict[str, str]) -> list[CodeIssue]:
        """
        Review the given files from a coding-style perspective.

        Report actionable issues involving naming conventions, formatting,
        comments, duplicated literals, and magic values. Use precise file paths
        and line numbers. Do not report purely subjective preferences or invent
        issues when the code is already clear and consistent.
        """
        ...


@identify("DesignPrincipleArchitect")
class DesignPrincipleArchitect:
    @infer()
    async def review(self, file_contents: dict[str, str]) -> list[CodeIssue]:
        """
        Review the given files from a software-design-principles perspective.

        Focus on SOLID, DRY, cohesion, coupling, responsibility boundaries, and
        useful abstractions. Report only concrete, actionable issues with precise
        file paths and line numbers.
        """
        ...


@identify("MaintainabilityAssessor")
class MaintainabilityAssessor:
    @infer()
    async def review(self, file_contents: dict[str, str]) -> list[CodeIssue]:
        """
        Review the given files from a maintainability and readability perspective.

        Focus on complexity, clarity, documentation, error handling, and
        testability. Report only concrete, actionable issues with precise file
        paths and line numbers.
        """
        ...


@identify("DependencySpecialist")
class DependencySpecialist:
    @infer()
    async def review(self, file_contents: dict[str, str]) -> list[CodeIssue]:
        """
        Review the given files from an external-dependency perspective.

        Focus on unnecessary dependencies, unsafe or incorrect library usage,
        unbounded or incompatible versions, and visible security risks. Report
        only evidence supported by the supplied files, with precise paths and
        line numbers.
        """
        ...


@identify("ReportingAgent")
class ReportingAgent:
    @infer()
    async def create_report(
        self,
        all_issues: list[CodeIssue],
        understanding: ProjectUnderstanding,
    ) -> QualityReport:
        """
        Consolidate the identified issues and project understanding into a final
        code-quality report.

        Write a concise overall summary and group every issue by its perspective.
        Preserve issue details exactly, avoid duplicating issues, and include
        perspectives with no issues only when that adds useful context.
        """
        ...

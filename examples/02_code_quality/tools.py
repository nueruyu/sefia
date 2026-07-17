import asyncio
import subprocess
from pathlib import Path

from sefia import exceptions


class FileOperationToolError(exceptions.ToolError):
    """Base for file-related tool errors."""

    def __init__(self, message: str, path: str):
        super().__init__(message)
        self.path = path


class FileNotFoundToolError(FileOperationToolError):
    """Raised when a file is not found."""


class PermissionDeniedToolError(FileOperationToolError):
    """Raised when a file cannot be accessed."""


class Git:
    """A tool for interacting with a Git repository."""

    async def list_tracked_files(self, path: str) -> list[str]:
        """
        Execute 'git ls-files' in the specified directory and return the tracked
        file paths relative to the repository root.
        """
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["git", "ls-files", "-z"],
                cwd=path,
                capture_output=True,
                check=True,
            )
            output = result.stdout.decode("utf-8")
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            OSError,
            UnicodeError,
        ) as exc:
            raise exceptions.ToolError(
                f"Failed to list git-tracked files in '{path}': {exc}"
            ) from exc

        return [path for path in output.split("\0") if path]


class Files:
    """A tool for file system operations."""

    async def read_files(self, full_paths: list[str]) -> dict[str, str]:
        """
        Read multiple UTF-8 text files and return a mapping from each path to its
        content. Store an error message for files that cannot be read.
        """

        async def _read_single(path_str: str) -> tuple[str, str]:
            try:
                content = await asyncio.to_thread(
                    Path(path_str).read_text,
                    encoding="utf-8",
                )
                return path_str, content
            except FileNotFoundError as exc:
                raise FileNotFoundToolError(
                    f"File not found: {exc}", path=path_str
                ) from exc
            except PermissionError as exc:
                raise PermissionDeniedToolError(
                    f"Permission denied: {exc}", path=path_str
                ) from exc
            except (OSError, UnicodeError) as exc:
                raise exceptions.ToolError(f"Error reading file: {exc}") from exc

        results = await asyncio.gather(*(_read_single(path) for path in full_paths))
        return dict(results)

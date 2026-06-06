import asyncio
import subprocess
from pathlib import Path

from sefia import tool


class GitTool:
    """A tool for interacting with a Git repository."""

    @tool
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
            raise RuntimeError(
                f"Failed to list git-tracked files in '{path}': {exc}"
            ) from exc

        return [path for path in output.split("\0") if path]


class FileTool:
    """A tool for file system operations."""

    @tool
    async def read_files(self, full_paths: list[str]) -> dict[str, str]:
        """
        Read multiple UTF-8 text files and return a mapping from each path to its
        content. Store an error message for files that cannot be read.
        """
        contents: dict[str, str] = {}
        for path_str in full_paths:
            try:
                contents[path_str] = await asyncio.to_thread(
                    Path(path_str).read_text,
                    encoding="utf-8",
                )
            except (OSError, UnicodeError) as exc:
                contents[path_str] = f"Error reading file: {exc}"
        return contents

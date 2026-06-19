import subprocess
from importlib import import_module

import pytest
from sefia import exceptions

# ``02_code_quality`` starts with a digit, so it must be loaded via importlib.
tools = import_module("examples.02_code_quality.tools")


def _init_repo(path):
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@example.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


class TestGitTool:
    async def test_lists_tracked_files(self, tmp_path):
        _init_repo(tmp_path)
        (tmp_path / "tracked.py").write_text("print('hi')", encoding="utf-8")
        (tmp_path / "untracked.py").write_text("ignored", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.py"], cwd=tmp_path, check=True)

        files = await tools.GitTool().list_tracked_files(str(tmp_path))

        assert files == ["tracked.py"]

    async def test_empty_repo_returns_no_files(self, tmp_path):
        _init_repo(tmp_path)

        files = await tools.GitTool().list_tracked_files(str(tmp_path))

        assert files == []

    async def test_non_git_directory_raises_tool_error(self, tmp_path):
        with pytest.raises(exceptions.ToolError):
            await tools.GitTool().list_tracked_files(str(tmp_path))


class TestFileTool:
    async def test_reads_multiple_files(self, tmp_path):
        first = tmp_path / "a.txt"
        second = tmp_path / "b.txt"
        first.write_text("alpha", encoding="utf-8")
        second.write_text("beta", encoding="utf-8")

        contents = await tools.FileTool().read_files([str(first), str(second)])

        assert contents == {str(first): "alpha", str(second): "beta"}

    async def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "missing.txt"

        with pytest.raises(exceptions.FileNotFoundToolError) as exc_info:
            await tools.FileTool().read_files([str(missing)])

        assert exc_info.value.path == str(missing)

    async def test_reads_empty_list(self):
        assert await tools.FileTool().read_files([]) == {}

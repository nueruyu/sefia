import typer
from typer.testing import CliRunner

from examples._common.typer_utils import add_session_commands, async_command
from sefios.cli.exceptions import UnknownSessionError

runner = CliRunner()


class _FakeSessions:
    def __init__(self):
        self.sessions: list[str] = []
        self.active: str | None = None

    def create_session(self) -> str:
        session_id = f"session-{len(self.sessions)}"
        self.sessions.append(session_id)
        self.active = session_id
        return session_id

    def switch_session(self, session_id: str) -> str:
        if session_id not in self.sessions:
            raise UnknownSessionError(session_id)
        self.active = session_id
        return session_id


class TestAsyncCommand:
    def test_runs_coroutine_synchronously(self):
        @async_command
        async def double(value: int) -> int:
            return value * 2

        assert double(21) == 42


class TestSessionCommands:
    def _app(self, sessions: _FakeSessions) -> typer.Typer:
        app = typer.Typer()
        add_session_commands(app, sessions)
        return app

    def test_new_creates_and_activates_session(self):
        sessions = _FakeSessions()

        result = runner.invoke(self._app(sessions), ["session", "new"])

        assert result.exit_code == 0
        assert sessions.active == "session-0"
        assert "session-0" in result.output

    def test_switch_to_known_session(self):
        sessions = _FakeSessions()
        sessions.create_session()
        sessions.create_session()

        result = runner.invoke(self._app(sessions), ["session", "switch", "session-0"])

        assert result.exit_code == 0
        assert sessions.active == "session-0"

    def test_switch_to_unknown_session_fails(self):
        sessions = _FakeSessions()

        result = runner.invoke(self._app(sessions), ["session", "switch", "ghost"])

        assert result.exit_code == 1
        assert "ghost" in result.output

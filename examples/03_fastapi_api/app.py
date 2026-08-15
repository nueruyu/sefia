import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from sefios import SQLitePersistence
from sefios.fastapi import SefiaHTTP
from sefios.fastapi.exceptions import (
    AmbiguousInputError,
    InputRequired,
    UnknownInputError,
    UnknownSessionError,
)

from .agents import Interviewer
from .models import (
    BriefSchema,
    InputRequiredResponse,
    InterviewCompletedResponse,
    InterviewResponse,
    SessionCreatedResponse,
    TurnRequest,
)

EXAMPLE_DIR = Path(__file__).parent


def create_app(sefia_http: SefiaHTTP | None = None) -> FastAPI:
    from dotenv import load_dotenv

    load_dotenv()
    api = sefia_http or SefiaHTTP(
        model=os.environ.get("EXAMPLE_DEFAULT_MODEL", "gpt-4o-mini"),
        persistence=SQLitePersistence(EXAMPLE_DIR / ".local" / "sessions.sqlite3"),
    )
    interviewer = Interviewer(api.input_tool)
    app = FastAPI(title="Sefia FastAPI Example")

    async def unknown_session(_request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, UnknownSessionError)
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    async def unknown_input(_request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, UnknownInputError)
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    async def ambiguous_input(_request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, AmbiguousInputError)
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "interaction_ids": exc.interaction_ids},
        )

    async def input_required(_request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, InputRequired)
        # The Input tool always identifies its request, so a pause surfaced to
        # the client carries an interaction_id (the core type allows None for
        # tools that don't).
        assert exc.interaction_id is not None
        return JSONResponse(
            status_code=200,
            content=InputRequiredResponse(
                interaction_id=exc.interaction_id,
                prompt=exc.prompt,
            ).model_dump(),
        )

    app.add_exception_handler(UnknownSessionError, unknown_session)
    app.add_exception_handler(UnknownInputError, unknown_input)
    app.add_exception_handler(AmbiguousInputError, ambiguous_input)
    app.add_exception_handler(InputRequired, input_required)

    async def index() -> FileResponse:
        return FileResponse(EXAMPLE_DIR / "index.html")

    async def create_session() -> SessionCreatedResponse:
        return SessionCreatedResponse(session_id=api.create_session())

    async def events(session_id: str):
        return api.events(session_id)

    async def interview(session_id: str, body: TurnRequest) -> InterviewResponse:
        # Streaming forwards the parsed prompt text as `delta` events so the UI
        # can type the question out live; the raw structured @infer envelope is
        # never exposed -- only the decoded prompt/message text is streamed.
        async with api.session(session_id=session_id) as session:
            await session.accept_input(body.input, reply_to=body.reply_to)
            brief = await interviewer.run()
            return InterviewCompletedResponse(brief=BriefSchema.from_brief(brief))

    app.add_api_route("/", index, methods=["GET"], include_in_schema=False)
    app.add_api_route(
        "/sessions",
        create_session,
        methods=["POST"],
        response_model=SessionCreatedResponse,
    )
    app.add_api_route(
        "/sessions/{session_id}/events",
        events,
        methods=["GET"],
    )
    app.add_api_route(
        "/sessions/{session_id}/interview",
        interview,
        methods=["POST"],
        response_model=InterviewResponse,
    )

    return app


app = create_app()

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from sefios.fastapi import (
    AmbiguousInputError,
    InputRequired,
    SefiaHTTP,
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
        session_dir=EXAMPLE_DIR / ".local",
        model=os.environ.get("EXAMPLE_DEFAULT_MODEL", "gpt-4o-mini"),
    )
    interviewer = Interviewer(api.input_tool)
    app = FastAPI(title="Sefia FastAPI Example")

    @app.exception_handler(UnknownSessionError)
    async def _unknown_session(request: Request, exc: UnknownSessionError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(UnknownInputError)
    async def _unknown_input(request: Request, exc: UnknownInputError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(AmbiguousInputError)
    async def _ambiguous_input(request: Request, exc: AmbiguousInputError):
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "interaction_ids": exc.interaction_ids},
        )

    @app.exception_handler(InputRequired)
    async def _input_required(request: Request, exc: InputRequired):
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

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(EXAMPLE_DIR / "index.html")

    @app.post("/sessions", response_model=SessionCreatedResponse)
    async def create_session() -> SessionCreatedResponse:
        return SessionCreatedResponse(session_id=api.create_session())

    @app.get("/sessions/{session_id}/events")
    async def events(session_id: str):
        return api.events(session_id)

    @app.post("/sessions/{session_id}/interview", response_model=InterviewResponse)
    async def interview(session_id: str, body: TurnRequest) -> InterviewResponse:
        # Streaming forwards the parsed prompt text as `delta` events so the UI
        # can type the question out live; the raw structured @infer envelope is
        # never exposed -- only the decoded prompt/message text is streamed.
        async with api.session(session_id=session_id) as session:
            await session.accept_input(body.input, reply_to=body.reply_to)
            brief = await interviewer.run()
            return InterviewCompletedResponse(brief=BriefSchema.from_brief(brief))

    return app


app = create_app()

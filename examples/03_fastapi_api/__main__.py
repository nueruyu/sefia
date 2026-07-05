"""Run the example server with ``python -m examples.03_fastapi_api``."""

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    uvicorn.run(
        "examples.03_fastapi_api.app:create_app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    main()

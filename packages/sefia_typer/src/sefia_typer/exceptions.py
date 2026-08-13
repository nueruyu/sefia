class UnknownSessionError(Exception):
    """Raised when a requested CLI session is not known."""

    def __init__(self, session_id: str):
        super().__init__(f"Unknown session: {session_id}")
        self.session_id = session_id


__all__ = ["UnknownSessionError"]

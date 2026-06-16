from textwrap import dedent

from rich.console import Console
from rich.panel import Panel


def print_session_interrupted_hint(console: Console) -> None:
    """Prints a standardized hint to the console when a session is interrupted."""
    hint = dedent(
        """
        Session interrupted to wait for your input.
        To resume, run the script again with your answer.
        """
    ).strip()
    console.print()
    console.print(Panel(hint, title="WAITING FOR INPUT", border_style="yellow"))

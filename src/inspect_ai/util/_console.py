import readline  # noqa: F401
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console


@contextmanager
def input_screen(header: str | None = None) -> Iterator[Console]:
    """Input screen for receiving user input.

    Context manager that clears the task display and provides a
    screen for receiving console input.

    Args:
       header (str | None): Header line to print above console
         content (defaults to printing no header)

    Returns:
       Console to use for input.
    """
    from inspect_ai._display._display import task_screen

    with task_screen().input_screen(header=header) as console:
        yield console
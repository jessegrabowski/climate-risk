import logging

from typing import Literal

LogLevel = Literal["debug", "info", "warning", "error", "critical"]


def _log_maybe(
    msg: str,
    *args: object,
    logger: logging.Logger,
    verbose: bool = False,
    level: LogLevel = "info",
) -> None:
    """
    Log a message only when the caller asked for it.

    Parameters
    ----------
    msg : str
        The message to log.
    *args : object
        Arguments for the message's format placeholders.
    logger : Logger
        Logger to write to.
    verbose : bool, optional
        Emit the message. Default False.
    level : str, optional
        Method of ``logger`` to call. Default 'info'.
    """
    if verbose:
        # Attribute the record to the caller rather than to this line.
        getattr(logger, level)(msg, *args, stacklevel=2)

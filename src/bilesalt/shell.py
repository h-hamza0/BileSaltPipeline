from __future__ import annotations

import logging
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, Union

LOGGER = logging.getLogger("bilesalt")

PathLike = Union[str, Path]


class CommandError(RuntimeError):
    def __init__(self, command: Sequence[str], returncode: Optional[int], stderr: str = "") -> None:
        self.command = list(command)
        self.returncode = returncode
        self.stderr = stderr
        tail = "\n".join(stderr.strip().splitlines()[-15:])
        super().__init__(f"Command failed ({returncode}): {' '.join(self.command)}\n{tail}".rstrip())


def run(
    command: Sequence[object],
    *,
    stdin: Optional[str] = None,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    argv = [str(part) for part in command]
    LOGGER.info("$ %s", " ".join(argv))
    try:
        result = subprocess.run(
            argv,
            input=stdin,
            text=True,
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            capture_output=capture,
        )
    except FileNotFoundError as exc:
        raise CommandError(argv, None, f"executable not found: {argv[0]}") from exc
    if capture and result.stdout:
        LOGGER.debug(result.stdout)
    if check and result.returncode != 0:
        raise CommandError(argv, result.returncode, result.stderr or "")
    return result

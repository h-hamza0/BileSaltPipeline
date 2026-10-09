from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Dict, Union

Value = Union[str, int, float, Sequence[str]]


def _canonical(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _render(value: Value) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


class MdpFile:
    def __init__(self, options: Mapping[str, str] | None = None) -> None:
        self._options: Dict[str, str] = {}
        self._spelling: Dict[str, str] = {}
        for key, value in (options or {}).items():
            self[key] = value

    @classmethod
    def read(cls, path: Union[str, Path]) -> MdpFile:
        mdp = cls()
        for raw in Path(path).read_text().splitlines():
            line = raw.split(";", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            mdp[key.strip()] = value.strip()
        return mdp

    def __setitem__(self, key: str, value: Value) -> None:
        canon = _canonical(key)
        spelling = self._spelling.setdefault(canon, key.strip())
        self._options[spelling] = _render(value)

    def __getitem__(self, key: str) -> str:
        return self._options[self._spelling[_canonical(key)]]

    def __contains__(self, key: str) -> bool:
        return _canonical(key) in self._spelling

    def get(self, key: str, default: str | None = None) -> str | None:
        return self[key] if key in self else default

    def update(self, overrides: Mapping[str, Value]) -> None:
        for key, value in overrides.items():
            self[key] = value

    def items(self) -> Iterable[tuple[str, str]]:
        return self._options.items()

    def write(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text("".join(f"{key} = {value}\n" for key, value in self._options.items()))
        return path

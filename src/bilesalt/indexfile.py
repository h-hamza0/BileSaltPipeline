from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union


def read_groups(path: Union[str, Path]) -> Dict[str, int]:
    groups: Dict[str, int] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            groups.setdefault(line.strip("[] ").split()[0], len(groups))
    return groups


def read_atoms(path: Union[str, Path]) -> Dict[str, List[int]]:
    members: Dict[str, List[int]] = {}
    current = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("["):
            current = line.strip("[] ").split()[0]
            members[current] = []
        elif current is not None:
            members[current].extend(int(token) - 1 for token in line.split())
    return members


def group_number(groups: Dict[str, int], name: str) -> int:
    try:
        return groups[name]
    except KeyError:
        raise KeyError(f"Index group '{name}' not found; available: {', '.join(groups)}") from None

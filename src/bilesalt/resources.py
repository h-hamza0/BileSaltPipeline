from __future__ import annotations

from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parent / "data"
MDP_DIR = DATA_DIR / "mdp"


def mdp_template(name: str) -> Path:
    path = MDP_DIR / f"{name}.mdp"
    if not path.exists():
        raise FileNotFoundError(f"No bundled MDP template named '{name}'")
    return path


def available_templates() -> List[str]:
    return sorted(p.stem for p in MDP_DIR.glob("*.mdp"))

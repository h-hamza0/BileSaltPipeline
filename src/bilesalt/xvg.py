from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np


def read_xvg(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, str]]:
    rows = []
    attrs: Dict[str, str] = {"xaxis": "", "yaxis": "", "title": ""}
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#&":
            continue
        if stripped[0] == "@":
            tokens = stripped.split(None, 2)
            if len(tokens) == 3 and tokens[1] in attrs and '"' in tokens[2]:
                attrs[tokens[1]] = tokens[2].split('"')[1]
            continue
        try:
            rows.append([float(token) for token in stripped.split()])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric data in {path}")
    width = min(len(row) for row in rows)
    return np.array([row[:width] for row in rows]), attrs

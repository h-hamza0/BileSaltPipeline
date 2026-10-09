from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from bilesalt import indexfile
from bilesalt.gromacs import Gromacs


@dataclass
class AnalysisContext:
    location: Path
    name: str
    option: str
    gmx: Gromacs

    @property
    def stage(self) -> Path:
        return self.location / self.name

    def file(self, suffix: str) -> Path:
        return self.stage / f"{self.name}{suffix}"

    @property
    def output(self) -> Path:
        return self.stage / f"{self.option}.xvg"

    @property
    def setup(self) -> Path:
        nested = self.location / "AA" / "setup"
        return nested if nested.exists() else self.location / "setup"

    @property
    def index(self) -> Path:
        return self.setup / "index.ndx"

    @property
    def topology(self) -> Path:
        return self.setup / "topol.top"

    def groups(self, index: Optional[Path] = None) -> Dict[str, int]:
        return indexfile.read_groups(index or self.index)

    def group(self, name: str, index: Optional[Path] = None) -> int:
        return indexfile.group_number(self.groups(index), name)

    def atoms(self, index: Optional[Path] = None) -> Dict[str, List[int]]:
        return indexfile.read_atoms(index or self.index)


def as_list(option: Union[str, List[str], None]) -> List[str]:
    if option is None:
        return []
    return [option] if isinstance(option, str) else list(option)

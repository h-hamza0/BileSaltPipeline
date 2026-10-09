from __future__ import annotations

from pathlib import Path
from typing import List


class TopologyBuilder:
    def __init__(self, system) -> None:
        self.system = system

    def render(self) -> str:
        forcefield = self.system.forcefield
        files = sorted(p for p in forcefield.iterdir() if p.suffix == ".itp")
        lines: List[str] = []
        lines += [
            f"#include <{p.resolve()}>" for p in files if p.name == "forcefield.itp" or "martini" in p.name
        ]
        for molecule in self.system.molecules:
            lines += [f"#include <{p.resolve()}>" for p in files if p.name == f"{molecule.name}_ffbonded.itp"]
        lines += [f"#include <{p.resolve()}>" for p in files if p.name in {"ions.itp", "tip3p.itp"}]
        lines += [f"#include <{m.itp_file}>" for m in self.system.molecules if m.itp_file is not None]
        lines += ["", "[ system ]", "System", "", "[ molecules ]"]
        lines += [f"{m.name}   {m.nmol}" for m in self.system.molecules]
        return "\n".join(lines) + "\n"

    def write(self) -> Path:
        path = self.system.directory / "setup" / "topol.top"
        path.write_text(self.render())
        self.system.topology = path
        return path

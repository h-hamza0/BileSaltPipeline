from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from bilesalt import shell


class BackmappingError(RuntimeError):
    pass


class Backmapper:
    def __init__(
        self,
        executable: str = "cg2at",
        gmx: str = "gmx",
        overlap: float = 0.25,
        answers: Sequence[str] = ("12", "10", "2"),
    ) -> None:
        self.executable = executable
        self.gmx = gmx
        self.overlap = overlap
        self.answers = tuple(answers)

    def command(self, structure: Path, location: str) -> list:
        return [
            self.executable,
            "-c",
            structure.resolve(),
            "-gmx",
            self.gmx,
            "-loc",
            location,
            "-ov",
            self.overlap,
        ]

    def convert(self, structure: Path, workdir: Path, location: str = "backmap") -> Path:
        structure = Path(structure)
        if not structure.exists():
            raise BackmappingError(f"Coarse-grained structure not found: {structure}")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        shell.run(self.command(structure, location), stdin="\n".join(self.answers) + "\n", cwd=workdir)
        final = workdir / location / "FINAL" / "final_cg2at_de_novo.pdb"
        if not final.exists():
            raise BackmappingError(f"cg2at finished without producing {final}")
        return final.resolve()

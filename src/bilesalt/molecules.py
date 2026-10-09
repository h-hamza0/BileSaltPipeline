from __future__ import annotations

import shutil
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from bilesalt import shell
from bilesalt.config import MoleculeConfig


class Molecule:
    def __init__(self, config: MoleculeConfig) -> None:
        self.config = config
        self.name = config.name
        self.nmol = config.nmol
        self.itp_file: Optional[Path] = None
        self.gro_file: Optional[Path] = None
        self.pdb_file: Optional[Path] = None

    @property
    def insert(self) -> bool:
        return self.config.insert

    @property
    def replace(self) -> Optional[str]:
        return self.config.replace


class MoleculeManager:
    def __init__(self, system) -> None:
        self.system = system
        self.molecules: OrderedDict[str, Molecule] = OrderedDict()

    def add(self, config: MoleculeConfig) -> Molecule:
        molecule = Molecule(config)
        self.molecules[molecule.name] = molecule
        return molecule

    def __iter__(self) -> Iterator[Molecule]:
        return iter(self.molecules.values())

    def __contains__(self, name: str) -> bool:
        return name in self.molecules

    def get(self, name: str) -> Molecule:
        return self.molecules[name]

    def set_count(self, name: str, count: int) -> None:
        self.molecules[name].nmol = count

    def _find(self, name: str, suffix: str) -> Optional[Path]:
        root = self.system.config.input_dir
        phase = self.system.config.phase
        candidates = [root / phase / f"{name}{suffix}", root / f"{name}{suffix}"]
        candidates += [root / other / f"{name}{suffix}" for other in ("AA", "CG") if other != phase]
        return next((c for c in candidates if c.exists()), None)

    def copy_itp(self) -> None:
        target = self.system.directory / "setup" / "itp"
        target.mkdir(parents=True, exist_ok=True)
        for molecule in self:
            if not molecule.config.itp:
                continue
            source = self._find(molecule.name, ".itp")
            if source is None:
                raise FileNotFoundError(f"No {molecule.name}.itp found under {self.system.config.input_dir}")
            destination = target / source.name
            shutil.copy(source, destination)
            molecule.itp_file = destination.resolve()

    def copy_gro(self) -> None:
        target = self.system.directory / "setup" / "molecules"
        target.mkdir(parents=True, exist_ok=True)
        for molecule in self:
            if not (molecule.insert or molecule.config.packmol):
                continue
            source = self._find(molecule.name, ".gro")
            if source is None:
                raise FileNotFoundError(f"No {molecule.name}.gro found under {self.system.config.input_dir}")
            destination = target / source.name
            shutil.copy(source, destination)
            molecule.gro_file = destination.resolve()
            if molecule.config.packmol:
                molecule.pdb_file = destination.with_suffix(".pdb").resolve()
                self.system.gmx.to_pdb(destination, molecule.pdb_file)

    def insert(self) -> None:
        state = self.system.state
        setup = self.system.directory / "setup"
        for molecule in self:
            if not (molecule.insert or molecule.config.packmol):
                continue
            if molecule.config.packmol:
                output = setup / f"{molecule.name}_boxed.pdb"
                self.pack(state.structure, molecule.pdb_file, output)
                self.system.gmx.center(output, output, self.system.config.box_size)
            else:
                output = setup / f"{molecule.name}_boxed.gro"
                self.system.gmx.insert_molecules(
                    molecule.gro_file,
                    output,
                    molecule.nmol,
                    molecule.config.insertion_radius,
                    structure=state.structure,
                    box=self.system.config.box_size if state.structure is None else None,
                    replace=molecule.replace,
                )
            state.structure = output.resolve()

    def pack(
        self,
        base: Optional[Path],
        molecule: Path,
        output: Path,
        separation: float = 100.0,
        tolerance: float = 2.0,
    ) -> None:
        if base is None:
            raise ValueError("packmol insertion requires an existing structure")
        spec = (
            f"tolerance {tolerance}\noutput {output.resolve()}\nfiletype pdb\n"
            f"structure {base}\n  number 1\n  fixed 0. 0. 0. 0. 0. 0.\nend structure\n"
            f"structure {molecule}\n  number 1\n  fixed {separation}. 0. 0. 0. 0. 0.\nend structure\n"
        )
        shell.run(["packmol"], stdin=spec)

    def manage(self) -> None:
        self.copy_itp()
        self.copy_gro()
        self.insert()

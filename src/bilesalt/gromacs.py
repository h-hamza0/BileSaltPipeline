from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Dict, List, Optional

from bilesalt import indexfile, shell
from bilesalt.shell import PathLike


class Gromacs:
    def __init__(self, executable: str = "gmx") -> None:
        self.executable = executable

    def run(
        self,
        *args: object,
        stdin: Optional[str] = None,
        cwd: Optional[PathLike] = None,
        capture: bool = True,
        check: bool = True,
    ):
        return shell.run([self.executable, *args], stdin=stdin, cwd=cwd, capture=capture, check=check)

    def grompp(
        self,
        mdp: PathLike,
        structure: PathLike,
        topology: PathLike,
        output: PathLike,
        index: Optional[PathLike] = None,
        restraints: Optional[PathLike] = None,
        maxwarn: int = 10,
    ) -> None:
        args: List[object] = [
            "grompp",
            "-f",
            mdp,
            "-c",
            structure,
            "-p",
            topology,
            "-o",
            output,
            "-po",
            Path(output).with_name("mdout.mdp"),
            "-maxwarn",
            maxwarn,
        ]
        if restraints is not None:
            args += ["-r", restraints]
        if index is not None:
            args += ["-n", index]
        self.run(*args)

    def mdrun(
        self,
        deffnm: str,
        ntmpi: int = 1,
        ntomp: int = 12,
        pin: str = "off",
        nb: str = "gpu",
        checkpoint: Optional[PathLike] = None,
        cwd: Optional[PathLike] = None,
    ) -> int:
        args: List[object] = [
            "mdrun",
            "-deffnm",
            deffnm,
            "-ntmpi",
            ntmpi,
            "-pin",
            pin,
            "-ntomp",
            ntomp,
            "-nb",
            nb,
            "-v",
        ]
        if checkpoint is not None:
            args += ["-cpi", checkpoint, "-cpt", 5]
        return self.run(*args, cwd=cwd, capture=False, check=False).returncode

    def genion(
        self,
        tpr: PathLike,
        output: PathLike,
        topology: PathLike,
        group: int,
        positive: str = "NA",
        negative: str = "CLA",
    ) -> None:
        self.run(
            "genion",
            "-s",
            tpr,
            "-o",
            output,
            "-p",
            topology,
            "-pname",
            positive,
            "-nname",
            negative,
            "-neutral",
            stdin=f"{group}\n",
        )

    def insert_molecules(
        self,
        molecule: PathLike,
        output: PathLike,
        number: int,
        radius: float,
        structure: Optional[PathLike] = None,
        box: Optional[Sequence[float]] = None,
        replace: Optional[str] = None,
        tries: int = 500,
    ) -> None:
        args: List[object] = [
            "insert-molecules",
            "-ci",
            molecule,
            "-o",
            output,
            "-nmol",
            number,
            "-radius",
            radius,
            "-try",
            tries,
        ]
        if structure is not None:
            args += ["-f", structure]
            if replace:
                args += ["-replace", replace]
        else:
            if box is None:
                raise ValueError("insert_molecules needs either a structure or a box")
            args += ["-box", *box]
        self.run(*args)

    def center(self, structure: PathLike, output: PathLike, box: Sequence[float]) -> None:
        self.run("editconf", "-f", structure, "-o", output, "-c", "-box", *box, "-pbc", "yes", "-bt", "cubic")

    def place(
        self, structure: PathLike, output: PathLike, center: Sequence[float], box: Sequence[float]
    ) -> None:
        self.run("editconf", "-f", structure, "-o", output, "-center", *center, "-box", *box)

    def to_pdb(self, structure: PathLike, output: PathLike) -> None:
        self.run("editconf", "-f", structure, "-o", output)

    def solvate(
        self, structure: PathLike, solvent: PathLike, output: PathLike, radius: Optional[float] = None
    ) -> None:
        args: List[object] = ["solvate", "-cp", structure, "-cs", solvent, "-o", output]
        if radius is not None:
            args += ["-radius", radius]
        self.run(*args)

    def distance(
        self,
        trajectory: PathLike,
        tpr: PathLike,
        index: PathLike,
        group_a: str,
        group_b: str,
        output: PathLike,
    ) -> None:
        selection = f'com of group "{group_a}" plus com of group "{group_b}"'
        self.run("distance", "-s", tpr, "-f", trajectory, "-n", index, "-select", selection, "-oall", output)

    def dump_frame(
        self,
        trajectory: PathLike,
        tpr: PathLike,
        time: float,
        output: PathLike,
        index: Optional[PathLike] = None,
        group: str = "0",
    ) -> None:
        args: List[object] = ["trjconv", "-s", tpr, "-f", trajectory, "-dump", time, "-o", output]
        if index is not None:
            args += ["-n", index]
        self.run(*args, stdin=f"{group}\n")

    def make_index(
        self, reference: PathLike, output: PathLike, definitions: Mapping[str, Sequence[str]]
    ) -> Dict[str, int]:
        self.run("make_ndx", "-f", reference, "-o", output, stdin="q\n")
        existing = indexfile.read_groups(output)
        next_id = len(existing)
        script = ""
        for name, members in definitions.items():
            present = [m for m in members if m in existing]
            for missing in set(members) - set(present):
                shell.LOGGER.warning("Index group '%s' is not present in %s", missing, reference)
            if not present:
                continue
            script += "|".join(str(existing[m]) for m in present) + f"\nname {next_id} {name}\n"
            next_id += 1
        self.run("make_ndx", "-f", reference, "-o", output, stdin=script + "q\n")
        return indexfile.read_groups(output)

    def energy_term_index(self, edr: PathLike, term: str) -> int:
        listing = self.run("energy", "-f", edr, stdin="0\n", check=False)
        text = listing.stdout + listing.stderr
        tokens = text.replace("\n", " ").split()
        for position, token in enumerate(tokens[:-1]):
            if tokens[position + 1] == term and token.isdigit():
                return int(token)
        raise KeyError(f"Energy term '{term}' not found in {edr}")

    def energy(self, edr: PathLike, term: str, output: PathLike) -> None:
        index = self.energy_term_index(edr, term)
        self.run("energy", "-f", edr, "-o", output, stdin=f"{index}\n\n")

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from bilesalt import resources
from bilesalt.indexfile import group_number


def rotation_to_axis(vector: np.ndarray, axis: np.ndarray):
    vector = vector / np.linalg.norm(vector)
    cross = np.cross(vector, axis)
    norm = np.linalg.norm(cross)
    if norm < 1e-6:
        return 0.0, np.asarray(axis, dtype=float)
    angle = float(np.degrees(np.arccos(np.clip(np.dot(vector, axis), -1.0, 1.0))))
    return angle, cross / norm


class BoxManager:
    def __init__(self, system) -> None:
        self.system = system

    @property
    def setup(self) -> Path:
        return self.system.directory / "setup"

    def center(self) -> Path:
        output = self.setup / "centered.gro"
        self.system.gmx.center(
            self.system.state.structure, output, [int(self.system.config.box_distance)] * 3
        )
        self.system.state.structure = output.resolve()
        return output

    def solvate_cg(self) -> Path:
        output = self.setup / "solvated.gro"
        water = self.setup / "molecules" / "water.gro"
        self.system.gmx.solvate(self.system.state.structure, water, output, self.system.config.water_radius)
        self.system.state.structure = output.resolve()
        return output

    def solvate_aa(self) -> Path:
        output = self.setup / "solvated.gro"
        self.system.gmx.solvate(self.system.state.structure, f"{self.system.config.water_model}.gro", output)
        self.system.state.structure = output.resolve()
        return output

    def ionize(self) -> Path:
        output = self.setup / "ionized.gro"
        tpr = self.setup / "ions.tpr"
        gmx = self.system.gmx
        gmx.grompp(
            resources.mdp_template("ions"), self.system.state.structure, self.system.topology, tpr, maxwarn=0
        )
        group = group_number(self.system.groups, "SOL")
        gmx.genion(
            tpr,
            output,
            self.system.topology,
            group,
            self.system.config.positive_ion,
            self.system.config.negative_ion,
        )
        self.system.state.structure = output.resolve()
        return output

    def count_components(self) -> Counter:
        import MDAnalysis as mda

        universe = mda.Universe(str(self.system.state.structure))
        tracked = self.system.molecules.molecules
        counts = Counter(res.resname for res in universe.residues if res.resname in tracked)
        for name, count in counts.items():
            self.system.molecules.set_count(name, count)
        return counts

    def align(self) -> Path:
        import MDAnalysis as mda
        from MDAnalysis.transformations import rotateby

        config = self.system.config
        universe = mda.Universe(str(self.system.state.structure))
        peptide = universe.select_atoms("resname " + " ".join(config.peptide_residues))
        membrane = universe.select_atoms("resname " + " ".join(config.membrane_residues))
        if len(peptide) == 0 or len(membrane) == 0:
            raise ValueError("Alignment needs both peptide and membrane residues in the structure")
        direction = membrane.center_of_mass() - peptide.center_of_mass()
        angle, axis = rotation_to_axis(direction, np.array([0.0, 0.0, 1.0]))
        if angle:
            universe.trajectory.add_transformations(
                rotateby(angle, axis, point=universe.atoms.center_of_geometry())
            )
        output = self.system.directory / "targeted.gro"
        with mda.Writer(str(output)) as writer:
            writer.write(universe.atoms)
        self.system.state.structure = output.resolve()
        return output

    def elongate(self) -> Path:
        import MDAnalysis as mda

        universe = mda.Universe(str(self.system.state.structure))
        positions = universe.atoms.positions
        extent = (positions.max(axis=0) - positions.min(axis=0)) / 10.0
        lateral = (extent.max() + 1.0) / 2.0
        vertical = (extent[2] + 1.0) / 2.0
        length = self.system.config.box_distance
        output = self.system.directory / "elongated.gro"
        self.system.gmx.place(
            self.system.state.structure,
            output,
            center=[lateral * 1.5, lateral * 1.5, length - vertical],
            box=[lateral * 2.2, lateral * 2.2, length],
        )
        self.system.state.structure = output.resolve()
        return output

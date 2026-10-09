from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from bilesalt import resources
from bilesalt.config import StageConfig
from bilesalt.indexfile import read_atoms
from bilesalt.mdp import MdpFile
from bilesalt.shell import LOGGER
from bilesalt.slurm import split_evenly, wait_for_files
from bilesalt.xvg import read_xvg

CONTROL_OPTIONS = {"US", "lipid", "drug", "mdp_fname"}
DENSE_REGIONS = ((2.0, 4.0, 0.1),)
SPARSE_SPACING = 0.1
MAX_WINDOW_DISTANCE = 11.0
MAX_RESTARTS = 5


class StageError(RuntimeError):
    pass


def umbrella_targets(
    min_distance: float,
    max_distance: float,
    dense_regions: Sequence[Tuple[float, float, float]] = DENSE_REGIONS,
    sparse_spacing: float = SPARSE_SPACING,
) -> List[float]:
    targets: List[float] = []
    for start, end, spacing in dense_regions:
        targets += list(np.arange(start, end + spacing, spacing))
    current = np.ceil(min_distance * 10) / 10
    for start, end in sorted((s, e) for s, e, _ in dense_regions):
        while current < start:
            targets.append(current)
            current += sparse_spacing
        current = end + sparse_spacing
    while current <= np.floor(max_distance * 10) / 10:
        targets.append(current)
        current += sparse_spacing
    return [float(t) for t in sorted(set(np.round(targets, 4)))]


def match_frames(
    times: np.ndarray, distances: np.ndarray, targets: Sequence[float], limit: float = MAX_WINDOW_DISTANCE
) -> List[Tuple[float, float]]:
    selected: List[Tuple[float, float]] = []
    used = set()
    for target in targets:
        index = int(np.abs(distances - target).argmin())
        if index not in used:
            selected.append((target, float(times[index])))
            used.add(index)
    return [(d, t) for d, t in selected if d < limit]


def nearest_atom_id(positions: np.ndarray, ids: np.ndarray, reference: np.ndarray) -> int:
    return int(ids[np.linalg.norm(positions - reference, axis=1).argmin()])


class Stage:
    def __init__(self, config: StageConfig) -> None:
        self.config = config
        self.type = config.type
        self.options: Dict[str, object] = dict(config.options)
        self.umbrella = "US" in self.options
        self.directory: Optional[Path] = None
        self.mdp_path: Optional[Path] = None
        self.frames: List[Path] = []

    @property
    def mdp_overrides(self) -> Dict[str, object]:
        return {k: v for k, v in self.options.items() if k not in CONTROL_OPTIONS}

    def prepare(self, system) -> None:
        self.directory = system.config.output_dir / self.type
        if self.directory.exists():
            shutil.rmtree(self.directory)
        self.directory.mkdir(parents=True)
        if self.type == "CONFIG":
            return
        self.mdp_path = self._stage_mdp()

    def _stage_mdp(self) -> Path:
        if "mdp_fname" in self.options:
            source = Path(str(self.options["mdp_fname"])).resolve()
        else:
            source = resources.mdp_template(self.type)
        destination = self.directory / f"{self.type}.mdp"
        shutil.copy(source, destination)
        return destination

    def write_mdp(self, extra: Optional[Dict[str, object]] = None) -> None:
        mdp = MdpFile.read(self.mdp_path)
        mdp.update(self.mdp_overrides)
        if extra:
            mdp.update(extra)
        mdp.write(self.mdp_path)

    def pull_reference_atom(self, system) -> int:
        import MDAnalysis as mda

        if not system.state.frames:
            raise StageError(f"Stage {self.type} needs umbrella frames from a preceding CONFIG stage")
        universe = mda.Universe(str(system.state.frames[0]))
        members = read_atoms(system.index)["MM"]
        group = universe.atoms[members]
        return nearest_atom_id(group.positions, group.ids, group.center_of_mass())

    def run(self, system) -> None:
        if self.directory is None:
            raise StageError("Stage.prepare must be called before Stage.run")
        if self.type == "CONFIG":
            self._generate_frames(system)
        elif self.umbrella:
            self._run_umbrella(system)
        else:
            self._run_local(system)

    def _run_local(self, system) -> None:
        self.write_mdp()
        tpr = self.directory / f"{self.type}.tpr"
        structure = system.state.structure
        system.gmx.grompp(
            self.mdp_path, structure, system.topology, tpr, index=system.index, restraints=structure
        )
        self._mdrun_with_restarts(system)
        system.state.update(self.directory / f"{self.type}.gro", self.directory / f"{self.type}.xtc", tpr)

    def _mdrun_with_restarts(self, system) -> None:
        output = self.directory / f"{self.type}.gro"
        checkpoint = self.directory / f"{self.type}.cpt"
        options = dict(ntomp=system.config.ntomp, nb=system.config.nb, cwd=self.directory)
        system.gmx.mdrun(self.type, **options)
        for attempt in range(MAX_RESTARTS):
            if output.exists():
                return
            if not checkpoint.exists():
                break
            LOGGER.warning("%s did not finish; restart %d from checkpoint", self.type, attempt + 1)
            system.gmx.mdrun(self.type, checkpoint=checkpoint.name, **options)
        if not output.exists():
            raise StageError(f"mdrun for stage {self.type} did not produce {output.name}")

    def _generate_frames(self, system) -> None:
        state = system.state
        if state.trajectory is None or state.tpr is None:
            raise StageError("CONFIG needs the trajectory and run input from a preceding pulling stage")
        lipid, drug = self.options.get("lipid"), self.options.get("drug")
        if not lipid or not drug:
            raise StageError("CONFIG requires 'lipid' and 'drug' index group names")
        distances_file = self.directory / "dist.xvg"
        system.gmx.distance(state.trajectory, state.tpr, system.index, lipid, drug, distances_file)
        data, _ = read_xvg(distances_file)
        times, distances = data[:, 0], data[:, 1]
        targets = umbrella_targets(float(distances.min()), float(distances.max()))
        selected = match_frames(times, distances, targets)
        frames = [(time, self.directory / f"{dist}.gro") for dist, time in selected]
        for chunk in split_evenly(frames, system.slurm.config.frame_jobs):
            body = system.slurm.frame_extraction_script(
                state.trajectory, state.tpr, system.index, chunk, gmx=system.config.gmx_path
            )
            system.slurm.submit(body, "FRAMEGEN", self.directory, cpus=4, memory="12gb")
        paths = [path for _, path in frames]
        wait_for_files(paths, system.slurm.config.poll_seconds)
        self.frames = paths
        state.frames = paths

    def _run_umbrella(self, system) -> None:
        state = system.state
        if not state.frames:
            raise StageError(f"Stage {self.type} has no umbrella frames to work on")
        self.write_mdp({"pull_group2_pbcatom": self.pull_reference_atom(system)})
        cpu_only = self.type == "EM_US"
        for chunk in split_evenly(state.frames, system.slurm.config.umbrella_jobs):
            body = system.slurm.umbrella_script(
                self.mdp_path,
                system.topology,
                system.index,
                chunk,
                cpu_only,
                gmx=system.config.gmx_path,
                ntomp=system.config.ntomp,
            )
            system.slurm.submit(body, "SUBJOB", self.directory, gpu=not cpu_only)
        outputs = [self.directory / f"{frame.stem}.gro" for frame in state.frames]
        wait_for_files(outputs, system.slurm.config.poll_seconds)
        self.frames = outputs
        state.frames = outputs

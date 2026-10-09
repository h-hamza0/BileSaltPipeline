from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from bilesalt.backmapping import Backmapper
from bilesalt.box import BoxManager
from bilesalt.config import MoleculeConfig, StageConfig, SystemConfig
from bilesalt.gromacs import Gromacs
from bilesalt.molecules import MoleculeManager
from bilesalt.shell import LOGGER
from bilesalt.slurm import SlurmSubmitter
from bilesalt.stages import Stage
from bilesalt.topology import TopologyBuilder


@dataclass
class PipelineState:
    structure: Optional[Path] = None
    trajectory: Optional[Path] = None
    tpr: Optional[Path] = None
    frames: List[Path] = field(default_factory=list)

    def update(self, structure: Path, trajectory: Optional[Path] = None, tpr: Optional[Path] = None) -> None:
        self.structure = Path(structure).resolve()
        self.trajectory = None if trajectory is None else Path(trajectory).resolve()
        self.tpr = None if tpr is None else Path(tpr).resolve()


class System:
    def __init__(
        self, config: SystemConfig, state: PipelineState, slurm: SlurmSubmitter, gmx: Optional[Gromacs] = None
    ) -> None:
        self.config = config
        self.state = state
        self.slurm = slurm
        self.gmx = gmx or Gromacs(config.gmx_path)
        self.forcefield = config.forcefield
        self.topology: Optional[Path] = None
        self.index: Optional[Path] = None
        self.groups: Dict[str, int] = {}
        self.stages: List[Stage] = []
        self.molecules = MoleculeManager(self)
        self.box = BoxManager(self)
        self.topologies = TopologyBuilder(self)
        self.backmapper = Backmapper(
            config.cg2at_executable, config.gmx_path, config.cg2at_overlap, config.cg2at_answers
        )

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def directory(self) -> Path:
        return self.config.directory

    def add_molecule(self, config: MoleculeConfig) -> None:
        self.molecules.add(config)

    def add_stage(self, config: StageConfig) -> None:
        self.stages.append(Stage(config))

    def create_directories(self) -> None:
        root = self.directory
        if root.exists():
            LOGGER.warning("%s already exists and will be rebuilt from scratch", root)
            shutil.rmtree(root)
        setup = root / "setup"
        for sub in ("molecules", "itp", "forcefield"):
            (setup / sub).mkdir(parents=True)
        shutil.copytree(self.config.forcefield, setup / "forcefield", dirs_exist_ok=True)
        self.forcefield = setup / "forcefield"
        shutil.copy(self.config.input_dir / "water.gro", setup / "molecules" / "water.gro")
        if self.config.source is not None:
            self._import_source(setup)

    def _import_source(self, setup: Path) -> None:
        source = self.config.source
        structure = setup / "clust.gro"
        shutil.copy(source / "MD_AA" / "MD_AA.gro", structure)
        shutil.copy(source / "AA" / "setup" / "index.ndx", setup / "index.ndx")
        self.index = (setup / "index.ndx").resolve()
        self.state.structure = structure.resolve()

    def build_index(self) -> None:
        index = (self.directory / "setup" / "index.ndx").resolve()
        self.groups = self.gmx.make_index(self.state.structure, index, self.config.index_groups)
        self.index = index

    def refresh(self) -> None:
        self.build_index()
        self.box.count_components()
        self.topologies.write()

    def backmap(self) -> Path:
        final = self.backmapper.convert(self.state.structure, self.directory)
        self.state.structure = final
        return final

    def setup(self) -> None:
        config = self.config
        self.create_directories()
        self.molecules.manage()
        if config.phase == "CG":
            if config.resolvate:
                self.box.solvate_cg()
            self.backmap()
        elif config.resolvate:
            if config.source is not None:
                self.box.align()
                self.box.elongate()
            self.box.solvate_aa()
        self.refresh()
        self.box.ionize()
        self.refresh()

    def run_stages(self) -> None:
        for stage in self.stages:
            stage.prepare(self)
        for stage in self.stages:
            stage.run(self)

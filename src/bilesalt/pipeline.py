from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from bilesalt.config import PipelineConfig
from bilesalt.shell import LOGGER
from bilesalt.slurm import SlurmSubmitter
from bilesalt.system import PipelineState, System


class Pipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.state = PipelineState()
        self.slurm = SlurmSubmitter(config.slurm)
        self.systems: Dict[str, System] = {}
        for system_config in config.systems:
            system = System(system_config, self.state, self.slurm)
            for molecule in config.molecules_for(system_config.name):
                system.add_molecule(molecule)
            for stage in config.stages_for(system_config.name):
                system.add_stage(stage)
            self.systems[system_config.name] = system

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Pipeline:
        return cls(PipelineConfig.load(path))

    def describe(self) -> List[str]:
        lines = []
        for system in self.systems.values():
            cfg = system.config
            lines.append(f"system {cfg.name} [{cfg.phase}] -> {cfg.directory}")
            for molecule in system.molecules:
                lines.append(
                    f"  molecule {molecule.name} x{molecule.nmol}{' (insert)' if molecule.insert else ''}"
                )
            for stage in system.stages:
                lines.append(f"  stage {stage.type}{' (umbrella)' if stage.umbrella else ''}")
        return lines

    def run(self) -> PipelineState:
        for system in self.systems.values():
            LOGGER.info("Setting up system %s", system.name)
            system.setup()
            system.run_stages()
        return self.state

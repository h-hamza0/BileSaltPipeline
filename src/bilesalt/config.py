from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

PHASES = ("CG", "AA")
GROUP_PATTERN = re.compile(r"(\w+):(\[[^\]]*\])")


class ConfigError(ValueError):
    pass


def as_bool(value: Union[str, bool, None]) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    lowered = str(value).strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0", "none", ""}:
        return False
    raise ConfigError(f"Cannot interpret '{value}' as a boolean")


def as_optional_path(value: Optional[str]) -> Optional[Path]:
    if value is None or str(value).strip().lower() in {"none", ""}:
        return None
    return Path(value)


def as_optional_str(value: Optional[str]) -> Optional[str]:
    if value is None or str(value).strip().lower() in {"none", ""}:
        return None
    return str(value)


def parse_index_groups(text: str) -> Dict[str, List[str]]:
    groups = {name: list(ast.literal_eval(members)) for name, members in GROUP_PATTERN.findall(text)}
    if not groups:
        raise ConfigError(f"No index groups found in '{text}'")
    return groups


def parse_value(value: str) -> Union[str, List[str]]:
    parts = value.split()
    return parts if len(parts) > 1 else value


def split_blocks(text: str) -> List[Tuple[str, Dict[str, str]]]:
    blocks: List[Tuple[str, Dict[str, str]]] = []
    header: Optional[str] = None
    body: Dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if header is None:
            header, body = line, {}
        elif line == "end":
            if not body:
                raise ConfigError(f"Block '{header}' is empty (line {number})")
            blocks.append((header, body))
            header = None
        elif "=" in line:
            key, value = line.split("=", 1)
            body[key.strip()] = value.strip()
        else:
            raise ConfigError(f"Line {number}: expected 'key = value' inside block '{header}'")
    if header is not None:
        raise ConfigError(f"Block '{header}' is missing its 'end' line")
    return blocks


@dataclass
class SystemConfig:
    name: str
    input_dir: Path
    output_dir: Path
    forcefield: Path
    index_groups: Dict[str, List[str]]
    phase: str
    gmx_path: str = "gmx"
    box_size: Tuple[float, float, float] = (12.0, 12.0, 12.0)
    box_distance: float = 0.0
    water_radius: float = 0.21
    resolvate: bool = False
    source: Optional[Path] = None
    water_model: str = "spc216"
    ntomp: int = 12
    nb: str = "gpu"
    positive_ion: str = "NA"
    negative_ion: str = "CLA"
    cg2at_answers: Tuple[str, ...] = ("12", "10", "2")
    cg2at_overlap: float = 0.25
    cg2at_executable: str = "cg2at"
    peptide_residues: Tuple[str, ...] = ("ALA", "DPHE", "PHE", "LYS", "CYS", "THR", "THO", "DTRP")
    membrane_residues: Tuple[str, ...] = ("NATC", "NATD", "NATDC", "POPC")

    REQUIRED = ("name", "input_dir", "output_dir", "forcefield", "idxGRPS", "cg2at")

    @classmethod
    def from_block(cls, body: Dict[str, str]) -> SystemConfig:
        missing = [key for key in cls.REQUIRED if key not in body]
        if missing:
            raise ConfigError(f"System block is missing: {', '.join(missing)}")
        phase = body["cg2at"].strip().upper()
        if phase not in PHASES:
            raise ConfigError(f"cg2at must be one of {PHASES}, got '{body['cg2at']}'")
        size = body.get("boxSize", "12 12 12").split()
        if len(size) != 3:
            raise ConfigError("boxSize needs three values")
        extra = {}
        if "cg2at_answers" in body:
            extra["cg2at_answers"] = tuple(body["cg2at_answers"].split())
        if "peptide_residues" in body:
            extra["peptide_residues"] = tuple(body["peptide_residues"].split())
        if "membrane_residues" in body:
            extra["membrane_residues"] = tuple(body["membrane_residues"].split())
        return cls(
            name=body["name"],
            input_dir=Path(body["input_dir"]),
            output_dir=Path(body["output_dir"]),
            forcefield=Path(body["forcefield"]),
            index_groups=parse_index_groups(body["idxGRPS"]),
            phase=phase,
            gmx_path=body.get("gmx_path", "gmx"),
            box_size=(float(size[0]), float(size[1]), float(size[2])),
            box_distance=float(body.get("boxDist", 0)),
            water_radius=float(body.get("Wradius", 0.21)),
            resolvate=as_bool(body.get("resolvate")),
            source=as_optional_path(body.get("US")),
            water_model=body.get("water_model", "spc216"),
            ntomp=int(body.get("ntomp", 12)),
            nb=body.get("nb", "gpu"),
            positive_ion=body.get("positive_ion", "NA"),
            negative_ion=body.get("negative_ion", "CLA"),
            cg2at_overlap=float(body.get("cg2at_overlap", 0.25)),
            cg2at_executable=body.get("cg2at_executable", "cg2at"),
            **extra,
        )

    @property
    def directory(self) -> Path:
        return self.output_dir / self.name


@dataclass
class MoleculeConfig:
    system: str
    name: str
    nmol: int
    insertion_radius: float
    insert: bool = False
    packmol: bool = False
    cgat: str = "False"
    itp: bool = False
    replace: Optional[str] = None

    REQUIRED = ("system", "name", "nmol", "insertion_radius")

    @classmethod
    def from_block(cls, body: Dict[str, str]) -> MoleculeConfig:
        missing = [key for key in cls.REQUIRED if key not in body]
        if missing:
            raise ConfigError(f"Molecule block is missing: {', '.join(missing)}")
        return cls(
            system=body["system"],
            name=body["name"],
            nmol=int(body["nmol"]),
            insertion_radius=float(body["insertion_radius"]),
            insert=as_bool(body.get("insert")),
            packmol=as_bool(body.get("packmol")),
            cgat=body.get("CGAT", "False"),
            itp=as_bool(body.get("itp")),
            replace=as_optional_str(body.get("replace")),
        )


@dataclass
class StageConfig:
    system: str
    type: str
    options: Dict[str, Union[str, List[str]]] = field(default_factory=dict)

    @classmethod
    def from_block(cls, body: Dict[str, str]) -> StageConfig:
        for key in ("system", "type"):
            if key not in body:
                raise ConfigError(f"Stage block is missing: {key}")
        options = {k: parse_value(v) for k, v in body.items() if k not in {"system", "type"}}
        return cls(system=body["system"], type=body["type"], options=options)


@dataclass
class SlurmConfig:
    cpu_partition: Optional[str] = None
    gpu_partition: Optional[str] = None
    mail_user: Optional[str] = None
    gmxrc: Optional[str] = None
    modules_purge: bool = True
    cpus: int = 12
    memory: str = "32gb"
    frame_jobs: int = 5
    umbrella_jobs: int = 15
    poll_seconds: float = 30.0

    @classmethod
    def from_block(cls, body: Dict[str, str]) -> SlurmConfig:
        known = {
            "cpu_partition": str,
            "gpu_partition": str,
            "mail_user": str,
            "gmxrc": str,
            "memory": str,
            "cpus": int,
            "frame_jobs": int,
            "umbrella_jobs": int,
            "poll_seconds": float,
        }
        unknown = set(body) - set(known) - {"modules_purge"}
        if unknown:
            raise ConfigError(f"Unknown slurm options: {', '.join(sorted(unknown))}")
        values = {key: known[key](value) for key, value in body.items() if key in known}
        if "modules_purge" in body:
            values["modules_purge"] = as_bool(body["modules_purge"])
        return cls(**values)


@dataclass
class PipelineConfig:
    systems: List[SystemConfig] = field(default_factory=list)
    molecules: List[MoleculeConfig] = field(default_factory=list)
    stages: List[StageConfig] = field(default_factory=list)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    def system(self, name: str) -> SystemConfig:
        for system in self.systems:
            if system.name == name:
                return system
        raise ConfigError(f"Unknown system '{name}'")

    def molecules_for(self, name: str) -> List[MoleculeConfig]:
        return [m for m in self.molecules if m.system == name]

    def stages_for(self, name: str) -> List[StageConfig]:
        return [s for s in self.stages if s.system == name]

    def validate(self) -> None:
        if not self.systems:
            raise ConfigError("No 'sys' block found")
        names = [s.name for s in self.systems]
        if len(set(names)) != len(names):
            raise ConfigError("System names must be unique")
        for item in [*self.molecules, *self.stages]:
            if item.system not in names:
                label = getattr(item, "name", None) or getattr(item, "type", "?")
                raise ConfigError(f"'{label}' refers to unknown system '{item.system}'")

    @classmethod
    def parse(cls, text: str) -> PipelineConfig:
        config = cls()
        for header, body in split_blocks(text):
            kind = header.split("_")[0]
            if kind == "sys":
                config.systems.append(SystemConfig.from_block(body))
            elif kind == "molecule":
                config.molecules.append(MoleculeConfig.from_block(body))
            elif kind == "stage":
                config.stages.append(StageConfig.from_block(body))
            elif kind == "slurm":
                config.slurm = SlurmConfig.from_block(body)
            else:
                raise ConfigError(f"Unknown block '{header}'")
        config.validate()
        return config

    @classmethod
    def load(cls, path: Union[str, Path]) -> PipelineConfig:
        return cls.parse(Path(path).read_text())

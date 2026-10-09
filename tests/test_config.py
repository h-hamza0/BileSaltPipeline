from __future__ import annotations

from pathlib import Path

import pytest

from bilesalt.config import ConfigError, PipelineConfig, as_bool, parse_index_groups

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "jobs"

MINIMAL = """\
sys
name = AA
input_dir = in
output_dir = out
forcefield = ff
idxGRPS = BIL:['A','B'] SOLV:['SOL']
cg2at = AA
end

molecule_1
system = AA
name = A
nmol = 4
insertion_radius = 0.25
insert = True
replace = SOL
end

stage
system = AA
type = EM_AA
nsteps = 10
ref_t = 298 298
end
"""


def test_parse_index_groups() -> None:
    assert parse_index_groups("BIL:['NATC','POPC'] SOLV:['SOL','NA']") == {
        "BIL": ["NATC", "POPC"],
        "SOLV": ["SOL", "NA"],
    }


@pytest.mark.parametrize("value,expected", [("True", True), ("false", False), ("None", False), (None, False)])
def test_as_bool(value, expected) -> None:
    assert as_bool(value) is expected


def test_as_bool_rejects_garbage() -> None:
    with pytest.raises(ConfigError):
        as_bool("maybe")


def test_minimal_pipeline_parses() -> None:
    config = PipelineConfig.parse(MINIMAL)
    system = config.system("AA")
    assert system.phase == "AA" and system.source is None and system.box_size == (12.0, 12.0, 12.0)
    molecule = config.molecules_for("AA")[0]
    assert molecule.nmol == 4 and molecule.insert and molecule.replace == "SOL"
    stage = config.stages_for("AA")[0]
    assert stage.options == {"nsteps": "10", "ref_t": ["298", "298"]}


def test_missing_required_system_key() -> None:
    with pytest.raises(ConfigError, match="missing"):
        PipelineConfig.parse(MINIMAL.replace("forcefield = ff\n", ""))


def test_invalid_phase() -> None:
    with pytest.raises(ConfigError, match="cg2at"):
        PipelineConfig.parse(MINIMAL.replace("cg2at = AA", "cg2at = XX"))


def test_unterminated_block() -> None:
    with pytest.raises(ConfigError, match="end"):
        PipelineConfig.parse(MINIMAL.rsplit("end", 1)[0])


def test_molecule_referencing_unknown_system() -> None:
    with pytest.raises(ConfigError, match="unknown system"):
        PipelineConfig.parse(MINIMAL.replace("system = AA\nname = A", "system = ZZ\nname = A"))


def test_slurm_block_and_unknown_option() -> None:
    config = PipelineConfig.parse(MINIMAL + "\nslurm\ncpu_partition = batch\ncpus = 4\nend\n")
    assert config.slurm.cpu_partition == "batch" and config.slurm.cpus == 4
    with pytest.raises(ConfigError, match="Unknown slurm"):
        PipelineConfig.parse(MINIMAL + "\nslurm\nbogus = 1\nend\n")


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.txt")))
def test_shipped_examples_parse(path: Path) -> None:
    config = PipelineConfig.load(path)
    assert config.systems and config.stages

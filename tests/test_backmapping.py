from __future__ import annotations

import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from bilesalt.backmapping import Backmapper, BackmappingError
from bilesalt.pipeline import Pipeline

FAKE_CG2AT = """#!/bin/bash
echo "$@" > args.txt
cat > answers.txt
while [ $# -gt 0 ]; do
  case "$1" in
    -c) structure="$2";;
    -loc) location="$2";;
    -gmx) gmx="$2";;
  esac
  shift
done
mkdir -p "$location/FINAL"
"$gmx" editconf -f "$structure" -o "$location/FINAL/final_cg2at_de_novo.pdb" > /dev/null 2>&1
"""


@pytest.fixture
def fake_cg2at(tmp_path: Path) -> Path:
    path = tmp_path / "bin" / "cg2at"
    path.parent.mkdir()
    path.write_text(FAKE_CG2AT)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def test_backmapper_builds_expected_command(tmp_path: Path) -> None:
    backmapper = Backmapper("cg2at", "gmx", 0.3)
    structure = tmp_path / "cg.gro"
    command = backmapper.command(structure, "backmap")
    assert command[0] == "cg2at"
    assert command[command.index("-ov") + 1] == 0.3
    assert command[command.index("-loc") + 1] == "backmap"


def test_backmapper_rejects_missing_structure(tmp_path: Path) -> None:
    with pytest.raises(BackmappingError):
        Backmapper().convert(tmp_path / "missing.gro", tmp_path)


def test_backmapper_reports_missing_output(tmp_path: Path) -> None:
    script = tmp_path / "cg2at"
    script.write_text("#!/bin/bash\ncat > /dev/null\n")
    script.chmod(0o755)
    structure = tmp_path / "cg.gro"
    structure.write_text("x")
    with pytest.raises(BackmappingError, match="without producing"):
        Backmapper(str(script)).convert(structure, tmp_path / "work")


@pytest.mark.gromacs
def test_backmapper_forwards_answers_and_returns_final_structure(
    tmp_path, gmx_top, gmx_executable, fake_cg2at
):
    structure = tmp_path / "cg.gro"
    shutil.copy(gmx_top / "spc216.gro", structure)
    backmapper = Backmapper(str(fake_cg2at), gmx_executable, 0.25, ("12", "10", "2"))
    final = backmapper.convert(structure, tmp_path / "work")
    assert final.name == "final_cg2at_de_novo.pdb" and final.exists()
    assert (tmp_path / "work" / "backmap" / "answers.txt").exists() is False
    answers = (tmp_path / "work" / "answers.txt").read_text().split()
    assert answers == ["12", "10", "2"]


@pytest.mark.gromacs
def test_cg_system_backmaps_into_atomistic_topology(tmp_path, gmx_top, gmx_executable, fake_cg2at):
    inp = tmp_path / "input"
    inp.mkdir()
    shutil.copytree(gmx_top / "oplsaa.ff", inp / "oplsaa.ff")
    shutil.copy(gmx_top / "spc216.gro", inp / "water.gro")
    source = tmp_path / "previous"
    (source / "MD_AA").mkdir(parents=True)
    (source / "AA" / "setup").mkdir(parents=True)
    subprocess.run(
        [
            gmx_executable,
            "solvate",
            "-cs",
            "spc216.gro",
            "-box",
            "3",
            "3",
            "3",
            "-o",
            str(source / "MD_AA" / "MD_AA.gro"),
        ],
        check=True,
        capture_output=True,
    )
    (source / "AA" / "setup" / "index.ndx").write_text("[ System ]\n1\n")
    job = tmp_path / "job.txt"
    job.write_text(f"""\
sys
name = CG
gmx_path = {gmx_executable}
cg2at_executable = {fake_cg2at}
input_dir = {inp}
output_dir = {tmp_path / "out"}
forcefield = {inp}/oplsaa.ff
idxGRPS = SOLV:['SOL']
cg2at = CG
resolvate = False
US = {source}
negative_ion = CL
end

molecule_1
system = CG
name = SOL
nmol = 0
insertion_radius = 0.105
end

molecule_2
system = CG
name = NA
nmol = 0
insertion_radius = 0.25
end

molecule_3
system = CG
name = CL
nmol = 0
insertion_radius = 0.25
end
""")
    state = Pipeline.from_file(job).run()
    out = tmp_path / "out" / "CG"
    assert (out / "backmap" / "FINAL" / "final_cg2at_de_novo.pdb").exists()
    assert state.structure == (out / "setup" / "ionized.gro").resolve()
    topology = (out / "setup" / "topol.top").read_text()
    assert "SOL   " in topology and "SOL   0" not in topology

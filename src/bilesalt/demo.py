from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from bilesalt.pipeline import Pipeline
from bilesalt.system import PipelineState

SODIUM_GRO = (
    "NA\n    1\n"
    + "{:5d}{:<5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n".format(1, "NA", "NA", 1, 0.0, 0.0, 0.0)
    + "   1.00000   1.00000   1.00000\n"
)

JOB_TEMPLATE = """\
sys
name = AA
gmx_path = {gmx}
input_dir = {inp}
output_dir = {out}
forcefield = {inp}/oplsaa.ff
idxGRPS = SOLV:['SOL','NA','CL'] MM:['NA']
boxSize = 3 3 3
boxDist = 0
cg2at = AA
resolvate = True
ntomp = 2
nb = cpu
negative_ion = CL
end

molecule_1
system = AA
name = NA
nmol = 3
insertion_radius = 0.25
insert = True
itp = False
end

molecule_2
system = AA
name = SOL
nmol = 0
insertion_radius = 0.105
end

molecule_3
system = AA
name = CL
nmol = 0
insertion_radius = 0.25
end

stage
system = AA
type = EM_AA
nsteps = 20
end

stage
system = AA
type = NVT_AA
tc-grps = System
tau_t = 0.1
ref_t = 300
gen_temp = 300
nsteps = 10
end
"""


class DemoError(RuntimeError):
    pass


def gromacs_topology_dir(gmx: str) -> Path:
    executable = shutil.which(gmx) or gmx
    try:
        result = subprocess.run([executable, "--version"], capture_output=True, text=True)
    except OSError as exc:
        raise DemoError(f"Cannot execute '{gmx}'") from exc
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith("Data prefix:"):
            top = Path(line.split(":", 1)[1].strip()) / "share" / "gromacs" / "top"
            if (top / "oplsaa.ff").is_dir():
                return top
    raise DemoError("Could not locate the GROMACS force-field library")


def prepare(workdir: Path, gmx: str = "gmx") -> Path:
    top = gromacs_topology_dir(gmx)
    workdir = Path(workdir).resolve()
    inp = workdir / "input"
    inp.mkdir(parents=True, exist_ok=True)
    if not (inp / "oplsaa.ff").exists():
        shutil.copytree(top / "oplsaa.ff", inp / "oplsaa.ff")
    shutil.copy(top / "spc216.gro", inp / "water.gro")
    (inp / "NA.gro").write_text(SODIUM_GRO)
    job = workdir / "demo_job.txt"
    job.write_text(JOB_TEMPLATE.format(gmx=gmx, inp=inp, out=workdir / "output"))
    return job


def run(workdir: Path, gmx: str = "gmx") -> Optional[PipelineState]:
    return Pipeline.from_file(prepare(workdir, gmx)).run()

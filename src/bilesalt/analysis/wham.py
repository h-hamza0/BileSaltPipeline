from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from bilesalt import shell
from bilesalt.slurm import SlurmSubmitter


def highest_window_index(directory: Path) -> Optional[int]:
    best: Optional[int] = None
    for path in directory.glob("*pullf.xvg"):
        numbers = re.findall(r"\d+", path.stem)
        if numbers:
            value = int(numbers[-1])
            best = value if best is None else max(best, value)
    return best


def find_windows(root: Path) -> List[Path]:
    return sorted(root.glob("US_*/PULL_US"))


@dataclass
class WhamJob:
    directory: Path
    cpus: int = 8
    memory: str = "16G"
    bootstraps: int = 200
    gmx: str = "gmx"

    @property
    def label(self) -> str:
        return f"{self.directory.parent.parent.name}_{self.directory.parent.name}"

    def zprof0(self) -> str:
        known = highest_window_index(self.directory)
        if known is not None:
            return f"ZPROF0={known}"
        return (
            "LAST_PULLF=$(ls *pullf.xvg | sort -V | tail -1)\n"
            "ZPROF0=$(echo \"$LAST_PULLF\" | grep -oE '[0-9]+' | tail -1)"
        )

    def script(self) -> str:
        return (
            "set -euo pipefail\n"
            f"cd {self.directory.resolve()}\n"
            "ls *.tpr | sort -V > tpr-files.dat\n"
            "ls *pullf.xvg | sort -V > pullf-files.dat\n"
            f"export OMP_NUM_THREADS={self.cpus}\n"
            f"{self.zprof0()}\n"
            'if [ -z "$ZPROF0" ]; then echo "could not determine zprof0" >&2; exit 1; fi\n'
            f"{self.gmx} wham -it tpr-files.dat -if pullf-files.dat -o -hist -unit kCal "
            f"-zprof0 $ZPROF0 -bsres profile.xvg -nBootstrap {self.bootstraps} -bs-method traj -ac\n"
        )


def submit_all(root: Path, submitter: SlurmSubmitter, dry_run: bool = False, **job_options) -> List[str]:
    directories = find_windows(root)
    if not directories:
        raise FileNotFoundError(f"No US_*/PULL_US directories under {root}")
    submitted = []
    for directory in directories:
        job = WhamJob(directory, **job_options)
        if dry_run:
            submitted.append(f"# {job.label}\n{job.script()}")
        else:
            shell.LOGGER.info("Submitting WHAM for %s", directory)
            submitter.submit(job.script(), job.label, directory, cpus=job.cpus, memory=job.memory)
            submitted.append(job.label)
    return submitted

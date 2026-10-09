from __future__ import annotations

import re
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Callable, List, Optional

from bilesalt import shell
from bilesalt.config import SlurmConfig


class JobTimeout(TimeoutError):
    pass


def split_evenly(items: Sequence, parts: int) -> List[list]:
    if parts < 1:
        raise ValueError("parts must be positive")
    size, extra = divmod(len(items), parts)
    chunks = []
    start = 0
    for index in range(parts):
        stop = start + size + (1 if index < extra else 0)
        chunks.append(list(items[start:stop]))
        start = stop
    return [chunk for chunk in chunks if chunk]


def wait_for_files(
    paths: Iterable[Path],
    poll_seconds: float = 30.0,
    timeout: Optional[float] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    pending = [Path(p) for p in paths]
    waited = 0.0
    while True:
        pending = [p for p in pending if not p.exists()]
        if not pending:
            return
        if timeout is not None and waited >= timeout:
            raise JobTimeout(f"{len(pending)} expected files were never written, e.g. {pending[0]}")
        sleep(poll_seconds)
        waited += poll_seconds


class SlurmSubmitter:
    def __init__(self, config: Optional[SlurmConfig] = None) -> None:
        self.config = config or SlurmConfig()

    def preamble(self) -> str:
        lines = ["#!/bin/bash"]
        if self.config.modules_purge:
            lines.append("module purge")
        if self.config.gmxrc:
            lines.append(f"source {self.config.gmxrc}")
        return "\n".join(lines) + "\n"

    def header(self, name: str, cpus: int, memory: str, gpu: bool) -> List[str]:
        partition = self.config.gpu_partition if gpu else self.config.cpu_partition
        args = [
            f"--job-name={name}",
            "--output=output_%j.log",
            "--ntasks=1",
            f"--cpus-per-task={cpus}",
            f"--mem={memory}",
        ]
        if gpu:
            args += ["--gpus-per-node=1", "--gres=gpu:1"]
        if partition:
            args.append(f"--partition={partition}")
        if self.config.mail_user:
            args += [f"--mail-user={self.config.mail_user}", "--mail-type=END,FAIL"]
        return args

    def submit(
        self,
        body: str,
        name: str,
        cwd: Path,
        cpus: Optional[int] = None,
        memory: Optional[str] = None,
        gpu: bool = False,
    ) -> Optional[str]:
        script = self.preamble() + body
        args = self.header(name, cpus or self.config.cpus, memory or self.config.memory, gpu)
        result = shell.run(["sbatch", *args], stdin=script, cwd=cwd)
        match = re.search(r"Submitted batch job (\d+)", result.stdout or "")
        return match.group(1) if match else None

    def frame_extraction_script(
        self, trajectory: Path, tpr: Path, index: Path, frames: Sequence[tuple], gmx: str = "gmx"
    ) -> str:
        outputs = " ".join(f'"{path}"' for _, path in frames)
        times = " ".join(str(time_) for time_, _ in frames)
        return (
            f"outputs=({outputs})\ntimes=({times})\n"
            'for i in "${!outputs[@]}"; do\n'
            f'    echo "0" | {gmx} trjconv -s {tpr} -f {trajectory} -dump "${{times[$i]}}" '
            f'-o "${{outputs[$i]}}" -n {index}\n'
            "done\n"
        )

    def umbrella_script(
        self,
        mdp: Path,
        topology: Path,
        index: Path,
        frames: Sequence[Path],
        cpu_only: bool,
        gmx: str = "gmx",
        ntomp: int = 12,
    ) -> str:
        quoted = " ".join(f'"{f.resolve()}"' for f in frames)
        mdrun = (
            f'srun {gmx} mdrun -deffnm "$base" -nb cpu -ntomp {ntomp} -pin off -v'
            if cpu_only
            else f'srun {gmx} mdrun -deffnm "$base" -ntmpi 1 -ntomp {ntomp} -nb gpu -pin off -v -bonded gpu'
        )
        return (
            "export GMX_GPU_DD_COMMS=true\nexport GMX_GPU_PME_PP_COMMS=true\n"
            "export GMX_FORCE_UPDATE_DEFAULT_GPU=true\n"
            f"export OMP_NUM_THREADS={ntomp}\n\n"
            f"frames=({quoted})\n"
            'for f in "${frames[@]}"; do\n'
            '    base=$(basename "$f" .gro)\n'
            f'    srun {gmx} grompp -f {mdp} -c "$f" -r "$f" -p {topology} -n {index} '
            '-o "${base}.tpr" -maxwarn 10\n'
            f"    {mdrun}\n"
            "done\n"
        )

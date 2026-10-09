from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

GMX = shutil.which("gmx") or str(Path.home() / "gromacs-opencl-install" / "bin" / "gmx")


def gromacs_data() -> Path:
    if not Path(GMX).exists():
        return Path()
    result = subprocess.run([GMX, "--version"], capture_output=True, text=True)
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith("Data prefix:"):
            return Path(line.split(":", 1)[1].strip()) / "share" / "gromacs" / "top"
    return Path()


@pytest.fixture(scope="session")
def gmx_top() -> Path:
    top = gromacs_data()
    if not (top / "oplsaa.ff").is_dir():
        pytest.skip("GROMACS with bundled force fields is not available")
    return top


@pytest.fixture
def gmx_executable() -> str:
    return GMX

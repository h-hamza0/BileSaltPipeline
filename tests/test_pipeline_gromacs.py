from __future__ import annotations

from pathlib import Path

import pytest

from bilesalt import demo
from bilesalt.pipeline import Pipeline

pytestmark = pytest.mark.gromacs


@pytest.fixture
def workspace(tmp_path: Path, gmx_top: Path, gmx_executable: str) -> Path:
    return demo.prepare(tmp_path, gmx_executable)


def test_validate_reports_plan(workspace: Path) -> None:
    lines = Pipeline.from_file(workspace).describe()
    assert lines[0].startswith("system AA [AA]")
    assert any("stage EM_AA" in line for line in lines)


def test_aa_system_builds_and_runs_stages(workspace: Path) -> None:
    state = Pipeline.from_file(workspace).run()
    out = workspace.parent / "output"
    topology = (out / "AA" / "setup" / "topol.top").read_text()
    assert "[ molecules ]" in topology and "NA   " in topology
    assert (out / "AA" / "setup" / "index.ndx").exists()
    assert (out / "EM_AA" / "EM_AA.gro").exists()
    assert (out / "NVT_AA" / "NVT_AA.gro").exists()
    assert state.structure == (out / "NVT_AA" / "NVT_AA.gro").resolve()
    assert state.tpr == (out / "NVT_AA" / "NVT_AA.tpr").resolve()

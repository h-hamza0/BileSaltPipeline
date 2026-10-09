from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bilesalt import indexfile
from bilesalt.analysis import COMMANDS, AnalysisPlan
from bilesalt.analysis.commands import connected_clusters
from bilesalt.analysis.wham import WhamJob, find_windows, highest_window_index
from bilesalt.config import ConfigError
from bilesalt.xvg import read_xvg

PLAN = """\
system
resultFolder = /data/run
end

stage_1
name = EM_AA
analysis = potential
end

stage_2
name = MD_AA
analysis = clusOverTime extractLargest
extractLargest = 25
clusOverTime = CMPL CMPL
end
"""


def test_plan_parsing() -> None:
    plan = AnalysisPlan.parse(PLAN)
    assert plan.result_folder == Path("/data/run")
    assert [s.name for s in plan.steps] == ["EM_AA", "MD_AA"]
    assert plan.steps[1].commands == ["clusOverTime", "extractLargest"]
    assert plan.steps[1].options["clusOverTime"] == ["CMPL", "CMPL"]


def test_unknown_command_is_rejected() -> None:
    plan = AnalysisPlan.parse(PLAN.replace("analysis = potential", "analysis = nonsense"))
    with pytest.raises(ConfigError, match="nonsense"):
        _ = plan.steps[0].commands


def test_plan_requires_result_folder() -> None:
    with pytest.raises(ConfigError):
        AnalysisPlan.parse("stage_1\nname = X\nanalysis = potential\nend\n")


def test_every_documented_command_is_registered() -> None:
    expected = {
        "potential",
        "temperature",
        "pressure",
        "gyrate",
        "density",
        "sasa",
        "aspectRatio",
        "cleanUp",
        "prettyView",
        "clusOverTime",
        "pbcNOJUMP",
        "radial",
        "extractLargest",
        "HBA_RES",
        "SASA_RES",
    }
    assert expected <= set(COMMANDS)


def test_connected_clusters() -> None:
    adjacency = [[1], [0, 2], [1], [4], [3], []]
    clusters = sorted(sorted(c) for c in connected_clusters(adjacency))
    assert clusters == [[0, 1, 2], [3, 4], [5]]


def test_index_parsing(tmp_path: Path) -> None:
    ndx = tmp_path / "i.ndx"
    ndx.write_text("[ System ]\n   1    2    3\n[ MM ]\n   2    3\n")
    assert indexfile.read_groups(ndx) == {"System": 0, "MM": 1}
    assert indexfile.read_atoms(ndx)["MM"] == [1, 2]
    with pytest.raises(KeyError, match="Water"):
        indexfile.group_number(indexfile.read_groups(ndx), "Water")


def test_read_xvg_extracts_axes(tmp_path: Path) -> None:
    path = tmp_path / "a.xvg"
    path.write_text(
        '# c\n@ title "Potential"\n@ xaxis  label "Time (ps)"\n@ yaxis  label "E"\n0 1.5\n1 2.5\n'
    )
    data, attrs = read_xvg(path)
    assert data.shape == (2, 2) and attrs["xaxis"] == "Time (ps)" and attrs["title"] == "Potential"


def test_wham_job_script_uses_highest_window(tmp_path: Path) -> None:
    window = tmp_path / "s" / "US_1" / "PULL_US"
    window.mkdir(parents=True)
    for n in (2, 10, 7):
        (window / f"{n}_pullf.xvg").write_text("")
    assert find_windows(tmp_path / "s") == [window]
    assert highest_window_index(window) == 10
    job = WhamJob(window, gmx="gmx")
    script = job.script()
    assert "ZPROF0=10" in script and "gmx wham" in script and job.label == "s_US_1"


@pytest.mark.gromacs
def test_energy_extraction_and_plot_from_real_run(tmp_path: Path, gmx_top, gmx_executable) -> None:
    from bilesalt import demo
    from bilesalt.gromacs import Gromacs
    from bilesalt.pipeline import Pipeline

    Pipeline.from_file(demo.prepare(tmp_path, gmx_executable)).run()
    plan = AnalysisPlan.parse(
        f"system\nresultFolder = {tmp_path / 'output'}\nend\n\n"
        "stage_1\nname = EM_AA\nanalysis = potential\nend\n\n"
        "stage_2\nname = NVT_AA\nanalysis = temperature\nend\n"
    )
    results = plan.run(Gromacs(gmx_executable))
    potential = results["EM_AA"]["potential"][0]["data"]
    assert potential.shape[1] == 2 and np.isfinite(potential).all()
    assert (tmp_path / "output" / "EM_AA" / "potential.png").exists()
    assert (tmp_path / "output" / "collection.pkl").exists()

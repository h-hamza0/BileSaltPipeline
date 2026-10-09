from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bilesalt.config import SlurmConfig, StageConfig
from bilesalt.slurm import JobTimeout, SlurmSubmitter, split_evenly, wait_for_files
from bilesalt.stages import Stage, StageError, match_frames, nearest_atom_id, umbrella_targets
from bilesalt.system import PipelineState


def test_umbrella_targets_cover_range_at_regular_spacing() -> None:
    targets = umbrella_targets(0.52, 5.04)
    assert targets[0] == pytest.approx(0.6) and targets[-1] == pytest.approx(5.0)
    assert np.allclose(np.diff(targets), 0.1, atol=1e-3)


def test_match_frames_skips_duplicates_and_far_windows() -> None:
    times = np.array([0.0, 10.0, 20.0])
    distances = np.array([1.0, 1.04, 12.0])
    selected = match_frames(times, distances, [1.0, 0.9, 12.0])
    assert selected == [(1.0, 0.0)]


def test_nearest_atom_id() -> None:
    positions = np.array([[0.0, 0, 0], [5.0, 0, 0], [9.0, 0, 0]])
    assert nearest_atom_id(positions, np.array([10, 20, 30]), np.array([5.5, 0, 0])) == 20


def test_split_evenly_balances_and_drops_empty() -> None:
    assert [len(c) for c in split_evenly(list(range(7)), 3)] == [3, 2, 2]
    assert len(split_evenly([1, 2], 5)) == 2


def test_wait_for_files_polls_until_present(tmp_path: Path) -> None:
    target = tmp_path / "x.gro"
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        target.write_text("")

    wait_for_files([target], poll_seconds=7, sleep=sleep)
    assert calls == [7]


def test_wait_for_files_times_out(tmp_path: Path) -> None:
    with pytest.raises(JobTimeout):
        wait_for_files([tmp_path / "never"], poll_seconds=1, timeout=2, sleep=lambda s: None)


def test_slurm_header_reflects_configuration() -> None:
    submitter = SlurmSubmitter(SlurmConfig(gpu_partition="gpu", mail_user="a@b.c"))
    header = submitter.header("JOB", 12, "32gb", gpu=True)
    assert "--partition=gpu" in header and "--gres=gpu:1" in header and "--mail-user=a@b.c" in header
    assert not any(h.startswith("--partition") for h in SlurmSubmitter().header("J", 1, "1gb", gpu=False))


def test_umbrella_script_selects_cpu_or_gpu_mdrun(tmp_path: Path) -> None:
    submitter = SlurmSubmitter()
    frames = [tmp_path / "2.1.gro", tmp_path / "2.2.gro"]
    cpu = submitter.umbrella_script(tmp_path / "m.mdp", tmp_path / "t.top", tmp_path / "i.ndx", frames, True)
    gpu = submitter.umbrella_script(tmp_path / "m.mdp", tmp_path / "t.top", tmp_path / "i.ndx", frames, False)
    assert "-nb cpu" in cpu and "-nb gpu" in gpu and "2.1.gro" in cpu


def test_control_options_are_not_written_to_mdp(tmp_path: Path) -> None:
    stage = Stage(StageConfig("AA", "PULL_US", {"US": "True", "nsteps": "5", "lipid": "A"}))
    assert stage.umbrella and stage.mdp_overrides == {"nsteps": "5"}


def test_config_stage_extracts_frames(tmp_path: Path) -> None:
    stage = Stage(StageConfig("AA", "CONFIG", {"US": "True", "lipid": "MM", "drug": "OCT"}))
    submitted = []

    class FakeGmx:
        def distance(self, trajectory, tpr, index, lipid, drug, output):
            rows = "".join(f"{t} {d}\n" for t, d in zip(range(0, 100, 10), np.linspace(0.5, 1.4, 10)))
            Path(output).write_text("@ title\n" + rows)

    class FakeSlurm(SlurmSubmitter):
        def submit(self, body, name, cwd, **kwargs):
            submitted.append(body)
            for line in body.splitlines():
                if line.startswith("outputs=("):
                    for token in line[len("outputs=(") : -1].split():
                        Path(token.strip('"')).write_text("")

    system = SimpleNamespace(
        config=SimpleNamespace(output_dir=tmp_path, gmx_path="gmx"),
        gmx=FakeGmx(),
        slurm=FakeSlurm(SlurmConfig(poll_seconds=0)),
        index=tmp_path / "i.ndx",
        state=PipelineState(trajectory=tmp_path / "t.xtc", tpr=tmp_path / "t.tpr"),
    )
    stage.prepare(system)
    stage.run(system)
    assert submitted and len(system.state.frames) >= 8
    assert all(p.exists() for p in system.state.frames)


def test_config_stage_requires_groups(tmp_path: Path) -> None:
    stage = Stage(StageConfig("AA", "CONFIG", {"US": "True"}))
    system = SimpleNamespace(
        config=SimpleNamespace(output_dir=tmp_path), state=PipelineState(trajectory=tmp_path, tpr=tmp_path)
    )
    stage.prepare(system)
    with pytest.raises(StageError, match="lipid"):
        stage.run(system)

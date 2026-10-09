from __future__ import annotations

import numpy as np
import pytest

from bilesalt.box import rotation_to_axis
from bilesalt.cli import main
from bilesalt.config import MoleculeConfig
from bilesalt.logp import pmf_convergence as pmf
from bilesalt.molecules import Molecule
from bilesalt.topology import TopologyBuilder


def test_rotation_aligns_vector_with_z() -> None:
    vector = np.array([1.0, 2.0, -0.5])
    angle, axis = rotation_to_axis(vector, np.array([0.0, 0.0, 1.0]))
    theta = np.radians(angle)
    k = axis / np.linalg.norm(axis)
    unit = vector / np.linalg.norm(vector)
    rotated = (
        unit * np.cos(theta) + np.cross(k, unit) * np.sin(theta) + k * np.dot(k, unit) * (1 - np.cos(theta))
    )
    assert rotated == pytest.approx([0, 0, 1], abs=1e-9)


def test_rotation_noop_when_already_aligned() -> None:
    angle, _ = rotation_to_axis(np.array([0.0, 0.0, 3.0]), np.array([0.0, 0.0, 1.0]))
    assert angle == 0.0


def test_topology_lists_includes_and_molecule_counts(tmp_path) -> None:
    ff = tmp_path / "setup" / "forcefield"
    ff.mkdir(parents=True)
    for name in ("forcefield.itp", "ions.itp", "tip3p.itp", "NATC_ffbonded.itp", "other.itp"):
        (ff / name).write_text("")
    itp = tmp_path / "NATC.itp"
    itp.write_text("")

    class Fake:
        directory = tmp_path
        forcefield = ff
        topology = None
        molecules = []

    system = Fake()
    mol = Molecule(MoleculeConfig("AA", "NATC", 5, 1.0, itp=True))
    mol.itp_file = itp
    system.molecules = [mol, Molecule(MoleculeConfig("AA", "SOL", 0, 0.1))]
    text = TopologyBuilder(system).render()
    assert text.index("forcefield.itp") < text.index("NATC_ffbonded.itp") < text.index("ions.itp")
    assert "other.itp" not in text and "NATC   5" in text and "SOL   0" in text


def test_cli_validate_and_templates(capsys) -> None:
    from pathlib import Path

    job = Path(__file__).resolve().parents[1] / "examples" / "jobs" / "FAST_NATC_1.txt"
    assert main(["validate", str(job)]) == 0
    assert "stage MD_AA" in capsys.readouterr().out
    assert main(["templates"]) == 0
    assert "EM_AA" in capsys.readouterr().out


def test_cli_reports_bad_config(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("sys\nname = x\nend\n")
    assert main(["validate", str(bad)]) == 2
    assert "error" in capsys.readouterr().err


def test_trend_tests_detect_drift_and_noise() -> None:
    drifting = np.array([8.1, 7.0, 6.2, 5.9, 5.4])
    _, _, p_slope = pmf.slope_test(drifting)
    assert p_slope < 0.05
    _, p_rank = pmf.trend_pvalue(np.array([1.0, 3.0, 2.0, 4.0, 2.5, 3.5][:5]))
    assert p_rank > 0.05


def test_incomplete_beta_matches_scipy() -> None:
    scipy_special = pytest.importorskip("scipy.special")
    assert pmf._betainc_half(2.5, 0.5, 0.3) == pytest.approx(scipy_special.betainc(2.5, 0.5, 0.3), rel=1e-6)


def test_internal_wham_pipeline_recovers_flat_profile(tmp_path, monkeypatch) -> None:
    rng = np.random.default_rng(0)
    cfg_k, kt = 300.0, 0.0083144621 * 298.0
    meta = []
    for i, centre in enumerate(np.arange(0.1, 4.6, 0.2)):
        path = tmp_path / f"pullx_{i}.xvg"
        times = np.arange(1000, 80000, 80.0)
        values = rng.normal(centre, np.sqrt(kt / cfg_k), times.size)
        path.write_text("".join(f"{t} {v}\n" for t, v in zip(times, values)))
        meta.append(f"{path} {centre} {cfg_k}")
    (tmp_path / "pullx-files.dat").write_text("\n".join(meta) + "\n")
    monkeypatch.chdir(tmp_path)
    code = pmf.main(
        [
            "--wham",
            "internal",
            "--blocks",
            "4",
            "--cum-steps",
            "4",
            "--bins",
            "46",
            "--region-a",
            "4.0",
            "4.5",
            "--region-b",
            "0.0",
            "0.6",
            "--outdir",
            "out",
        ]
    )
    assert code == 0
    data = np.loadtxt(tmp_path / "out" / "pmf_blocked.dat")
    in_range = data[(data[:, 0] > 0.3) & (data[:, 0] < 4.0), 1]
    assert np.abs(in_range).max() < 2.0
    assert (tmp_path / "out" / "cumulative_dG.dat").exists()

from __future__ import annotations

from pathlib import Path

from bilesalt import resources
from bilesalt.mdp import MdpFile


def test_read_strips_comments_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.mdp"
    path.write_text("; header\n\ndt = 0.002 ; inline\nref_t = 298 298\n")
    mdp = MdpFile.read(path)
    assert mdp["dt"] == "0.002" and mdp["ref_t"] == "298 298"


def test_dash_and_underscore_are_equivalent(tmp_path: Path) -> None:
    mdp = MdpFile({"tc-grps": "A B"})
    mdp.update({"tc_grps": ["MM", "SOLV"]})
    assert list(mdp.items()) == [("tc-grps", "MM SOLV")]
    assert "TC_GRPS" in mdp


def test_write_round_trip(tmp_path: Path) -> None:
    mdp = MdpFile({"integrator": "md", "nsteps": 100})
    out = mdp.write(tmp_path / "b.mdp")
    assert MdpFile.read(out)["nsteps"] == "100"


def test_all_bundled_templates_are_parseable() -> None:
    names = resources.available_templates()
    assert {"EM_AA", "NVT_AA", "NPT_AA", "MD_AA", "PULL_US", "ions"} <= set(names)
    for name in names:
        assert "integrator" in MdpFile.read(resources.mdp_template(name))

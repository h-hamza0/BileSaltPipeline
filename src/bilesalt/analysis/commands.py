from __future__ import annotations

import pickle
import shutil
from typing import Callable, Dict, List

import numpy as np

from bilesalt.analysis.context import AnalysisContext, as_list
from bilesalt.xvg import read_xvg

RESIDUE_GROUPS = ("DPHE", "CYS", "PHE", "DTRP", "LYS", "THR", "THO")
HEAD_BEADS = ("NC3", "PO4", "GL1", "GL2")
TAIL_BEADS = ("C1A", "D2A", "C2A", "C3A", "C4A", "C1B", "D2B", "C2B", "C3B", "C4B")
REGION_HEAD_BEADS = HEAD_BEADS + ("ES", "C3")
REGION_TAIL_BEADS = ("R0", "R1", "R2", "R3", "R4", "R5", "C1", "C2") + TAIL_BEADS + ("C5A", "C5B")


def _atoms_expression(names) -> str:
    return " | ".join(f"a {n}" for n in names)


def potential(ctx: AnalysisContext, opt) -> None:
    ctx.gmx.energy(ctx.file(".edr"), "Potential", ctx.output)


def temperature(ctx: AnalysisContext, opt) -> None:
    ctx.gmx.energy(ctx.file(".edr"), "Temperature", ctx.output)


def pressure(ctx: AnalysisContext, opt) -> None:
    ctx.gmx.energy(ctx.file(".edr"), "Pressure", ctx.output)


def gyrate(ctx: AnalysisContext, opt) -> None:
    group = ctx.group(opt)
    trimmed_index = ctx.setup / "removedPEG.ndx"
    count = len(ctx.groups())
    script = f"{group} & !a O1 & !a NH\nname {count} TRIM\nq\n"
    ctx.gmx.run("make_ndx", "-f", ctx.stage / "clust.gro", "-n", ctx.index, "-o", trimmed_index, stdin=script)
    ctx.gmx.run(
        "gyrate",
        "-f",
        ctx.file("_NOJUMP.xtc"),
        "-o",
        ctx.output,
        "-n",
        trimmed_index,
        "-seltype",
        "whole_res_com",
        "-selrpos",
        "whole_res_com",
        "-s",
        ctx.file(".tpr"),
        stdin=f"{count}\n",
    )


def density(ctx: AnalysisContext, opt) -> None:
    axis = as_list(opt)[0][-1] if opt else "Z"
    ctx.gmx.run(
        "density",
        "-f",
        ctx.file("_NOJUMP.xtc"),
        "-o",
        ctx.output,
        "-n",
        ctx.index,
        "-s",
        ctx.file(".tpr"),
        "-d",
        axis,
    )


def sasa(ctx: AnalysisContext, opt) -> None:
    ctx.gmx.run(
        "sasa",
        "-f",
        ctx.file(".xtc"),
        "-o",
        ctx.output,
        "-n",
        ctx.index,
        "-s",
        ctx.file(".tpr"),
        stdin=f"{ctx.group(as_list(opt)[0])}\n",
    )


def sasaHP(ctx: AnalysisContext, opt) -> None:
    sasa(ctx, opt)


def sasaHD(ctx: AnalysisContext, opt) -> None:
    sasa(ctx, opt)


def setRegions(ctx: AnalysisContext, opt) -> None:
    count = len(ctx.groups())
    script = (
        f"{_atoms_expression(REGION_HEAD_BEADS)}\nname {count} HEAD\n"
        f"{_atoms_expression(REGION_TAIL_BEADS)}\nname {count + 1} TAIL\nq\n"
    )
    ctx.gmx.run(
        "make_ndx", "-f", ctx.stage / "MD.gro", "-n", ctx.index, "-o", ctx.setup / "addS.ndx", stdin=script
    )


def radial(ctx: AnalysisContext, opt) -> None:
    import MDAnalysis as mda
    from MDAnalysis.core.groups import AtomGroup

    options = as_list(opt)
    structure = ctx.stage / "prettyView_clust.gro"
    extended = ctx.setup / "add.ndx"
    count = len(ctx.groups())
    script = (
        f"{_atoms_expression(HEAD_BEADS)}\nname {count} HEAD\n"
        f"{_atoms_expression(TAIL_BEADS)}\nname {count + 1} TAIL\nq\n"
    )
    ctx.gmx.run("make_ndx", "-f", structure, "-n", ctx.index, "-o", extended, stdin=script)
    members = ctx.atoms(extended)
    universe = mda.Universe(str(structure))
    reference = AtomGroup(universe.atoms[members[options[0]]]).center_of_mass()
    distances = {
        group: [
            float(np.linalg.norm(reference - position) / 10)
            for position in universe.atoms[members[group]].positions
        ]
        for group in options[1:]
    }
    with open(ctx.stage / "radial.pkl", "wb") as handle:
        pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cleanUp(ctx: AnalysisContext, opt) -> None:
    group, cutoff = as_list(opt)[:2]
    structure = ctx.stage / "clust.gro"
    tpr = ctx.file("_NEW.tpr")
    if not tpr.exists():
        tpr = ctx.file(".tpr")
    largest = ctx.stage / "maxclust.ndx"
    ctx.gmx.run(
        "clustsize",
        "-f",
        structure,
        "-n",
        ctx.index,
        "-mcn",
        largest,
        "-cut",
        cutoff,
        stdin=f"{ctx.group(group)}\n",
    )
    destination = ctx.setup / "indexCL.ndx"
    shutil.move(str(largest), str(destination))
    ctx.gmx.run(
        "trjconv",
        "-f",
        structure,
        "-n",
        destination,
        "-o",
        ctx.stage / f"{ctx.option}.gro",
        "-pbc",
        "whole",
        "-conect",
        "-s",
        tpr,
    )


def prettyView(ctx: AnalysisContext, opt) -> None:
    group = ctx.group(as_list(opt)[0])
    ctx.gmx.run(
        "trjconv",
        "-f",
        ctx.file(".gro"),
        "-s",
        ctx.file(".tpr"),
        "-o",
        ctx.stage / "clust.gro",
        "-pbc",
        "cluster",
        "-n",
        ctx.index,
        stdin=f"{group}\n{group}\n",
    )


def pbcNOJUMP(ctx: AnalysisContext, opt) -> None:
    group = ctx.group(as_list(opt)[0])
    ctx.gmx.run(
        "trjconv",
        "-f",
        ctx.file(".xtc"),
        "-o",
        ctx.file("_NOJUMP.xtc"),
        "-s",
        ctx.file(".tpr"),
        "-pbc",
        "mol",
        "-n",
        ctx.index,
        "-center",
        "-b",
        100000,
        stdin=f"{group}\n" * 3,
    )


def convertTpr(ctx: AnalysisContext, opt) -> None:
    ctx.gmx.run(
        "convert-tpr",
        "-s",
        ctx.file(".tpr"),
        "-n",
        ctx.index,
        "-o",
        ctx.file("_trim.tpr"),
        stdin="non-Water\n",
    )


def newTPR(ctx: AnalysisContext, opt) -> None:
    dry = ctx.file("_dry.gro")
    ctx.gmx.grompp(
        ctx.file(".mdp"), dry, ctx.topology, ctx.file("_new.tpr"), index=ctx.index, restraints=dry, maxwarn=0
    )


def clusOverTime(ctx: AnalysisContext, opt) -> None:
    first, second = as_list(opt)[:2]
    ctx.gmx.run(
        "trjconv",
        "-f",
        ctx.file(".xtc"),
        "-s",
        ctx.file(".tpr"),
        "-pbc",
        "cluster",
        "-o",
        ctx.file("_CLUST.xtc"),
        "-n",
        ctx.index,
        "-skip",
        10,
        stdin=f"{ctx.group(first)}\n{ctx.group(second)}\n",
    )


def trim(ctx: AnalysisContext, opt) -> None:
    ctx.gmx.run(
        "trjconv",
        "-f",
        ctx.file(".gro"),
        "-s",
        ctx.file(".tpr"),
        "-o",
        ctx.stage / "trimmed.pdb",
        "-n",
        ctx.index,
        "-conect",
        stdin="non-Water\n",
    )


def aspectRatio(ctx: AnalysisContext, opt) -> None:
    index = ctx.setup / "removedPEG.ndx"
    moments = ctx.stage / "moi.xvg"
    ctx.gmx.run(
        "principal",
        "-f",
        ctx.file("_NOJUMP.xtc"),
        "-n",
        index,
        "-s",
        ctx.file(".tpr"),
        "-om",
        moments,
        stdin=f"{ctx.group(as_list(opt)[0], index)}\n",
    )
    data, _ = read_xvg(moments)
    time, ordered = data[:, 0], np.sort(data[:, 1:4], axis=1)[:, ::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (2 * ordered[:, 0] - ordered[:, 1] - ordered[:, 2]) / ordered.sum(axis=1)
    ratio[~np.isfinite(ratio)] = np.nan
    np.save(ctx.stage / "aspectRatio.npy", np.vstack((time, ratio)))


def contactAnalysis(ctx: AnalysisContext, opt) -> None:
    import MDAnalysis as mda
    import pandas as pd
    from MDAnalysis.analysis import contacts

    universe = mda.Universe(str(ctx.file(".tpr")), str(ctx.file(".xtc")))
    membrane = universe.select_atoms("resname NATC or resname NATD or resname NATDC")
    selections = {name: universe.select_atoms(f"resname {name}") for name in RESIDUE_GROUPS}
    rows = []
    for ts in universe.trajectory:
        row = {"frame": ts.frame}
        for name, selection in selections.items():
            distances = contacts.distance_array(selection.positions, membrane.positions)
            row[name] = int((distances < 4.5).sum())
        rows.append(row)
    pd.DataFrame(rows).set_index("frame").to_pickle(ctx.stage / f"{ctx.option}_BS-OCT.pkl")


def SASA_RES(ctx: AnalysisContext, opt) -> None:
    import MDAnalysis as mda
    from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis

    universe = mda.Universe(str(ctx.file(".tpr")), str(ctx.file(".xtc")))
    areas: List[list] = []
    for name in RESIDUE_GROUPS:
        analysis = SASAAnalysis(universe.select_atoms(f"resname {name}"))
        analysis.step = 500
        analysis.run()
        areas.append([name, analysis.results.total_area])
    with open(ctx.stage / f"{ctx.option}.pkl", "wb") as handle:
        pickle.dump(areas, handle)


def HBA_RES(ctx: AnalysisContext, opt) -> None:
    import MDAnalysis as mda
    import pandas as pd
    from MDAnalysis.lib.distances import capped_distance

    universe = mda.Universe(str(ctx.file(".tpr")), str(ctx.file(".xtc")))
    solvent = universe.select_atoms("resname SOL")
    selections = {name: universe.select_atoms(f"resname {name}") for name in RESIDUE_GROUPS}
    rows = []
    for _ in universe.trajectory:
        row = {}
        for name, selection in selections.items():
            pairs = capped_distance(
                selection.positions, solvent.positions, max_cutoff=3.5, box=universe.dimensions
            )
            row[name] = len(np.unique(pairs[0][:, 1])) / selection.n_atoms
        rows.append(row)
    pd.DataFrame(rows).to_pickle(ctx.stage / "solventContact.pkl")


def connected_clusters(adjacency: List[List[int]]) -> List[List[int]]:
    visited = [False] * len(adjacency)
    clusters = []
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        visited[start] = True
        stack, cluster = [start], []
        while stack:
            node = stack.pop()
            cluster.append(node)
            for neighbour in adjacency[node]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    stack.append(neighbour)
        clusters.append(cluster)
    return clusters


def extractLargest(ctx: AnalysisContext, opt) -> None:
    import MDAnalysis as mda
    from MDAnalysis.selections.gromacs import SelectionWriter
    from scipy.spatial import cKDTree

    radius = float(as_list(opt)[0])
    universe = mda.Universe(str(ctx.stage / "clust.gro"))
    residues = list(universe.residues)
    centers = np.array([residue.atoms.center_of_mass() for residue in residues])
    tree = cKDTree(centers)
    adjacency = [
        [j for j in tree.query_ball_point(centers[i], radius) if j != i] for i in range(len(residues))
    ]
    targets = {i for i, residue in enumerate(residues) if residue.resname == "OCT"}
    clusters = [c for c in connected_clusters(adjacency) if targets.intersection(c)]
    if not clusters:
        raise ValueError("No cluster contains an OCT residue")
    largest = max(clusters, key=len)
    indices = np.concatenate([residues[i].atoms.indices for i in largest])
    selection = universe.atoms[indices]
    selection = selection[np.lexsort((selection.indices, selection.resids, selection.resnames))]
    selection.write(str(ctx.stage / "clustL.gro"))
    with SelectionWriter(str(ctx.setup / "adjusted.ndx"), mode="w") as writer:
        writer.write(selection, name="LG")


Command = Callable[[AnalysisContext, object], None]

COMMANDS: Dict[str, Command] = {
    "potential": potential,
    "temperature": temperature,
    "pressure": pressure,
    "gyrate": gyrate,
    "density": density,
    "sasa": sasa,
    "sasaHP": sasaHP,
    "sasaHD": sasaHD,
    "aspectRatio": aspectRatio,
    "cleanUp": cleanUp,
    "prettyView": prettyView,
    "clusOverTime": clusOverTime,
    "pbcNOJUMP": pbcNOJUMP,
    "radial": radial,
    "setRegions": setRegions,
    "contactAnalysis": contactAnalysis,
    "SASA_RES": SASA_RES,
    "HBA_RES": HBA_RES,
    "convertTpr": convertTpr,
    "extractLargest": extractLargest,
    "trim": trim,
    "newTPR": newTPR,
}

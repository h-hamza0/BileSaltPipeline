from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from bilesalt.analysis.commands import COMMANDS
from bilesalt.analysis.context import AnalysisContext, as_list
from bilesalt.config import ConfigError, parse_value, split_blocks
from bilesalt.gromacs import Gromacs
from bilesalt.shell import LOGGER
from bilesalt.xvg import read_xvg

Curve = Dict[str, object]


@dataclass
class AnalysisStep:
    name: str
    options: Dict[str, Union[str, List[str]]]

    @property
    def commands(self) -> List[str]:
        requested = as_list(self.options.get("analysis"))
        unknown = [c for c in requested if c not in COMMANDS]
        if unknown:
            raise ConfigError(f"Unknown analysis command(s): {', '.join(unknown)}")
        return requested

    def execute(self, location: Path, gmx: Gromacs) -> Dict[str, List[Curve]]:
        curves: Dict[str, List[Curve]] = {}
        for command in self.commands:
            context = AnalysisContext(location, self.name, command, gmx)
            LOGGER.info("%s: %s", self.name, command)
            COMMANDS[command](context, self.options.get(command))
            curves[command] = collect_curves(context, command)
        return curves


def collect_curves(context: AnalysisContext, command: str) -> List[Curve]:
    if command == "aspectRatio":
        data = np.load(context.stage / "aspectRatio.npy").T
        return [{"data": data, "xaxis": "Time (ps)", "yaxis": "Aspect ratio", "title": "Aspect Ratio"}]
    if command == "radial":
        from scipy import stats

        with open(context.stage / "radial.pkl", "rb") as handle:
            distances = pickle.load(handle)
        grid = np.linspace(0, 9, 1000)
        return [
            {
                "data": np.column_stack((grid, stats.gaussian_kde(values)(grid))),
                "label": group,
                "xaxis": "Distance (nm)",
                "yaxis": "Density",
                "title": "Radial Distribution",
            }
            for group, values in distances.items()
        ]
    if context.output.exists():
        data, attrs = read_xvg(context.output)
        return [{"data": data, **attrs}]
    return []


def plot_curves(curves: Dict[str, List[Curve]], destination: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for command, series in curves.items():
        if not series:
            continue
        figure, axes = plt.subplots(figsize=(5.5, 4))
        for curve in series:
            data = curve["data"]
            style = {"lw": 1.5, "label": curve["label"]} if "label" in curve else {"lw": 1.5, "color": "k"}
            axes.plot(data[:, 0], data[:, 1], **style)
        axes.set_xlabel(series[0].get("xaxis", ""))
        axes.set_ylabel(series[0].get("yaxis", ""))
        axes.set_title(series[0].get("title", command), pad=12)
        if any("label" in curve for curve in series):
            axes.legend()
        path = destination / f"{command}.png"
        figure.savefig(path, dpi=300, transparent=True, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


@dataclass
class AnalysisPlan:
    result_folder: Path
    steps: List[AnalysisStep] = field(default_factory=list)

    @classmethod
    def parse(cls, text: str) -> AnalysisPlan:
        folder: Optional[Path] = None
        steps: List[AnalysisStep] = []
        for header, body in split_blocks(text):
            kind = header.split("_")[0]
            if kind == "system":
                folder = Path(body["resultFolder"])
            elif kind == "stage":
                if "name" not in body:
                    raise ConfigError(f"Block '{header}' needs a 'name'")
                options = {k: parse_value(v) for k, v in body.items() if k != "name"}
                steps.append(AnalysisStep(body["name"], options))
            else:
                raise ConfigError(f"Unknown block '{header}'")
        if folder is None:
            raise ConfigError("Analysis file needs a 'system' block with 'resultFolder'")
        return cls(folder, steps)

    @classmethod
    def load(cls, path: Union[str, Path]) -> AnalysisPlan:
        return cls.parse(Path(path).read_text())

    def run(self, gmx: Optional[Gromacs] = None, plot: bool = True) -> Dict[str, Dict[str, List[Curve]]]:
        gmx = gmx or Gromacs()
        results: Dict[str, Dict[str, List[Curve]]] = {}
        for step in self.steps:
            curves = step.execute(self.result_folder, gmx)
            if plot:
                plot_curves(curves, self.result_folder / step.name)
            results[step.name] = curves
        with open(self.result_folder / "collection.pkl", "wb") as handle:
            pickle.dump(
                {
                    n: {c: [{k: v for k, v in s.items()} for s in series] for c, series in r.items()}
                    for n, r in results.items()
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        return results

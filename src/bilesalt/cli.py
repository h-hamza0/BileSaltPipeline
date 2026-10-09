from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from bilesalt import __version__, resources
from bilesalt.config import ConfigError, SlurmConfig
from bilesalt.demo import DemoError
from bilesalt.slurm import SlurmSubmitter


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING - 10 * min(verbose + 1, 2)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")


def _run(args: argparse.Namespace) -> int:
    from bilesalt.pipeline import Pipeline

    Pipeline.from_file(args.job).run()
    return 0


def _validate(args: argparse.Namespace) -> int:
    from bilesalt.pipeline import Pipeline

    for line in Pipeline.from_file(args.job).describe():
        print(line)
    return 0


def _submit(args: argparse.Namespace) -> int:
    jobs = sorted(Path(args.directory).glob("*.txt"))
    if not jobs:
        print(f"No .txt job files in {args.directory}", file=sys.stderr)
        return 1
    submitter = SlurmSubmitter(
        SlurmConfig(
            cpu_partition=args.partition,
            mail_user=args.mail_user,
            gmxrc=args.gmxrc,
            cpus=args.cpus,
            memory=args.mem,
        )
    )
    for job in jobs:
        body = f"bilesalt run {job.resolve()}\n"
        if args.dry_run:
            print(f"# {job.name}\n{submitter.preamble()}{body}")
        else:
            job_id = submitter.submit(body, job.stem, job.parent.resolve())
            print(f"{job.name}: submitted as job {job_id}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from bilesalt.analysis import AnalysisPlan
    from bilesalt.gromacs import Gromacs

    AnalysisPlan.load(args.plan).run(Gromacs(args.gmx), plot=not args.no_plot)
    return 0


def _wham(args: argparse.Namespace) -> int:
    from bilesalt.analysis import wham

    submitter = SlurmSubmitter(SlurmConfig(cpu_partition=args.partition, mail_user=args.mail_user))
    outputs = wham.submit_all(
        Path(args.root), submitter, dry_run=args.dry_run, cpus=args.cpus, memory=args.mem, gmx=args.gmx
    )
    print("\n".join(outputs))
    return 0


def _convergence(args: argparse.Namespace) -> int:
    from bilesalt.logp import pmf_convergence

    return pmf_convergence.main(args.arguments)


def _demo(args: argparse.Namespace) -> int:
    from bilesalt import demo

    state = demo.run(args.workdir, args.gmx)
    print(f"Demo finished. Final structure: {state.structure}")
    return 0


def _templates(args: argparse.Namespace) -> int:
    print("\n".join(resources.available_templates()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bilesalt", description="Bile-salt/peptide simulation pipeline")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute a pipeline job file")
    run.add_argument("job", type=Path)
    run.set_defaults(handler=_run)

    validate = sub.add_parser("validate", help="parse a job file and print the planned workflow")
    validate.add_argument("job", type=Path)
    validate.set_defaults(handler=_validate)

    submit = sub.add_parser("submit", help="submit every job file in a directory to Slurm")
    submit.add_argument("directory", type=Path)
    submit.add_argument("--partition")
    submit.add_argument("--mail-user")
    submit.add_argument("--gmxrc")
    submit.add_argument("--cpus", type=int, default=4)
    submit.add_argument("--mem", default="16gb")
    submit.add_argument("--dry-run", action="store_true")
    submit.set_defaults(handler=_submit)

    analyze = sub.add_parser("analyze", help="run an analysis plan")
    analyze.add_argument("plan", type=Path)
    analyze.add_argument("--gmx", default="gmx")
    analyze.add_argument("--no-plot", action="store_true")
    analyze.set_defaults(handler=_analyze)

    wham = sub.add_parser("wham", help="submit gmx wham jobs for every US_*/PULL_US directory")
    wham.add_argument("root", type=Path)
    wham.add_argument("--partition")
    wham.add_argument("--mail-user")
    wham.add_argument("--cpus", type=int, default=8)
    wham.add_argument("--mem", default="16G")
    wham.add_argument("--gmx", default="gmx")
    wham.add_argument("--dry-run", action="store_true")
    wham.set_defaults(handler=_wham)

    convergence = sub.add_parser(
        "pmf-convergence", help="block and cumulative convergence of an umbrella PMF", add_help=False
    )
    convergence.add_argument("arguments", nargs=argparse.REMAINDER)
    convergence.set_defaults(handler=_convergence)

    demo = sub.add_parser(
        "demo", help="run a small self-contained GROMACS example (no HPC or external inputs)"
    )
    demo.add_argument("workdir", type=Path)
    demo.add_argument("--gmx", default="gmx")
    demo.set_defaults(handler=_demo)

    templates = sub.add_parser("templates", help="list bundled MDP templates")
    templates.set_defaults(handler=_templates)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return args.handler(args)
    except (ConfigError, FileNotFoundError, DemoError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

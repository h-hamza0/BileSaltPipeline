#!/usr/bin/env python3
"""
Convergence tests for an umbrella-sampling PMF: disjoint blocks and cumulative.

The standard practice: cut the production data of every window into N equal
contiguous time blocks (0-2 ns, 2-4 ns, ...), solve WHAM independently in each,
and ask whether the resulting profiles agree. If they do, the extra data was
not changing the answer. If they do not, run longer.

This script does that and quantifies the "agree" step, which is usually left to
the eye. Three things come out:

  1. AN ERROR BAR.  The spread across blocks gives sigma on whatever scalar you
     report -- dG, a barrier, logP.

  2. A DRIFT TEST.  This is the part that matters and the part eyeballing gets
     wrong. Block profiles can differ in two completely different ways:

        scatter  -- blocks disagree randomly. That is just statistical noise,
                    and the spread IS your error bar. Converged.
        drift    -- blocks move systematically from first to last. That is a
                    slow process still relaxing. NOT converged, and the block
                    spread understates the true error.

     Both look like "the profiles are a bit different" on a plot. They are
     distinguished here by testing the block-ordered scalar for a monotonic
     trend, with an exact permutation p-value (N=5 blocks gives p as low as
     1/5! = 0.008, so this has real power despite the tiny sample).

  3. A STOPPING RULE.  Since sigma scales as 1/sqrt(T) once you are in the
     asymptotic regime, the script extrapolates how much total sampling would
     be needed to reach a target error. That answers "when do I stop running"
     with a number rather than a feeling.

Usage
-----
    bilesalt pmf-convergence                       # 5 blocks, gmx wham
    bilesalt pmf-convergence --blocks 5 --reuse    # re-use block PMFs
    bilesalt pmf-convergence --wham internal       # no gmx needed

What this test CANNOT do: if every block sits in the same unrelaxed state, all
blocks agree and everything below looks perfect. Block agreement is evidence
against drift, not proof of convergence. Independent replicas are the only
check on a bias shared by all blocks equally.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from itertools import permutations
from typing import Tuple

import numpy as np

KB = {"kcal": 0.0019872041, "kJ": 0.0083144621}
LN10 = 2.302585


@dataclass
class Settings:
    tpr_list: str = "tpr-files.dat"
    pull_list: str = "pullx-files.dat"
    pull_type: str = "x"
    t_start: float = 1000.0
    t_end: float = 80000.0
    blocks: int = 5
    xi_min: float = 0.0
    xi_max: float = 4.6
    bins: int = 200
    temperature: float = 298.0
    units: str = "kJ"
    observable: str = "plateau"
    region_a: Tuple[float, float] = (4.0, 4.5)
    region_b: Tuple[float, float] = (0.0, 0.6)
    report_logp: bool = True
    target_sem: float = 4.184
    cumulative_steps: int = 16
    chem_accuracy: float = 4.184
    tolerance: float = 1e-6
    gmx: str = "gmx"
    outdir: str = "block_convergence"

    @property
    def kb(self) -> float:
        return KB[self.units]

    @property
    def rt_log10(self) -> float:
        return LN10 * self.kb * self.temperature


def read_xvg(path: str) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in "#@&":
                continue
            try:
                rows.append([float(v) for v in line.split()])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no numeric data in {path}")
    w = min(len(r) for r in rows)
    return np.array([r[:w] for r in rows])


def read_list(path: str) -> list[str]:
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def read_pmf(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a PMF file, dropping unsampled (inf/nan) bins."""
    d = read_xvg(path)
    x, a = d[:, 0], d[:, 1]
    m = np.isfinite(a)
    return x[m], a[m]


def run_gmx(cfg: Settings, b: float, e: float, out: str, hist: str) -> None:
    flag = "-ix" if cfg.pull_type == "x" else "-if"
    cmd = [
        cfg.gmx,
        "wham",
        "-it",
        cfg.tpr_list,
        flag,
        cfg.pull_list,
        "-b",
        f"{b:.3f}",
        "-e",
        f"{e:.3f}",
        "-o",
        out,
        "-hist",
        hist,
        "-min",
        str(cfg.xi_min),
        "-max",
        str(cfg.xi_max),
        "-bins",
        str(cfg.bins),
        "-unit",
        "kCal" if cfg.units == "kcal" else "kJ",
        "-temp",
        str(cfg.temperature),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"gmx wham failed on [{b:.0f}, {e:.0f}] ps")


def run_internal(cfg: Settings, b: float, e: float, out: str, _hist: str) -> None:
    """
    Self-contained 1D WHAM, so the script runs without gmx. Window centres and
    force constants are read from a plain-text metadata file (Grossfield
    format) if pull_list points at one; otherwise gmx is required.
    """
    meta = [ln.split() for ln in read_list(cfg.pull_list)]
    if not meta or len(meta[0]) < 3:
        raise RuntimeError("--wham internal needs a metadata file with 'path centre k' per line")
    beta = 1.0 / (cfg.kb * cfg.temperature)
    edges = np.linspace(cfg.xi_min, cfg.xi_max, cfg.bins + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])

    H, N, bias = [], [], []
    for path, x0, k, *_ in meta:
        d = read_xvg(path)
        t, x = d[:, 0], d[:, 1]
        x = x[(t >= b) & (t < e)]
        h, _ = np.histogram(x, bins=edges)
        if h.sum() == 0:
            continue
        H.append(h)
        N.append(h.sum())
        bias.append(0.5 * float(k) * (ctr - float(x0)) ** 2)
    H, N, bias = np.array(H), np.array(N, float), np.array(bias)

    tot = H.sum(axis=0)
    occ = tot > 0
    logH = np.log(np.where(occ, tot, 1.0))
    f = np.zeros(N.size)
    logN = np.log(N)
    for _ in range(200000):
        a = logN[:, None] + f[:, None] - beta * bias
        m = a.max(axis=0)
        logp = logH - (m + np.log(np.exp(a - m).sum(axis=0)))
        logp[~occ] = -np.inf
        q = logp[None, :] - beta * bias
        mq = np.max(np.where(np.isfinite(q), q, -np.inf), axis=1)
        fn = -(mq + np.log(np.exp(q - mq[:, None]).sum(axis=1)))
        fn -= fn[0]
        if np.max(np.abs(fn - f)) < cfg.tolerance:
            f = fn
            break
        f = fn
    A = -logp / beta
    A -= A[occ].min()
    with open(out, "w") as fh:
        fh.write("# xi  A\n")
        fh.writelines(f"{xi:.6f} {av:.6f}\n" if o else f"{xi:.6f} inf\n" for xi, av, o in zip(ctr, A, occ))


def align(x: np.ndarray, a: np.ndarray, region) -> np.ndarray:
    """Zero each profile on a common reference region -- WHAM's additive
    constant is arbitrary, so unaligned profiles cannot be compared."""
    m = (x >= region[0]) & (x <= region[1])
    if not m.any():
        raise SystemExit(f"reference region {region} contains no sampled bins")
    return a - a[m].mean()


def observable(cfg: Settings, x: np.ndarray, a: np.ndarray) -> float:
    mb = (x >= cfg.region_b[0]) & (x <= cfg.region_b[1])
    if not mb.any():
        raise SystemExit(f"region_b {cfg.region_b} contains no sampled bins")
    return float(a[mb].min() if cfg.observable == "well" else a[mb].mean())


def trend_pvalue(y: np.ndarray) -> tuple[float, float]:
    """
    Spearman rho of y against block order, exact permutation p (two-sided).

    Robust to a nonlinear drift or a single wild block, but it throws away
    MAGNITUDE: a sequence that falls by 5 sigma with one adjacent pair swapped
    scores no better than one that wanders by a hair. Use it alongside
    slope_test, never alone.
    """
    n = y.size
    if n < 4:
        return 0.0, 1.0
    rx = np.arange(n, dtype=float)
    ry = np.argsort(np.argsort(y)).astype(float)

    def rho(a, bb):
        d = a - bb
        return 1.0 - 6.0 * (d @ d) / (n * (n * n - 1))

    obs = rho(rx, ry)
    cnt = sum(1 for p in permutations(ry) if abs(rho(rx, np.array(p))) >= abs(obs))
    return float(obs), cnt / float(math.factorial(n))


def slope_test(y: np.ndarray) -> tuple[float, float, float]:
    """
    Least-squares regression of the block values on block order, testing
    slope = 0 by a t-test on M-2 degrees of freedom.

    This is the more powerful of the two tests because it uses the magnitudes.
    On a test series drifting 8.1 -> 5.4 with one adjacent pair out of order,
    the rank test gave p = 0.083 (missed it) while this gives p = 0.012.
    Returns (slope per block, total change over all blocks, p).
    """
    n = y.size
    if n < 3:
        return 0.0, 0.0, 1.0
    x = np.arange(n, dtype=float)
    sxx = ((x - x.mean()) ** 2).sum()
    slope = ((x - x.mean()) * (y - y.mean())).sum() / sxx
    resid = y - (y.mean() + slope * (x - x.mean()))
    df = n - 2
    s2 = (resid**2).sum() / df
    se = math.sqrt(s2 / sxx) if s2 > 0 else 0.0
    if se == 0:
        return float(slope), float(slope * (n - 1)), 0.0 if slope else 1.0
    t = slope / se
    # two-sided t survival without scipy
    xb = df / (df + t * t)
    p = _betainc_half(df / 2.0, 0.5, xb)
    return float(slope), float(slope * (n - 1)), float(min(max(p, 0.0), 1.0))


def _betainc_half(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a,b) by continued fraction; used for the
    t-distribution p-value so the script needs no scipy."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def solve_intervals(cfg, intervals, tag, solver, outdir, reuse, grid=None, label="segment"):
    """
    Solve WHAM on each (begin, end) interval and return (grid, profiles).

    Shared by both modes: the block mode passes disjoint intervals, the
    cumulative mode passes nested ones. Every profile is aligned on region_a
    against the SAME grid, so the two modes are directly comparable.
    """
    profiles = []
    for i, (b, e) in enumerate(intervals):
        out = os.path.join(outdir, f"pmf_{tag}{i}.xvg")
        hist = os.path.join(outdir, f"hist_{tag}{i}.xvg")
        if not (reuse and os.path.exists(out)):
            print(f"  {label} {i + 1}/{len(intervals)}: {b:.0f}-{e:.0f} ps")
            solver(cfg, b, e, out, hist)
        x, a = read_pmf(out)
        if grid is None:
            grid = x
            profiles.append(align(x, a, cfg.region_a))
        else:
            profiles.append(align(grid, np.interp(grid, x, a), cfg.region_a))
    return grid, np.array(profiles)


def report_cumulative(times, dg, u, chem_acc):
    """
    Feddersen-style convergence trace: dG from 0-t as t grows.

    Read the LAST VALUE against the half-data value, never the step size. For
    an exponentially relaxing system the cumulative estimate never forgets the
    start of the window: its residual bias falls off as ~1/k while the visible
    step between points falls off as ~1/k^2. The ratio of true bias to visible
    step therefore GROWS with the amount of data, so "the curve has flattened"
    understates the remaining error by a factor that increases the longer you
    run. Discarding the equilibration portion up front is what removes this.
    """
    print("\n=== cumulative convergence (0 to t) " + "=" * 36)
    print(f"  {'used / ps':>12} {'fraction':>10} {'dG':>12} {'step':>10}")
    for i, (t, v) in enumerate(zip(times, dg)):
        step = "" if i == 0 else f"{dg[i] - dg[i - 1]:+10.3f}"
        print(f"  {t:>12.0f} {(i + 1) / len(dg):>10.2f} {v:>12.3f} {step:>10}")

    half = dg[len(dg) // 2 - 1] if len(dg) >= 2 else dg[0]
    full = dg[-1]
    delta = abs(full - half)
    print(f"\n  dG(half data) {half:.3f}  ->  dG(all data) {full:.3f}   |change| {delta:.3f} {u}")
    print(f"  chemical-accuracy criterion: {chem_acc:.2f} {u}")
    if delta < chem_acc:
        print("  -> CONVERGED by this criterion: doubling the data moved dG by")
        print(f"     {delta:.3f} {u}, below the {chem_acc:.2f} threshold.")
    else:
        print(f"  -> NOT CONVERGED: doubling the data still moved dG by {delta:.3f} {u}.")
    if len(dg) > 2:
        laststep = abs(dg[-1] - dg[-2])
        print(
            f"\n  Note the final step is only {laststep:.3f} {u}, "
            f"{delta / max(laststep, 1e-9):.1f}x smaller than the"
        )
        print("  half-to-full change. Judge convergence on the latter: small")
        print("  steps are a property of averaging, not evidence of accuracy.")
    return full, delta


def cross_check(cum_full, last_block, dg_sem, u):
    """
    The diagnostic that ties the two modes together.

    A disjoint block is an unbiased estimate of its own time interval, so the
    LAST block is the cleanest estimate of the current state. The cumulative
    value still contains everything from the start. If they differ by more than
    the block error, the cumulative estimate is being dragged by early data --
    which is the bias described in report_cumulative, made visible.
    """
    print("\n=== cumulative vs final block " + "=" * 42)
    gap = abs(cum_full - last_block)
    print(f"  cumulative over all data : {cum_full:.3f} {u}")
    print(f"  final disjoint block     : {last_block:.3f} {u}")
    print(f"  gap {gap:.3f} vs block SEM {dg_sem:.3f} {u}")
    if gap > 2 * dg_sem:
        print("  -> The cumulative estimate is still carrying early data. The")
        print("     final block is the cleaner number; consider discarding more")
        print("     equilibration and rerunning both analyses.")
    else:
        print("  -> Consistent. The cumulative estimate is not being dragged by")
        print("     the early portion of the run.")


def report_blocks(cfg, P, dg, grid, edges, M, span, u, args):
    """Everything that only makes sense for disjoint blocks."""
    mean = P.mean(axis=0)
    sem = P.std(axis=0, ddof=1) / np.sqrt(M)
    dg_mean = dg.mean()
    dg_sem = dg.std(ddof=1) / np.sqrt(M)
    # ---- per-block table
    print(f"\n=== per-block {'dG' if cfg.observable == 'plateau' else 'well depth'} " + "=" * 40)
    print(f"  {'block':>6} {'window / ps':>18} {'value':>12}")
    for m in range(M):
        print(f"  {m + 1:>6} {edges[m]:>7.0f}-{edges[m + 1]:<10.0f} {dg[m]:>12.3f}")
    print(f"\n  mean {dg_mean:.3f} +/- {dg_sem:.3f} {u}   (spread {dg.max() - dg.min():.3f})")
    print(
        f"  each block used {span:.0f} ps; the SEM itself carries "
        f"~{100 / math.sqrt(2 * (M - 1)):.0f}% uncertainty at M={M}"
    )

    # ---- are the profiles "the same"?
    dev = np.abs(P - mean).max()
    print("\n=== do the block profiles agree? " + "=" * 39)
    print(f"  largest deviation of any block from the mean profile: {dev:.3f} {u}")
    print(f"  mean pointwise SEM across the profile:                {sem.mean():.3f} {u}")

    rho, p_rank = trend_pvalue(dg)
    slope, total_change, p_slope = slope_test(dg)
    print("\n  trend across blocks, two independent tests:")
    print(
        f"    regression slope {slope:+.3f} {u} per block "
        f"({total_change:+.3f} first to last), p = {p_slope:.3f}"
    )
    print(f"    rank correlation rho = {rho:+.2f}, exact p = {p_rank:.3f}")
    drifting = (p_slope < 0.05) or (p_rank < 0.05)
    if drifting:
        print("\n  VERDICT: DRIFTING. The blocks move systematically, not")
        print(f"  randomly, so a slow process is still relaxing. The +/- {dg_sem:.3f} above")
        print("  UNDERSTATES the true error. Run longer, or discard more")
        print("  equilibration and repeat.")
    else:
        print("\n  VERDICT: NO DETECTABLE DRIFT. The blocks differ randomly,")
        print(f"  which is statistical noise, and {dg_sem:.3f} {u} is a fair")
        print("  estimate of it. Note this cannot detect a bias shared by all")
        print("  blocks equally -- only replicas can.")

    # ---- logP
    if cfg.report_logp:
        rt = cfg.rt_log10
        print("\n=== logP " + "=" * 62)
        print(f"  2.303 RT = {rt:.3f} {u} at {cfg.temperature:.2f} K")
        print(f"  logP = {-dg_mean / rt:.2f} +/- {dg_sem / rt:.2f}")
        print(f"  (an error of {rt:.2f} {u} in dG is exactly 1.0 log unit)")

    # ---- stopping rule
    print("\n=== how much longer? " + "=" * 50)
    total = cfg.t_end - cfg.t_start
    if drifting:
        print("  Not meaningful while the blocks are drifting: the 1/sqrt(T)")
        print("  extrapolation assumes you are already in the asymptotic")
        print("  regime, and a drifting estimate is not.")
    elif dg_sem <= args.target:
        print(
            f"  Target of {args.target:.3f} {u} already met "
            f"({dg_sem:.3f} <= {args.target:.3f}). More sampling is not"
        )
        print("  the limiting factor on this number.")
    else:
        need = total * (dg_sem / args.target) ** 2
        print(f"  sigma scales as 1/sqrt(T), so reaching {args.target:.3f} {u} needs")
        print(
            f"  about {need:.0f} ps per window, versus {total:.0f} ps now -- a factor of {need / total:.1f}."
        )
        if cfg.report_logp:
            rt = cfg.rt_log10
            print(f"  ({args.target:.3f} {u} corresponds to {args.target / rt:.2f} log units)")

    return mean, sem, dg_mean, dg_sem, drifting, p_slope, p_rank


def build_parser() -> argparse.ArgumentParser:
    defaults = Settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", type=int, default=defaults.blocks)
    ap.add_argument(
        "--mode",
        choices=("blocks", "cumulative", "both"),
        default="both",
        help="'blocks' = disjoint time blocks (error bar + drift "
        "test); 'cumulative' = dG from 0 to t as t grows "
        "(Feddersen-style); 'both' (default) runs each and "
        "cross-checks them",
    )
    ap.add_argument(
        "--cum-steps",
        type=int,
        default=defaults.cumulative_steps,
        help="number of points on the cumulative curve",
    )
    ap.add_argument(
        "--chem-acc",
        type=float,
        default=defaults.chem_accuracy,
        help="convergence threshold for the cumulative curve, in "
        "UNITS/mol (default 4.184 kJ/mol = 1 kcal/mol)",
    )
    ap.add_argument("--reuse", action="store_true", help="reuse PMFs already in the output directory")
    ap.add_argument("--wham", choices=("gmx", "internal"), default="gmx")
    ap.add_argument("--outdir", default=defaults.outdir)
    ap.add_argument("--target", type=float, default=defaults.target_sem)
    ap.add_argument("--gmx", default=defaults.gmx)
    ap.add_argument("--tpr-list", default=defaults.tpr_list)
    ap.add_argument("--pull-list", default=defaults.pull_list)
    ap.add_argument("--pull-type", choices=("x", "f"), default=defaults.pull_type)
    ap.add_argument("--t-start", type=float, default=defaults.t_start, help="ps")
    ap.add_argument("--t-end", type=float, default=defaults.t_end, help="ps")
    ap.add_argument("--xi-min", type=float, default=defaults.xi_min)
    ap.add_argument("--xi-max", type=float, default=defaults.xi_max)
    ap.add_argument("--bins", type=int, default=defaults.bins)
    ap.add_argument("--temperature", type=float, default=defaults.temperature)
    ap.add_argument("--units", choices=tuple(KB), default=defaults.units)
    ap.add_argument("--observable", choices=("plateau", "well"), default=defaults.observable)
    ap.add_argument(
        "--region-a",
        type=float,
        nargs=2,
        default=defaults.region_a,
        metavar=("LO", "HI"),
        help="reference (zero) region in xi",
    )
    ap.add_argument(
        "--region-b",
        type=float,
        nargs=2,
        default=defaults.region_b,
        metavar=("LO", "HI"),
        help="target region in xi",
    )
    ap.add_argument("--no-logp", action="store_true", help="skip the dG -> logP conversion")
    return ap


def settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings(
        tpr_list=args.tpr_list,
        pull_list=args.pull_list,
        pull_type=args.pull_type,
        t_start=args.t_start,
        t_end=args.t_end,
        blocks=args.blocks,
        xi_min=args.xi_min,
        xi_max=args.xi_max,
        bins=args.bins,
        temperature=args.temperature,
        units=args.units,
        observable=args.observable,
        region_a=tuple(args.region_a),
        region_b=tuple(args.region_b),
        report_logp=not args.no_logp,
        target_sem=args.target,
        cumulative_steps=args.cum_steps,
        chem_accuracy=args.chem_acc,
        gmx=args.gmx,
        outdir=args.outdir,
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = settings_from_args(args)

    if args.wham == "gmx" and shutil.which(cfg.gmx) is None:
        sys.exit(f"'{cfg.gmx}' not on PATH. Source your GMXRC, or use --wham internal.")
    os.makedirs(cfg.outdir, exist_ok=True)

    M = cfg.blocks
    edges = np.linspace(cfg.t_start, cfg.t_end, M + 1)
    span = (cfg.t_end - cfg.t_start) / M
    solver = run_gmx if args.wham == "gmx" else run_internal
    u = f"{cfg.units}/mol"
    do_blocks = args.mode in ("blocks", "both")
    do_cum = args.mode in ("cumulative", "both")

    print(f"\nspanning {cfg.t_start:.0f}-{cfg.t_end:.0f} ps")
    print(f"aligned on {cfg.region_a} ; observable = '{cfg.observable}' over {cfg.region_b}")
    if do_blocks:
        print(f"blocks: {M} disjoint blocks of {span:.0f} ps")
    if do_cum:
        print(f"cumulative: {cfg.cumulative_steps} nested windows from {cfg.t_start:.0f} ps")
    print()

    grid = None
    dg_mean = dg_sem = None
    P = dg = None

    if do_blocks:
        intervals = [(edges[m], edges[m + 1]) for m in range(M)]
        grid, P = solve_intervals(cfg, intervals, "block", solver, cfg.outdir, args.reuse, grid, "block")
        dg = np.array([observable(cfg, grid, p) for p in P])
        (mean, sem, dg_mean, dg_sem, drifting, p_slope, p_rank) = report_blocks(
            cfg, P, dg, grid, edges, M, span, u, args
        )

    cum_t = cum_dg = None
    if do_cum:
        total = cfg.t_end - cfg.t_start
        fracs = np.arange(1, cfg.cumulative_steps + 1) / cfg.cumulative_steps
        cum_iv = [(cfg.t_start, cfg.t_start + f * total) for f in fracs]
        grid, C = solve_intervals(cfg, cum_iv, "cum", solver, cfg.outdir, args.reuse, grid, "cumulative")
        cum_dg = np.array([observable(cfg, grid, c) for c in C])
        cum_t = np.array([e for _, e in cum_iv]) - cfg.t_start
        cum_full, cum_delta = report_cumulative(cum_t, cum_dg, u, cfg.chem_accuracy)
        if do_blocks:
            cross_check(cum_full, dg[-1], dg_sem, u)

    if do_cum:
        cpath = os.path.join(cfg.outdir, "cumulative_dG.dat")
        with open(cpath, "w") as fh:
            fh.write("# cumulative convergence trace\n# time_used/ps  dG\n")
            fh.writelines(f"{t:12.2f} {v:12.5f}\n" for t, v in zip(cum_t, cum_dg))
        print(f"\n  wrote {cpath}")
    if not do_blocks:
        return 0

    path = os.path.join(cfg.outdir, "pmf_blocked.dat")
    with open(path, "w") as fh:
        fh.write(f"# block-convergence PMF, M = {M}, aligned on {cfg.region_a}\n")
        fh.write("# xi  mean  sem  " + "  ".join(f"block{m + 1}" for m in range(M)) + "\n")
        fh.writelines(
            f"{grid[i]:10.5f} {mean[i]:12.5f} {sem[i]:10.5f}  " + "  ".join(f"{p[i]:12.5f}" for p in P) + "\n"
            for i in range(grid.size)
        )
    print(f"\n  wrote {path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return 0

    npanel = 3 if do_cum else 2
    fig, ax = plt.subplots(1, npanel, figsize=(5.5 * npanel, 4.2))
    for m, prof in enumerate(P):
        ax[0].plot(grid, prof, lw=0.9, alpha=0.8, label=f"{edges[m]:.0f}-{edges[m + 1]:.0f} ps")
    ax[0].plot(grid, mean, lw=2, color="k", label="mean")
    ax[0].fill_between(grid, mean - sem, mean + sem, color="k", alpha=0.15)
    for r, c in ((cfg.region_a, "C2"), (cfg.region_b, "C3")):
        ax[0].axvspan(r[0], r[1], color=c, alpha=0.10)
    ax[0].set_xlabel(r"$\xi$")
    ax[0].set_ylabel(f"PMF / {u}")
    ax[0].legend(fontsize=7)
    ax[1].errorbar(np.arange(1, M + 1), dg, yerr=dg_sem, marker="o", capsize=3)
    ax[1].axhline(dg_mean, ls="--", color="k", lw=1)
    ax[1].fill_between([0.5, M + 0.5], dg_mean - dg_sem, dg_mean + dg_sem, color="k", alpha=0.12)
    ax[1].set_xlim(0.5, M + 0.5)
    ax[1].set_xticks(np.arange(1, M + 1))
    ax[1].set_xlabel("block (time order)")
    ax[1].set_ylabel(f"dG / {u}")
    ax[1].set_title(f"slope p = {p_slope:.3f},  rank p = {p_rank:.3f}", fontsize=9)
    if do_cum:
        ax[2].plot(cum_t / 1000, cum_dg, marker="o", color="C1")
        ax[2].axhline(cum_dg[-1], ls="--", color="k", lw=1)
        ax[2].fill_between(
            [cum_t[0] / 1000, cum_t[-1] / 1000],
            cum_dg[-1] - cfg.chem_accuracy,
            cum_dg[-1] + cfg.chem_accuracy,
            color="k",
            alpha=0.10,
            label=f"+/- {cfg.chem_accuracy:g} {u}",
        )
        ax[2].set_xlabel("simulation time used per window / ns")
        ax[2].set_ylabel(f"dG / {u}")
        ax[2].set_title("cumulative (0 to t)", fontsize=9)
    fig.tight_layout()
    figure_path = os.path.join(cfg.outdir, "pmf_block_convergence.pdf")
    fig.savefig(figure_path, dpi=600)
    print(f"  wrote {figure_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

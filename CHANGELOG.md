# Changelog

## 0.2.0
- Restructured the new-version scripts into an installable `bilesalt` package with a CLI.
- Completed `MoleculeManager` (itp/gro handling and insertion) that `System.setup` already depended on.
- Backmapping (`cg2at`) answers, overlap and executable are now configurable; no `os.chdir`.
- Slurm resources, partitions, notification address and GROMACS environment are configurable.
- Stage MDP files are parsed with inline-comment support and `-`/`_` key equivalence; control keys no longer leak into MDP files.
- Fixed: nearest-atom search for the pull reference never updated; the pull reference is now an atom number.
- Fixed: `HBA_RES` counted the wrong array; `contactAnalysis` produced an empty table.
- Busy-wait loops replaced by polling with sleep; failed `mdrun` raises a clear error after checkpoint restarts.
- Added tests (unit, mocked backmapping, real GROMACS integration) and a self-contained `bilesalt demo`.

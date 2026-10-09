# Architecture

```
PipelineConfig (config.py) ──► Pipeline (pipeline.py)
                                   │  shared PipelineState: structure, trajectory, tpr, umbrella frames
                                   ▼
                               System (system.py) ── one per `sys` block
    ┌──────────────┬──────────────┼───────────────┬────────────────┐
MoleculeManager  BoxManager   Backmapper     TopologyBuilder    Stage[]
(insert, itp)   (solvate,     (cg2at)        (topol.top)        (mdp → grompp → mdrun,
                 ionise,                                          or Slurm umbrella fan-out)
                 align)
                          all GROMACS calls go through Gromacs (gromacs.py) → shell.run
```

- **State hand-off.** `PipelineState` replaces the old shared `latestFile` list: each step reads
  `state.structure` and updates it, so a CG system's backmapped output feeds the atomistic system.
- **One command boundary.** Every external program runs through `bilesalt.shell.run`, which logs the command and
  raises `CommandError` with the stderr tail, so failures stop the pipeline at the offending step.
- **Setup order** (`System.setup`): create directories → copy force field/molecules → insert molecules →
  [solvate → backmap] or [align → elongate → solvate] → index, component count, topology → `genion` →
  index, component count, topology again from the neutralised structure.
- **Stages** copy their MDP template, apply overrides (control keys such as `US`, `lipid`, `drug` never reach the
  MDP), run `grompp`, and run `mdrun` with checkpoint restarts. Umbrella stages build Slurm scripts and poll
  for the expected per-window outputs.
- **Analysis** is a registry of `command(context, options)` functions; the plan runner extracts curves
  from the written `.xvg`/pickles and plots them.
- **Extending**: add an MDP to `data/mdp` to get a new stage type, or a function to
  `analysis/commands.py:COMMANDS` to get a new analysis.

# Job file reference

A job file is a sequence of blocks. Each block starts with a header line, contains `key = value` lines,
and ends with `end`. Lines starting with `#` are ignored. Block headers: `sys`, `molecule_<n>`,
`stage` (or `stage_<n>`), `slurm`. Systems run in file order and hand their final structure to the next.

## `sys`

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | System name; also the setup directory name |
| `input_dir` | yes | Contains `water.gro` and molecule `.gro` / `.itp` files (looked up as `<phase>/<name>.*`, then `<name>.*`) |
| `output_dir` | yes | Parent of the system and stage directories |
| `forcefield` | yes | Force-field directory, copied to `setup/forcefield` |
| `idxGRPS` | yes | Index groups to build: `NAME:['RES1','RES2'] NAME2:[...]` |
| `cg2at` | yes | `CG` (backmap the structure) or `AA` (already atomistic) |
| `resolvate` | no | Solvate the system (CG water for `CG`, `water_model` for `AA`) |
| `US` | no | Directory of a previous run: its `MD_AA/MD_AA.gro` and `AA/setup/index.ndx` seed this system. For `AA` systems with `resolvate`, the structure is also aligned to z and elongated |
| `boxSize`, `boxDist` | no | Box for the first insertion (nm) and long-axis length used when elongating |
| `Wradius` | no | `gmx solvate -radius` for CG water |
| `water_model` | no | Atomistic water box (default `spc216`) |
| `positive_ion`, `negative_ion` | no | Ion names for `gmx genion` (default `NA`, `CLA`) |
| `gmx_path` | no | GROMACS executable (default `gmx`) |
| `ntomp`, `nb` | no | `mdrun` threads and non-bonded device (default `12`, `gpu`) |
| `cg2at_executable`, `cg2at_answers`, `cg2at_overlap` | no | Backmapping controls (defaults `cg2at`, `12 10 2`, `0.25`) |
| `peptide_residues`, `membrane_residues` | no | Residue names used to align peptide→membrane along z |

## `molecule_<n>`

`system`, `name`, `nmol`, `insertion_radius` are required. `insert = True` inserts `nmol` copies with
`gmx insert-molecules` (optionally `replace = SOL`); `packmol = True` uses packmol instead; `itp = True`
copies `<name>.itp` into `setup/itp` and includes it in the topology. Counts are refreshed from the
final structure, so molecules created by solvation or `genion` only need an entry with `nmol = 0`.

## `stage`

`system` and `type` are required. `type` selects a bundled MDP template (`bilesalt templates`) or, with
`mdp_fname = path`, a custom file. Every other key overrides the matching MDP option
(`ref_t = 298 298`, `tc-grps = MM SOLV`, `nsteps = 1000`).

Umbrella stages carry `US = True`:

- `CONFIG` (`lipid`, `drug`): measures the lipid–drug COM distance along the preceding pulling
  trajectory and extracts one frame per 0.1 nm window up to 11 nm via Slurm.
- Any other `US = True` stage (`EM_US`, `PULL_EQUIL`, `PULL_US`): runs the template on every window
  through Slurm; `pull_group2_pbcatom` is set to the atom of group `MM` nearest its centre of mass.

## `slurm`

`cpu_partition`, `gpu_partition`, `mail_user`, `gmxrc`, `cpus`, `memory`, `modules_purge`, `frame_jobs`,
`umbrella_jobs`, `poll_seconds`. Nothing is hard-coded; unset values are simply omitted from `sbatch`.

## Analysis plans

```
system
resultFolder = /path/to/run
end

stage_1
name = MD_AA
analysis = clusOverTime extractLargest
clusOverTime = CMPL CMPL
extractLargest = 25
end
```

`name` is the stage directory; `analysis` lists the commands; a key named after a command carries that
command's options.

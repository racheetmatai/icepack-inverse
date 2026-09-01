# Backup and artifact policy

Last updated: 1 September 2026

This repository contains the source, notebooks, lightweight manifests, audit
records, and manuscript-support workflow for the JOG revision. Ordinary Git is
not used for generated model ensembles, inversion CSV exports, meshes,
temporary caches, multi-gigabyte CUDA results, or external raster products.

Those excluded artifacts are preserved in the dated Windows backup rooted at
`D:\JOG_BACKUP_20260901`. Its `BACKUP_README.md`, file inventory, and checksums
describe the copied Windows workspace and the Docker-side untracked-artifact
inventory. The production CUDA archive also carries its own internal inventory
and passed end-to-end verification; see
`F:\Codex\JOG\cuda_results\PRODUCTION_TRANSFER_VERIFICATION_20260828.md`.

External geophysical inputs are cited and recoverable from their authoritative
archives:

- ADMAP2S Antarctic magnetic anomaly, Eagles et al. (2024), PANGAEA DOI
  `10.1594/PANGAEA.965433`.
- AntGG2021 Antarctic gravity anomaly/height anomaly grids, Scheinert et al.
  (2024), PANGAEA DOI `10.1594/PANGAEA.971238`.

The Docker repository's `.gitignore` makes this separation explicit. Files are
excluded because they are generated, externally sourced, or too large for
ordinary GitHub storage—not because they are scientifically disposable.

For project context and the exact resume point, read
`PROJECT_HANDOFF_FOR_LLM.md` in this repository and
`F:\Codex\JOG\README_FOR_NEXT_LLM.md` in the Windows workspace.


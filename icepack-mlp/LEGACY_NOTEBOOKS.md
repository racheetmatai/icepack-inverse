# Status of historical training code

The notebooks and `mlp_ensemble.py` in this folder are retained as historical provenance for the published-paper workflow. They are not the accepted execution path for the revision.

The revision supersedes their preprocessing and fitting behavior because the historical code can scale before splitting, reuse a fixed split seed, create a second implicit validation split, rely on an unrecorded shuffle, monitor penalized validation loss, and reject models post hoc by score. The manifest-driven implementation in `production_training/` removes those ambiguities while retaining the approved MLP family.

Do not edit the historical notebooks to create revision production runs. Use `README_PRODUCTION_TRAINING.md` and the frozen Gate 2 registries.

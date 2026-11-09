# Training companion review

The repository was reorganized on 2026-11-09 after the intended emphasis was clarified: publish the training implementation and evidence associated with the existing HF models, rather than reconstruct or rerun the original training study. The optional inference integration remains secondary.

## Delivered scope

The training subtree includes 76 Python modules: all 23 mammography configs, their dynamic model/dataset/task consumers, the local static import closure, original runner, losses/metrics, optimizer/scheduler and EMA callbacks. A source manifest records origins and SHA-256 values. Whitespace-only normalization is recorded where applied; shared implementation logic was not modernized. The standalone training package metadata uses namespace discovery so nested modules are included.

Three evaluation notebooks preserve source with outputs cleared. Historical numerical result cells and the nine associated run IDs are recorded separately. No private run pickles, model weights, patient tables, image outputs or machine audit records are published. Model-code provenance for the optional inference adapter remains separately documented.

## Verification

- Four training-focused checks pass: source hashes/syntax and cleared notebook outputs; internal imports and recipe model/dataset/task module coverage; recovered multi-task loss with independent NumPy expected values and missing-label gradient checks; recorded result consistency.
- The independent reviewer confirmed the notebook numerical evidence matches upstream, config references resolve to included code, and launch examples match the preserved interface with adaptation requirements clearly stated.
- The separate training wheel includes all 76 source modules. The root source distribution includes training source/notebooks and excludes generated build directories, weights and private records.
- The 12 optional-inference tests and integration lint checks continue to pass.
- Independent assessment: no material findings within the training-companion scope.

## Limits

No full training dependency environment, training lifecycle, GPU run, new evaluation cohort, original data/splits, or HF-export tensor parity was recreated. The source is a recovered framework/config snapshot, not an exact historical dirty revision or locked execution environment. The crop config is a candidate with batch size 128 versus 256 in the published model card; exact final crop run identity is unverified. These limits are stated in the training guide and model mapping.

Original preservation directories remain local and ignored. No original archive was moved or deleted. Nothing was committed, pushed or published during this reorganization.

## Subsequent deep review

A new fresh-context read-only review traced the training consumers and notebook code against the training-companion scope. The initial staged tree for this pass was `bf4bba973b127eb77bc5bf07685609272fcc25d4` on branch main (no base commit or remote).

One reader-facing issue was confirmed: `eval_automodel.ipynb` contains unfinished copied ensemble tail cells after its single-model workflow. Running all cells would reference undefined model variables and later use an incompatible `run_test` argument. The notebook now opens with a warning; original code is preserved. `training/notebooks/README.md` identifies each notebook's scope, and the authoritative historical results remain in `eval_model.ipynb` and evaluation-evidence.json. The source-manifest downstream modification/hash is updated.

All 23 configs were imported with the existing dependencies. On CPU, the recovered screening model completed a reduced-size 32×32 synthetic forward/loss/backward check and the recovered crop model completed a sigmoid-coordinate forward check. Pretrained downloads were disabled and no optimizer steps or training run were launched. These checks exercise the actual model implementations but do not validate the full Lightning/Neptune/data lifecycle.

Full training runtime dependencies remain unavailable in the review environment (including Lightning, Neptune, MONAI and torcheval). No dependencies were installed solely for review. Distributed/GPU execution, original data and split membership, and original tensor parity remain outside verified scope. Earlier packaging and provenance boundaries remain applicable.

## Data preparation source addition

Thirteen scripts/notebooks from i-pan/datasets-src revision `74ad10027bed59759af119f8832f6fa6f45a2aec` were added after locating the original processing archive. A separate Apache-2.0 license and source manifest record origins and sanitization. Original code cells are retained, outputs/attachments/widgets removed. Independent review confirmed the manifests and public-data boundary.

The preserved crop-label notebook has an inconsistent pixel/normalized scaling section; the mask notebook relabels prior foreground across lesions. Both carry prominent warning markdown and are not advertised as validated Run All pipelines. The preparation guide also documents original split RNG/size caveats and missing initial DDSM PNG/cleaned-CSV generation. No dataset conversion, split generation, model inference, or dataset write was launched while recovering these scripts.

## Corrected preparation utilities review

New utilities under training/data_preparation/corrected are separate reference implementations; historical source and manifests remain unchanged. Crop coordinates are returned in original-image pixels with border offset restored. Mask drawing uses a per-lesion region, fixed category IDs, and explicit later-row overlap priority. The original notebook's yx coordinate mapping is the documented default, with an explicit xy alternative; physical boxes were not visually validated against the full dataset.

Independent review found a canonical-path grouping defect: different spellings of the same output path could overwrite an earlier lesion group. Regressions failed on the old implementation; normalized grouping and exclusive lossless PNG writes resolve it. All six focused tests pass, including both coordinate orders and alias handling. The reviewer found no remaining material findings within this scope. Existing output refusal and partial-completion behavior are documented. No real data preparation or model evaluation was launched.

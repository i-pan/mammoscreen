# Initial inference implementation review

Historical scope: this review preceded the training-first reorganization. The training companion is covered separately in TRAINING_REVIEW.md; the conclusions below apply to the optional inference integration.

Reviewed on 2026-11-09: initial uncommitted implementation on branch `main`, with no base commit or remotes. Scope included staged and working-tree runtime code, CLI, model implementations, image loading, packaging, tests, documentation and training provenance. The final code was challenged by an independent read-only reviewer.

## Findings resolved

1. Multi-page TIFFs silently selected their first page through OpenCV. The loader now rejects multi-page rasters, matching the single-image contract. A two-page TIFF regression failed before the fix and passed afterward.
2. The square screening network inherited version-dependent Albumentations padding. Original saved run `b386692c` records `PadIfNeeded` border mode 4 (reflection), while Albumentations 2 defaults to constant padding. The model now selects reflection explicitly. A pixel-level regression against an independent OpenCV oracle failed before the fix and passed afterward. Upstream and patched model hashes are recorded separately.

Additional verification exercises actual model probability aggregation across unequal view counts, breast ordering, and CLI serialization/loading options.

## Final verification

- All 12 tests pass; integration Ruff checks pass.
- Wheel and source distributions build; wheel imports succeed in isolated Python mode.
- Distribution inspection excludes model weights, run pickles and raw/private recovery archives.
- Real CLI inference from the built wheel succeeds offline using both pinned preserved checkpoints on a synthetic image, returning finite cancer and normalized density probabilities.
- Model provenance hashes and historical config Python syntax were checked.
- Independent final review: no remaining findings within the reviewed scope.

Runtime/test/package fingerprint (SHA-256 over sorted relative paths and bytes of Python under src/tests plus pyproject.toml): `c4fb9b532f9e5fb294976d56131ca75d9fea5b6bcf1f41cea9e23b54f8e90061`.

## Limits

CUDA was unavailable. No clinical cohort evaluation, calibration validation, original EMA-versus-HF tensor comparison, dependency-version matrix, or remote publication was performed. The synthetic check establishes execution, not diagnostic performance. Broad dependency ranges are not a claim of testing every supported version. Historical training configs remain recovery evidence with unavailable original data paths.

Changes are prepared for review; no commit, push or deployment was performed during this review. See VALIDATION.md for tested dependency versions.

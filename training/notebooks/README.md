# Historical evaluation notebooks

| Notebook | Purpose and scope |
| --- | --- |
| `eval_model.ipynb` | The three original local EMA ensemble evaluations; source of the separately preserved numerical results. Requires the original data and weights to execute. |
| `eval_automodel.ipynb` | Unfinished HF evaluation scratchpad. Its initial single-model workflow is followed by copied ensemble cells with undefined variables and an incompatible `run_test` call. Annotated in the notebook; not a Run All recipe. |
| `test_model.ipynb` | Exploratory model/dataset inspection using historical paths, not a full evaluation recipe. |

Code cells retain their upstream source; outputs and execution counters are cleared. See `../evaluation-evidence.json` for recorded result cells from `eval_model.ipynb`. That numerical evidence was not rerun during recovery.

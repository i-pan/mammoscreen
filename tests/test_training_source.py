"""Check the training companion without launching training or unpickling records."""

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_training_manifest_and_notebook_output_boundary():
    manifest = json.loads((ROOT / "training/source-manifest.json").read_text())
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        if path.suffix == ".py":
            ast.parse(path.read_text())
        elif path.suffix == ".ipynb":
            for cell in json.loads(path.read_text())["cells"]:
                assert not cell.get("outputs")


def test_training_internal_imports_and_dynamic_recipe_modules_exist():
    base = ROOT / "training/skp"

    def exists(module):
        path = base.joinpath(*module.split(".")[1:])
        return path.is_dir() or path.with_suffix(".py").is_file()

    for path in base.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            for module in modules:
                if module == "skp" or module.startswith("skp."):
                    assert exists(module), (path, module)
    for path in (base / "configs/mammo").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                        and target.value.id == "cfg"):
                    continue
                if target.attr not in {"model", "dataset", "task"}:
                    continue
                value = ast.literal_eval(node.value)
                section = {"model": "models", "dataset": "datasets", "task": "tasks"}[target.attr]
                assert exists(f"skp.{section}.{value}")


def test_recovered_screening_loss_masks_missing_auxiliary_labels(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "training"))
    from skp.losses.custom import MammoCancerWithAuxLosses

    logits = torch.linspace(-1.5, 1.5, 22).reshape(2, 11).requires_grad_()
    targets = torch.tensor([[1, 0, 1, 0, 2, 3], [0, 1, 0, 1, -1, -1]])
    weights = [1, 2, 3, 4, 5, 6]
    result = MammoCancerWithAuxLosses({"loss_weights": weights})(
        {"logits": logits}, {"y": targets}
    )
    # Independent NumPy oracle for the six weighted loss terms.
    raw = logits.detach().numpy().astype(np.float64)
    bce = np.logaddexp(0, raw[:, :4]) - raw[:, :4] * targets[:, :4].numpy()
    expected = (bce.mean(axis=0) * weights[:4]).sum()
    for columns, target, weight in [(slice(4, 7), 2, 5), (slice(7, 11), 3, 6)]:
        values = raw[0, columns]
        expected += weight * (np.log(np.exp(values).sum()) - values[target])
    np.testing.assert_allclose(result["loss"].detach().numpy(), expected, rtol=1e-6)
    result["loss"].backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[1, 4:].any()


def test_recorded_evaluation_results_match_published_summary():
    evidence = json.loads((ROOT / "training/evaluation-evidence.json").read_text())
    assert len(evidence["runs"]) == 9
    assert evidence["auc_by_split"] == [0.9464, 0.9467, 0.9422]
    for value, cell in zip(evidence["auc_by_split"], evidence["result_cells"]):
        assert f"AUC: {value:.4f}" in cell["stdout"]


def test_data_preparation_provenance_and_notebook_sanitization():
    manifest = json.loads((ROOT / "training/data_preparation/source-manifest.json").read_text())
    assert len(manifest["files"]) == 13
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["published_sha256"]
        if path.suffix == ".py":
            ast.parse(path.read_text())
        else:
            notebook = json.loads(path.read_text())
            assert "widgets" not in notebook["metadata"]
            for cell in notebook["cells"]:
                assert not cell.get("outputs")
                assert not cell.get("attachments")
                assert not cell["metadata"]

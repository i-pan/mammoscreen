"""Bounded checks of the active logger/launcher, without launching training."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_local_logger_channels_and_nonzero_rank(tmp_path, monkeypatch):
    pytest.importorskip('lightning')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'training'))
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
    from mammoscreen_logging import LocalTrainingLogger
    import torch

    monkeypatch.setenv('MAMMOSCREEN_LOG_DIR', str(tmp_path))
    logger = LocalTrainingLogger(project='fixture', mode='debug')
    logger.log_hyperparams({'x': 2})
    logger.log_metrics({'loss': torch.tensor(0.25)}, step=7)
    logger.experiment['cfg'] = {'fold': 1}
    logger.experiment['val/auc'].append(0.75)
    logger.finalize('success')
    logger.experiment.stop()
    events = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert [(e['key'], e['value']) for e in events[:4]] == [
        ('hyperparameters', {'x': 2}), ('loss', 0.25), ('cfg', {'fold': 1}), ('val/auc', 0.75)]
    assert events[1]['step'] == 7
    monkeypatch.setattr(rank_zero_only, 'rank', 1)
    logger.log_metrics({'loss': 99})
    logger.experiment['training/ema_decay'].append(0.99)
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == len(events)


def test_training_adapter_preserves_ddp_relaunch_argv(monkeypatch):
    pytest.importorskip('lightning')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'training'))
    recipe = importlib.import_module('mammoscreen_recipe')
    from mammoscreen_logging import LocalTrainingLogger

    launch = ['mammoscreen_recipe.py', '_train', 'mammo.cfg_recipe_screen', '--devices', '2']
    monkeypatch.setattr(sys, 'argv', launch)
    observed = {}
    runner = SimpleNamespace(parse_args=lambda: list(sys.argv))

    def main():
        observed['parsed'] = runner.parse_args()
        observed['launch'] = list(sys.argv)
        observed['logger'] = runner.NeptuneLogger

    runner.main = main
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda module: None))
    monkeypatch.setattr(recipe.importlib.util, 'spec_from_file_location', lambda *args: spec)
    monkeypatch.setattr(recipe.importlib.util, 'module_from_spec', lambda spec: runner)
    recipe.train_locally(launch[2:])
    assert observed['parsed'] == [launch[0], *launch[2:]]
    assert observed['launch'] == launch
    assert observed['logger'] is LocalTrainingLogger

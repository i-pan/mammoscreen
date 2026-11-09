"""Verify split identity and actual config/command consumers without training."""

import argparse
import importlib.util
import json
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'training'))
    from data_preparation.corrected import splits
    spec = importlib.util.spec_from_file_location('mammoscreen_recipe', ROOT / 'training/mammoscreen_recipe.py')
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    import sys
    monkeypatch.setitem(sys.modules, 'mammoscreen_recipe', recipe)
    return splits, recipe


def raw_rows():
    return [dict(patient_id=str(p), image_id=f'{p}-{v}', laterality='L', view=v,
                 cancer=str(p % 2), biopsy=str(p % 2), invasive='0',
                 difficult_negative_case='False', density='B', BIRADS='1')
            for p in range(100) for v in ['CC', 'MLO']]


def plan(tmp_path, modules):
    splits, recipe = modules
    raw = tmp_path / 'train.csv'
    splits.write_csv(raw, raw_rows())
    directory = tmp_path / 'splits'
    splits.main(['--annotations', str(raw), '--output', str(directory), '--folds', '5'])
    output = tmp_path / 'recipe.json'
    recipe.main(['plan', '--splits', str(directory / 'splits.json'),
                 '--images', str(tmp_path / 'images'), '--ddsm-images', str(tmp_path / 'ddsm'),
                 '--ddsm-masks', str(tmp_path / 'masks'), '--ddsm-annotations', str(tmp_path / 'ddsm.csv'),
                 '--runs', str(tmp_path / 'runs'), '--output', str(output), '--project', 'user/test'])
    return output


def test_local_rng_distinct_patient_isolation_and_determinism(modules):
    splits, _ = modules
    state = random.getstate()
    a = splits.make_splits(raw_rows(), seed=9, folds=5)
    assert random.getstate() == state
    assert a == splits.make_splits(raw_rows(), seed=9, folds=5)
    held = [frozenset(splits.patient_ids(h)) for _, h in a]
    assert len(set(held)) == 3
    for train, holdout in a:
        assert not splits.patient_ids(train) & splits.patient_ids(holdout)
        for fold in range(5):
            validation = [r for r in train if r['fold'] == fold]
            other = [r for r in train if r['fold'] != fold]
            assert not splits.patient_ids(validation) & splits.patient_ids(other)
        assert {r['density'] for r in train} == {1}
        assert {r['BIRADS'] for r in train if r['biopsy']} == {0}


def test_all_members_route_to_matching_set_and_generated_checkpoints(tmp_path, modules, monkeypatch):
    splits, recipe = modules
    path = plan(tmp_path, modules)
    data = recipe.read_recipe(path)
    for partition in data['sets']:
        for name in recipe.MEMBERS:
            cfg = recipe.configuration(data, name, partition['id'])
            assert cfg.annotations_file == partition['training_csv']
            assert cfg.data_dir == data['images']
            assert cfg.ema['on'] and not cfg.pretrained
            assert cfg.load_pretrained_backbone == data['pretraining'][name]['checkpoint']
            if name == '1536':
                import albumentations as A
                assert all(t.border_mode == 4 for t in cfg.val_transforms.transforms
                           if isinstance(t, A.PadIfNeeded))
            assert Path(cfg.load_pretrained_backbone).name == 'last.ckpt'
    calls = []
    Path(data['images']).mkdir()
    for job in data['pretraining'].values():
        p = Path(job['checkpoint']); p.parent.mkdir(parents=True); p.touch()
    monkeypatch.setattr(recipe.subprocess, 'run', lambda args, **kwargs: calls.append((args, kwargs)))
    recipe.run_jobs(argparse.Namespace(recipe=path, stage='screening', set=2, devices=1, member=None))
    assert len(calls) == 3
    for args, options in calls:
        assert options['env']['MAMMOSCREEN_SET'] == '2'
        assert options['env']['MAMMOSCREEN_MEMBER'] in recipe.MEMBERS
        assert '--keep_pretrained_weights_path' in args
        assert args[1:4] == ['-m', 'mammoscreen_recipe', '_train']
        assert args[4] == recipe.SCREEN_CONFIG
        assert options['env']['MAMMOSCREEN_LOG_DIR'].endswith('/logs')
    # Import the actual config wrapper with the exact launch environment.
    import importlib
    for key in ['MAMMOSCREEN_RECIPE', 'MAMMOSCREEN_SET', 'MAMMOSCREEN_MEMBER']:
        monkeypatch.setenv(key, calls[1][1]['env'][key])
    active = importlib.import_module('skp.configs.mammo.cfg_recipe_screen').cfg
    assert active.annotations_file == data['sets'][1]['training_csv']
    assert active.load_pretrained_backbone == data['pretraining']['1920']['checkpoint']
    wrong = json.loads(path.read_text())
    wrong['sets'][0]['training_csv'] = wrong['sets'][1]['training_csv']
    path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match='wrong split'):
        recipe.read_recipe(path)


def test_csv_tampering_and_patient_leakage_fail(tmp_path, modules):
    splits, recipe = modules
    path = plan(tmp_path, modules)
    data = recipe.read_recipe(path)
    table = Path(data['sets'][0]['holdout_csv'])
    table.write_text(table.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        recipe.read_recipe(path)
    train, holdout = splits.make_splits(raw_rows(), folds=5)[0]
    holdout[0]['patient_id'] = train[0]['patient_id']
    with pytest.raises(ValueError, match='leakage'):
        splits.validate_partition(train, holdout, 5)


def test_auc_uses_mean_probabilities_per_breast_and_ties(modules):
    _, recipe = modules
    rows = [dict(breast_id='a', cancer=1, prediction=x) for x in [0.1, 0.9]]
    rows += [dict(breast_id='b', cancer=0, prediction=0.6),
             dict(breast_id='c', cancer=1, prediction=0.7),
             dict(breast_id='d', cancer=0, prediction=0.5)]
    result = recipe.summarize(rows)
    # Pairwise oracle: a wins0, ties1; c wins2 => 2.5/4.
    assert result == {'cancer_auc': 0.625, 'breasts': 4, 'images': 5}


def test_evaluation_consumes_generated_ema_and_writes_cohort_outputs(tmp_path, modules, monkeypatch):
    import albumentations as A
    import cv2
    import math
    import numpy as np
    import torch
    from skp.models.classification import net2d

    splits, recipe = modules
    path = plan(tmp_path, modules)
    data = recipe.read_recipe(path)
    partition = data['sets'][0]
    for row in splits.read_rows(partition['holdout_csv']):
        image = Path(data['images']) / row['filename']
        image.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(image), np.zeros((16, 8), dtype=np.uint8))

    class TinyNet(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.zeros(11))

        def forward(self, batch):
            return {'logits': self.bias[None].expand(batch['x'].shape[0], -1)}

    monkeypatch.setattr(net2d, 'Net', TinyNet)
    original_config = recipe.configuration

    def small_config(*args, **kwargs):
        cfg = original_config(*args, **kwargs)
        cfg.val_transforms = A.Resize(8, 8)
        return cfg

    monkeypatch.setattr(recipe, 'configuration', small_config)
    for bias, member in zip([-3., -1., 2.], partition['members'].values()):
        checkpoint = Path(member['checkpoint'])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': {'model.bias': torch.full((11,), bias)}}, checkpoint)
    output = tmp_path / 'evaluation'
    recipe.evaluate(argparse.Namespace(recipe=path, set=1, output=output, device='cpu'))
    predictions = splits.read_rows(output / 'image_predictions.csv')
    expected = sum(1 / (1 + math.exp(-x)) for x in [-3, -1, 2]) / 3
    assert all(float(row['prediction']) == pytest.approx(expected) for row in predictions)
    metrics = json.loads((output / 'metrics.json').read_text())
    assert metrics['cancer_auc'] == 0.5
    assert metrics['breasts'] * 2 == metrics['images'] == len(predictions)
    assert metrics['recipe_sha256'] == splits.sha256(path)
    assert set(metrics['checkpoint_sha256']) == {
        m['checkpoint'] for m in partition['members'].values()
    }
    with pytest.raises(FileExistsError):
        recipe.evaluate(argparse.Namespace(recipe=path, set=1, output=output, device='cpu'))

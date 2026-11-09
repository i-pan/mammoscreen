"""Opt-in CPU lifecycle check with real models and tiny synthetic data.

Run MAMMOSCREEN_RUN_LIFECYCLE=1 pytest -q tests/test_training_lifecycle.py
in the training environment. No datasets or pretrained downloads are used.
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('MAMMOSCREEN_RUN_LIFECYCLE') != '1', reason='Opt-in real training lifecycle')


def test_real_training_initialization_ema_and_evaluation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'training'))
    import albumentations as A
    import cv2
    import mammoscreen_recipe as recipe
    import numpy as np
    import torch
    from data_preparation.corrected import splits

    original_logger = getattr(importlib.import_module('lightning.pytorch.loggers'), 'NeptuneLogger', None)
    torch.set_num_threads(1)
    raw = [{'patient_id': str(p), 'image_id': str(p), 'laterality': 'L', 'view': 'CC',
            'cancer': p % 2, 'biopsy': p % 2, 'invasive': p % 2, 'difficult_negative_case': 0,
            'density': 'ABCD'[p % 4], 'BIRADS': p % 3} for p in range(40)]
    splits.write_csv(tmp_path / 'raw.csv', raw)
    splits.main(['--annotations', str(tmp_path / 'raw.csv'), '--output', str(tmp_path / 'splits'),
                 '--folds', '3', '--repeats', '1'])
    images, masks, ddsm = [tmp_path / name for name in ['images', 'masks', 'ddsm']]
    for directory in [images, masks, ddsm]:
        directory.mkdir()
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    for row in raw:
        (images / row['patient_id']).mkdir()
        assert cv2.imwrite(str(images / row['patient_id'] / (row['image_id'] + '.png')), image)
    labels = ['benign_calcification', 'benign_mass', 'malignant_calcification', 'malignant_mass']
    ddsm_rows = []
    for i in range(8):
        filename = f'{i}.png'
        assert cv2.imwrite(str(ddsm / filename), image)
        mask = np.zeros_like(image)
        for c in range(4):
            mask[8 + c * 8:16 + c * 8, 16:48] = c + 1
        assert cv2.imwrite(str(masks / filename), mask)
        ddsm_rows.append(dict(filepath=filename, patient_id=i, fold=i % 2,
                              **{label: int((i + c) % 2) for c, label in enumerate(labels)}))
    splits.write_csv(tmp_path / 'ddsm.csv', ddsm_rows)
    path = tmp_path / 'recipe.json'
    recipe.main(['plan', '--splits', str(tmp_path / 'splits/splits.json'),
                 '--images', str(images), '--ddsm-images', str(ddsm), '--ddsm-masks', str(masks),
                 '--ddsm-annotations', str(tmp_path / 'ddsm.csv'), '--runs', str(tmp_path / 'runs'),
                 '--output', str(path)])
    original_configuration = recipe.configuration

    def bounded_configuration(*args, **kwargs):
        cfg = original_configuration(*args, **kwargs)
        cfg.pretrained = False  # Never download ImageNet weights in this check.
        cfg.image_height = cfg.image_width = 64
        cfg.train_transforms = cfg.val_transforms = A.Compose([A.Resize(64, 64)])
        cfg.num_workers, cfg.persistent_workers = 0, False
        cfg.num_epochs, cfg.batch_size, cfg.val_batch_size = 1, 2, 2
        return cfg

    monkeypatch.setattr(recipe, 'configuration', bounded_configuration)
    monkeypatch.setenv('MAMMOSCREEN_RECIPE', str(path))
    monkeypatch.setenv('MAMMOSCREEN_SET', '1')
    data = recipe.read_recipe(path)
    for pretraining in [True, False]:
        config = recipe.DDSM_CONFIG if pretraining else recipe.SCREEN_CONFIG
        for name in recipe.MEMBERS:
            job = data['pretraining'][name] if pretraining else data['sets'][0]['members'][name]
            monkeypatch.setenv('MAMMOSCREEN_MEMBER', name)
            monkeypatch.setenv('MAMMOSCREEN_LOG_DIR', str(Path(job['checkpoint']).parent.parent / 'logs'))
            if not pretraining:
                from skp.models.classification.net2d import Net
                cfg = bounded_configuration(data, name, 1)
                initialized = Net(cfg)
                saved = torch.load(data['pretraining'][name]['checkpoint'], weights_only=True)['state_dict']
                torch.testing.assert_close(initialized.backbone.conv_stem.weight,
                                           saved['model.segmenter.encoder.conv_stem.weight'])
                del initialized, saved
            # A production job runs in a fresh process; clear only the active wrapper here.
            monkeypatch.delitem(sys.modules, 'skp.configs.' + config, raising=False)
            argv = ['_train', config, '--run_id', job['run_id'], '--kfold',
                    str(0 if pretraining else job['fold']), '--keep_pretrained_weights_path',
                    '--devices', '1', '--accelerator', 'cpu', '--strategy', 'auto',
                    '--precision', '32-true', '--no_sync_batchnorm', '--no_benchmark',
                    '--limit_train_batches', '2' if pretraining else '3', '--limit_val_batches', '2',
                    '--log_every_n_steps', '1']
            monkeypatch.setattr(sys, 'argv', [str(recipe.ROOT / 'mammoscreen_recipe.py'), *argv])
            recipe.main(argv)
            assert Path(job['checkpoint']).is_file()
            assert (Path(job['checkpoint']).parent.parent / 'logs/events.jsonl').is_file()
            if not pretraining:
                state = torch.load(job['checkpoint'], weights_only=True)['state_dict']
                assert state['model.linear.weight'].shape[0] == 11
                raw_state = torch.load(Path(job['checkpoint']).with_name('last.ckpt'),
                                       weights_only=True)['state_dict']
                assert not torch.equal(state['model.linear.weight'], raw_state['model.linear.weight'])
                del state, raw_state
    # All three actual generated EMA payloads must reach the real evaluator.
    recipe.evaluate(argparse.Namespace(recipe=path, set=1, output=tmp_path / 'evaluation', device='cpu'))
    assert (tmp_path / 'evaluation/metrics.json').is_file()
    # Restore either the original export (older Lightning) or its absence (new releases).
    assert getattr(importlib.import_module('lightning.pytorch.loggers'),
                   'NeptuneLogger', None) is original_logger

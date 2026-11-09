"""Explicit split -> DDSM initialization -> three-model EMA -> evaluation recipe."""

import argparse
import copy
import csv
import importlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from data_preparation.corrected.splits import patient_ids, read_rows, sha256, validate_partition

ROOT = Path(__file__).resolve().parent
MEMBERS = {
    '2048': ('mammo.cfg_ddsm_pretrained_aux_losses', 'mammo.cfg_ddsm_seg_cls_v02', 0),
    '1920': ('mammo.cfg_ddsm_pretrained_aux_losses_v02', 'mammo.cfg_ddsm_seg_cls_v03', 1),
    '1536': ('mammo.cfg_ddsm_pretrained_aux_losses_v04', 'mammo.cfg_ddsm_seg_cls_v05', 2),
}
SCREEN_CONFIG = 'mammo.cfg_recipe_screen'
DDSM_CONFIG = 'mammo.cfg_recipe_ddsm'


def read_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if recipe['schema_version'] != 1:
        raise ValueError('Unsupported recipe version')
    manifest_path = Path(recipe['splits_manifest'])
    if sha256(manifest_path) != recipe['splits_manifest_sha256']:
        raise ValueError('Split manifest changed since recipe creation')
    manifest = json.loads(manifest_path.read_text())
    if manifest['folds'] < 3 or len(manifest['sets']) != len(recipe['sets']):
        raise ValueError('Recipe requires at least three folds and matching split sets')
    if list(recipe['pretraining']) != list(MEMBERS):
        raise ValueError('Missing or reordered DDSM initialization members')
    for name, initialization in recipe['pretraining'].items():
        expected = Path(recipe['runs']) / DDSM_CONFIG / initialization['run_id'] / (
            'fold0/checkpoints/last.ckpt')
        if Path(initialization['checkpoint']) != expected:
            raise ValueError('Initialization is not the generated DDSM checkpoint')
    previous = set()
    checkpoint_paths = set()
    for partition, recorded in zip(recipe['sets'], manifest['sets']):
        if partition['id'] != recorded['id']:
            raise ValueError('Mismatched split-set IDs')
        for role in ['training', 'holdout']:
            path = Path(partition[role + '_csv'])
            if path != (manifest_path.parent / recorded[role + '_csv']).resolve():
                raise ValueError('Recipe points to the wrong split CSV')
            if sha256(path) != recorded[role + '_sha256']:
                raise ValueError('Split CSV changed since creation')
        train, held = read_rows(partition['training_csv']), read_rows(partition['holdout_csv'])
        validate_partition(train, held, manifest['folds'])
        signature = frozenset(patient_ids(held))
        if signature in previous:
            raise ValueError('Repeated holdout membership')
        previous.add(signature)
        if list(partition['members']) != list(MEMBERS):
            raise ValueError('Recipe must include all three resolutions in canonical order')
        for name, member in partition['members'].items():
            if member['fold'] != MEMBERS[name][2]:
                raise ValueError('Member fold does not match recipe')
            expected = Path(recipe['runs']) / SCREEN_CONFIG / member['run_id'] / (
                f'fold{member["fold"]}/checkpoints/ema_weights.pt')
            if Path(member['checkpoint']) != expected:
                raise ValueError('Member checkpoint is not the generated EMA path')
            if str(expected) in checkpoint_paths:
                raise ValueError('Two members reuse the same output checkpoint')
            checkpoint_paths.add(str(expected))
    return recipe


def build_plan(args):
    manifest_path = args.splits.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['folds'] < 3:
        raise ValueError('Three-member recipe requires at least three validation folds')
    runs = args.runs.resolve()
    plan = {'schema_version': 1, 'splits_manifest': str(manifest_path),
            'splits_manifest_sha256': sha256(manifest_path), 'images': str(args.images.resolve()),
            'runs': str(runs), 'project': args.project,
            'ddsm': {'images': str(args.ddsm_images.resolve()), 'masks': str(args.ddsm_masks.resolve()),
                     'annotations': str(args.ddsm_annotations.resolve())},
            'pretraining': {}, 'sets': []}
    for name in MEMBERS:
        run_id = f'ddsm-{name}'
        checkpoint = runs / DDSM_CONFIG / run_id / 'fold0/checkpoints/last.ckpt'
        plan['pretraining'][name] = {'run_id': run_id, 'checkpoint': str(checkpoint)}
    for part in manifest['sets']:
        members = {}
        for name, (_, _, fold) in MEMBERS.items():
            run_id = f'set{part["id"]:02d}-{name}'
            checkpoint = runs / SCREEN_CONFIG / run_id / f'fold{fold}/checkpoints/ema_weights.pt'
            members[name] = {'run_id': run_id, 'fold': fold, 'checkpoint': str(checkpoint)}
        plan['sets'].append({'id': part['id'], 'training_csv': str(
            (manifest_path.parent / part['training_csv']).resolve()), 'holdout_csv': str(
            (manifest_path.parent / part['holdout_csv']).resolve()), 'members': members})
    return plan


def configuration(recipe, member_name, set_id=None, pretraining=False, evaluation=False):
    """Clone historical values, then bind every active path to the explicit plan."""
    from skp.configs import Config

    screen_base, ddsm_base, fold = MEMBERS[member_name]
    base = importlib.import_module('skp.configs.' + (ddsm_base if pretraining else screen_base)).cfg
    cfg = Config()
    cfg.__dict__.update(copy.deepcopy(vars(base)))
    cfg.project = recipe['project']
    cfg.neptune_mode = 'debug'  # Disable historical offline sync; logging is local.
    cfg.save_dir = recipe['runs']
    cfg.fold = 0 if pretraining else fold
    if pretraining:
        cfg.data_dir = recipe['ddsm']['images']
        cfg.seg_data_dir = recipe['ddsm']['masks']
        cfg.annotations_file = recipe['ddsm']['annotations']
    else:
        partition = next(p for p in recipe['sets'] if p['id'] == set_id)
        cfg.data_dir = recipe['images']
        cfg.annotations_file = partition['training_csv']
        cfg.pretrained = False
        cfg.load_pretrained_backbone = (
            None if evaluation else recipe['pretraining'][member_name]['checkpoint'])
        cfg.ema['on'] = True
    if evaluation:
        cfg.pretrained = False
        cfg.load_pretrained_backbone = None
        cfg.enable_gradient_checkpointing = False
    # Pin the saved square-model contract for both training and evaluation.
    if member_name == '1536':
        import albumentations as A
        import cv2
        for compose in [cfg.train_transforms, cfg.val_transforms]:
            for transform in compose.transforms:
                if isinstance(transform, A.PadIfNeeded):
                    transform.border_mode = cv2.BORDER_REFLECT_101
    return cfg


def config_from_environment(pretraining=False):
    path, member = os.environ.get('MAMMOSCREEN_RECIPE'), os.environ.get('MAMMOSCREEN_MEMBER')
    if not path or member not in MEMBERS:
        raise RuntimeError('Use training/mammoscreen_recipe.py run; this config requires an explicit recipe')
    recipe = read_recipe(path)
    return configuration(recipe, member, int(os.environ.get('MAMMOSCREEN_SET', '1')),
                         pretraining=pretraining)


def run_jobs(args):
    recipe = read_recipe(args.recipe)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    env['MAMMOSCREEN_RECIPE'] = str(args.recipe.resolve())
    if args.stage == 'pretraining':
        jobs = [(None, name, item, 0, DDSM_CONFIG)
                for name, item in recipe['pretraining'].items()]
    else:
        jobs = [(part['id'], name, item, item['fold'], SCREEN_CONFIG)
                for part in recipe['sets'] if args.set is None or part['id'] == args.set
                for name, item in part['members'].items()]
        if not jobs:
            raise ValueError('Unknown evaluation set')
    if args.member is not None:
        jobs = [job for job in jobs if job[1] == args.member]
    required = ([recipe['ddsm']['images'], recipe['ddsm']['masks']]
                if args.stage == 'pretraining' else [recipe['images']])
    if any(not Path(path).is_dir() for path in required):
        raise FileNotFoundError('Prepare the required image/mask directories before training')
    if args.stage == 'pretraining' and not Path(recipe['ddsm']['annotations']).is_file():
        raise FileNotFoundError('Create the DDSM training annotation CSV first')
    # Preflight every selected job before starting any expensive work.
    for _, name, job, _, config in jobs:
        run_directory = Path(recipe['runs']) / config / job['run_id']
        if run_directory.exists():
            raise FileExistsError(f'Refusing to overwrite training run: {run_directory}')
        if args.stage == 'screening' and not Path(recipe['pretraining'][name]['checkpoint']).is_file():
            raise FileNotFoundError('Run matching DDSM pretraining before screening')
    for set_id, name, job, fold, config in jobs:
        run_directory = Path(recipe['runs']) / config / job['run_id']
        if run_directory.exists():
            raise FileExistsError(f'Refusing to overwrite training run: {run_directory}')
        if args.stage == 'screening' and not Path(recipe['pretraining'][name]['checkpoint']).is_file():
            raise FileNotFoundError('Run matching DDSM pretraining before screening')
        job_env = dict(env, MAMMOSCREEN_MEMBER=name, MAMMOSCREEN_SET=str(set_id or 1),
                       MAMMOSCREEN_LOG_DIR=str(Path(job['checkpoint']).parent.parent / 'logs'))
        command = [sys.executable, '-m', 'mammoscreen_recipe', '_train', config,
                   '--run_id', job['run_id'], '--kfold', str(fold),
                   '--keep_pretrained_weights_path', '--devices', str(args.devices)]
        subprocess.run(command, env=job_env, cwd=ROOT, check=True)


def summarize(predictions):
    """Mean probabilities per breast, then rank-based binary AUC with ties."""
    groups = defaultdict(list)
    for row in predictions:
        value, label = float(row['prediction']), int(row['cancer'])
        if not math.isfinite(value) or not 0 <= value <= 1 or label not in {0, 1}:
            raise ValueError('Invalid prediction or target')
        groups[row['breast_id']].append((value, label))
    breasts = []
    for rows in groups.values():
        if len({label for _, label in rows}) != 1:
            raise ValueError('Inconsistent labels within a breast')
        breasts.append((sum(p for p, _ in rows) / len(rows), rows[0][1]))
    positives = sum(y for _, y in breasts)
    negatives = len(breasts) - positives
    if not positives or not negatives:
        raise ValueError('AUC requires both cancer classes')
    ordered = sorted(breasts)
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        mean_rank = (index + 1 + end) / 2
        rank_sum += mean_rank * sum(y for _, y in ordered[index:end])
        index = end
    auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return {'cancer_auc': auc, 'breasts': len(breasts), 'images': len(predictions)}


def evaluate(args):
    import cv2
    import numpy as np
    import torch
    from skp.models.classification.net2d import Net

    recipe = read_recipe(args.recipe)
    partition = next((p for p in recipe['sets'] if p['id'] == args.set), None)
    if partition is None:
        raise ValueError('Unknown evaluation set')
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoints = [m['checkpoint'] for m in partition['members'].values()]
    if any(not Path(path).is_file() for path in checkpoints):
        raise FileNotFoundError('Train all three screening EMA members before evaluation')
    checkpoint_hashes = {path: sha256(path) for path in checkpoints}
    models = []
    for name, member in partition['members'].items():
        cfg = configuration(recipe, name, args.set, evaluation=True)
        model = Net(cfg)
        state = torch.load(member['checkpoint'], map_location='cpu', weights_only=True)['state_dict']
        state = {k[len('model.'):]: v for k, v in state.items() if k.startswith('model.')}
        model.load_state_dict(state, strict=True)
        models.append((model.eval().to(args.device), cfg))
    predictions = []
    with torch.inference_mode():
        for row in read_rows(partition['holdout_csv']):
            image = cv2.imread(str(Path(recipe['images']) / row['filename']), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('Missing cropped holdout image: ' + row['filename'])
            values = []
            for model, cfg in models:
                x = image
                ratio = cfg.pad_to_aspect_ratio
                if ratio and x.shape[0] / x.shape[1] > ratio:
                    extra = round(x.shape[0] / ratio) - x.shape[1]
                    x = np.pad(x, ((0, 0), (extra // 2, extra - extra // 2)))
                x = cfg.val_transforms(image=x)['image']
                x = torch.from_numpy(np.ascontiguousarray(x))[None, None].float().to(args.device)
                values.append(model({'x': x})['logits'][0, 0].sigmoid().item())
            predictions.append({'breast_id': row['breast_id'], 'filename': row['filename'],
                                'cancer': int(row['cancer']), 'prediction': sum(values) / len(values)})
    metrics = summarize(predictions)
    args.output.mkdir(parents=True)
    with (args.output / 'image_predictions.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    metrics.update({'set': args.set, 'recipe_sha256': sha256(args.recipe),
                    'holdout_csv': partition['holdout_csv'],
                    'checkpoints': checkpoints, 'checkpoint_sha256': checkpoint_hashes})
    with (args.output / 'metrics.json').open('x') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics))


def train_locally(argv):
    """Adapt the preserved runner in memory; its archived source stays intact."""
    from mammoscreen_logging import LocalTrainingLogger

    sys.path.insert(0, str(ROOT / 'skp'))
    spec = importlib.util.spec_from_file_location('mammoscreen_historical_train', ROOT / 'skp/train.py')
    runner = importlib.util.module_from_spec(spec)
    # New Lightning releases removed NeptuneLogger. Supply the archived import
    # only while loading this module; active runs always use our local logger.
    from lightning.pytorch import loggers
    original_logger = getattr(loggers, 'NeptuneLogger', None)
    loggers.NeptuneLogger = LocalTrainingLogger
    try:
        spec.loader.exec_module(runner)
    finally:
        if original_logger is None:
            del loggers.NeptuneLogger
        else:
            loggers.NeptuneLogger = original_logger
    runner.NeptuneLogger = LocalTrainingLogger
    # Lightning DDP re-executes the active module with sys.argv. Keep _train
    # there so child ranks apply this adapter too; only hide it while parsing.
    historical_parse = runner.parse_args

    def parse_training_args():
        launch_argv = sys.argv
        try:
            sys.argv = [launch_argv[0], *argv]
            return historical_parse()
        finally:
            sys.argv = launch_argv

    runner.parse_args = parse_training_args
    runner.main()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_train':
        train_locally(argv[1:])
        return
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    for flag in ['splits', 'images', 'ddsm-images', 'ddsm-masks', 'ddsm-annotations', 'runs', 'output']:
        plan.add_argument('--' + flag, type=Path, required=True)
    plan.add_argument('--project', default='mammoscreen-local', help='Local experiment label')
    run = commands.add_parser('run')
    run.add_argument('--recipe', type=Path, required=True)
    run.add_argument('--stage', choices=['pretraining', 'screening'], required=True)
    run.add_argument('--set', type=int)
    run.add_argument('--devices', type=int, default=2)
    run.add_argument('--member', choices=list(MEMBERS), help='Run just one resolution')
    check = commands.add_parser('check')
    check.add_argument('--recipe', type=Path, required=True)
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--recipe', type=Path, required=True)
    evaluation.add_argument('--set', type=int, required=True)
    evaluation.add_argument('--output', type=Path, required=True, help='New evaluation directory')
    evaluation.add_argument('--device', default='cpu')
    args = p.parse_args(argv)
    if args.command == 'plan':
        if args.output.exists():
            raise FileExistsError(args.output)
        recipe = build_plan(args)
        # Validate before publishing a plan; validation reads no images or weights.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(recipe, f, indent=2)
        try:
            read_recipe(args.output)
        except Exception:
            args.output.unlink()  # only the new plan just created here
            raise
    elif args.command == 'run':
        run_jobs(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        recipe = read_recipe(args.recipe)
        print(f'Validated {len(recipe["sets"])} patient-disjoint evaluation sets')


if __name__ == '__main__':
    main()

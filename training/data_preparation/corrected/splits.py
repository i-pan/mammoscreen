"""Patient-grouped, stratified splits with locally owned random state."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binary(value):
    value = str(value).strip().lower()
    if value in {'true', '1', '1.0'}:
        return 1
    if value in {'false', '0', '0.0'}:
        return 0
    raise ValueError(f'Expected binary label, got {value!r}')


def prepare_rows(rows):
    result = []
    seen = set()
    for source in rows:
        row = dict(source)
        pid, image = str(row['patient_id']).strip(), str(row['image_id']).strip()
        if not pid or not image or row['laterality'] not in {'L', 'R'}:
            raise ValueError('Missing image/patient ID or invalid laterality')
        if (pid, image) in seen:
            raise ValueError('Duplicate patient/image ID')
        seen.add((pid, image))
        row['patient_id'], row['image_id'] = pid, image
        row['filename'] = f'{pid}/{image}.png'
        row['breast_id'] = f'{pid}_{row["laterality"]}'
        for key in ['cancer', 'biopsy', 'invasive', 'difficult_negative_case']:
            row[key] = binary(row[key])
        density = str(row.get('density', '')).strip().upper()
        if density in {'', 'NAN', '-1'}:
            row['density'] = -1
        elif density in 'ABCD' and len(density) == 1:
            row['density'] = 'ABCD'.index(density)
        else:
            raise ValueError('Expected raw density A/B/C/D or missing; do not preprocess twice')
        birads = str(row.get('BIRADS', '')).strip()
        value = -1 if birads.lower() in {'', 'nan', '-1'} else float(birads)
        if not math.isfinite(value) or value not in {-1, 0, 1, 2}:
            raise ValueError('Expected BIRADS 0/1/2 or missing')
        row['BIRADS'] = 0 if row['biopsy'] else int(value)
        result.append(row)
    indices = {b: i for i, b in enumerate(sorted({r['breast_id'] for r in result}))}
    for row in result:
        row['breast_id_index'] = indices[row['breast_id']]
    return result


def patient_ids(rows):
    return {str(r['patient_id']) for r in rows}


def validate_partition(train, holdout, folds):
    if not train or not holdout or patient_ids(train) & patient_ids(holdout):
        raise ValueError('Empty partition or patient leakage between training and holdout')
    by_patient = defaultdict(set)
    for row in train:
        by_patient[str(row['patient_id'])].add(int(row['fold']))
    if any(len(v) != 1 for v in by_patient.values()):
        raise ValueError('Patient appears in multiple validation folds')
    if {int(r['fold']) for r in train} != set(range(folds)):
        raise ValueError('Missing or invalid folds')
    for subset in [holdout] + [[r for r in train if int(r['fold']) == i] for i in range(folds)]:
        if {binary(r['cancer']) for r in subset} != {0, 1}:
            raise ValueError('Each holdout/validation fold must contain both cancer classes')
    breast_labels = defaultdict(set)
    for row in train + holdout:
        breast_labels[row['breast_id']].add(binary(row['cancer']))
    if any(len(v) != 1 for v in breast_labels.values()):
        raise ValueError('Inconsistent cancer labels within a breast')


def make_splits(rows, seed=88, repeats=3, folds=10, holdout_fraction=0.1):
    if repeats < 1 or folds < 2 or not 0 < holdout_fraction < 1:
        raise ValueError('Invalid split parameters')
    rows = prepare_rows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row['patient_id'])].append(row)
    strata = {label: sorted(pid for pid, group in grouped.items()
                            if max(r['cancer'] for r in group) == label) for label in [0, 1]}
    rng = random.Random(seed)
    outputs, previous = [], set()
    for repeat in range(repeats):
        for attempt in range(100):
            held = set()
            assignments = {}
            for label in [0, 1]:
                patients = strata[label].copy()
                rng.shuffle(patients)
                count = max(1, round(len(patients) * holdout_fraction))
                if len(patients) - count < folds:
                    raise ValueError('Too few patients per cancer stratum for requested folds')
                held.update(patients[:count])
                for index, pid in enumerate(patients[count:]):
                    assignments[pid] = index % folds
            signature = frozenset(held)
            if signature not in previous:
                previous.add(signature)
                break
        else:
            raise ValueError('Could not obtain distinct holdout memberships')
        training = [dict(r, fold=assignments[str(r['patient_id'])])
                    for r in rows if str(r['patient_id']) not in held]
        holdout = [dict(r) for r in rows if str(r['patient_id']) in held]
        counts = Counter(r['cancer'] for r in training)
        maximum = max(counts.values())
        for row in training:
            row['sampling_weight'] = maximum / counts[row['cancer']]
        validate_partition(training, holdout, folds)
        outputs.append((training, holdout))
    return outputs


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True, help='Raw RSNA train.csv')
    p.add_argument('--output', type=Path, required=True, help='New directory')
    p.add_argument('--seed', type=int, default=88)
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    splits = make_splits(read_rows(args.annotations), args.seed, args.repeats, args.folds)
    args.output.mkdir(parents=True)
    sets = []
    for index, (train, holdout) in enumerate(splits, 1):
        a, b = f'train_kfold_v{index:02d}.csv', f'holdout_test_v{index:02d}.csv'
        write_csv(args.output / a, train)
        write_csv(args.output / b, holdout)
        sets.append({'id': index, 'training_csv': a, 'holdout_csv': b,
                     'training_sha256': sha256(args.output / a),
                     'holdout_sha256': sha256(args.output / b)})
    manifest = {'schema_version': 1, 'seed': args.seed, 'folds': args.folds,
                'method': 'Patient cancer-stratified holdout and round-robin grouped validation folds',
                'nominal_fractions': {'holdout': 0.1, 'validation': 0.9 / args.folds,
                                      'training': 0.9 * (1 - 1 / args.folds)},
                'input_sha256': sha256(args.annotations), 'sets': sets}
    with (args.output / 'splits.json').open('x') as f:
        json.dump(manifest, f, indent=2)


if __name__ == '__main__':
    main()

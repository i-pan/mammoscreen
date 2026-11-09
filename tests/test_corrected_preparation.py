import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'training/data_preparation/corrected/prepare.py'
spec = importlib.util.spec_from_file_location('corrected_preparation', PATH)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_original_image_coordinates_and_largest_component():
    image = np.zeros((100, 120), dtype=np.uint8)
    image[:3] = 255  # excluded scanner-border artifact
    image[20:60, 30:80] = 200
    image[70:74, 90:94] = 255
    assert prepare.crop_box(image) == (30, 20, 50, 40)
    with pytest.raises(ValueError, match='foreground'):
        prepare.crop_box(np.zeros((100, 120), dtype=np.uint8))


def test_crop_label_cli_preserves_pixel_boxes(tmp_path):
    images = tmp_path / 'images' / 'patient'
    images.mkdir(parents=True)
    image = np.zeros((100, 120), dtype=np.uint8)
    image[20:60, 30:80] = 200
    cv2.imwrite(str(images / 'view.png'), image)
    out = tmp_path / 'coords.csv'
    prepare.main(['crop-labels', '--images', str(images.parent), '--output', str(out)])
    with out.open() as f:
        assert list(csv.DictReader(f)) == [
            {'filename': 'patient/view.png', 'x': '30', 'y': '20', 'w': '50', 'h': '40'}
        ]
    with pytest.raises(FileExistsError):
        prepare.main(['crop-labels', '--images', str(images.parent), '--output', str(out)])


def test_disjoint_lesions_keep_their_categories_and_overlap_is_explicit():
    lesions = [(10, 10, 20, 20, 1), (60, 10, 20, 20, 4)]
    mask = prepare.ellipse_mask((50, 100), lesions)
    assert mask[20, 20] == 1 and mask[20, 70] == 4
    assert mask[45, 90] == 0
    overlap = prepare.ellipse_mask((50, 100), lesions + [(10, 10, 20, 20, 2)])
    assert overlap[20, 20] == 2 and overlap[20, 70] == 4
    with pytest.raises(ValueError):
        prepare.ellipse_mask((50, 100), [(95, 10, 20, 20, 1)])


@pytest.mark.parametrize('order,centers', [('yx', [(20, 30), (20, 70)]),
                                         ('xy', [(30, 20), (70, 20)])])
def test_mask_cli_csv_coordinate_mapping(tmp_path, order, centers):
    images = tmp_path / 'png'
    images.mkdir()
    cv2.imwrite(str(images / 'view.png'), np.zeros((100, 100), dtype=np.uint8))
    annotations = tmp_path / 'labels.csv'
    with annotations.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filepath', 'pathology', 'abnormality',
                                              'x1', 'y1', 'x2', 'y2'])
        writer.writeheader()
        writer.writerows([
            {'filepath': 'view.png', 'pathology': 'BENIGN', 'abnormality': 'mass',
             'x1': 10, 'y1': 20, 'x2': 30, 'y2': 40},
            {'filepath': './view.png', 'pathology': 'MALIGNANT', 'abnormality': 'calcification',
             'x1': 10, 'y1': 60, 'x2': 30, 'y2': 80},
        ])
    output = tmp_path / 'masks'
    args = ['ddsm-masks', '--images', str(images), '--annotations', str(annotations),
            '--output', str(output), '--bbox-order', order]
    prepare.main(args)
    mask = cv2.imread(str(output / 'view.png'), cv2.IMREAD_UNCHANGED)
    assert mask[centers[0]] == 2 and mask[centers[1]] == 3
    assert set(np.unique(mask)) == {0, 2, 3}
    with pytest.raises(FileExistsError):
        prepare.main(args)


def test_no_images_or_escaping_path_does_not_create_outputs(tmp_path):
    images = tmp_path / 'images'
    images.mkdir()
    output = tmp_path / 'coords.csv'
    with pytest.raises(ValueError, match='No PNG'):
        prepare.main(['crop-labels', '--images', str(images), '--output', str(output)])
    assert not output.exists()
    annotations = tmp_path / 'bad.csv'
    annotations.write_text('filepath,pathology,abnormality,x1,y1,x2,y2\n'
                           '../escape.png,BENIGN,mass,1,1,2,2\n')
    masks = tmp_path / 'masks'
    with pytest.raises(ValueError, match='escapes'):
        prepare.main(['ddsm-masks', '--images', str(images), '--annotations', str(annotations),
                      '--output', str(masks)])
    assert not masks.exists()

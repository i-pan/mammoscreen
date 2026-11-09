import numpy as np
import pytest
import torch

from mammoscreen import MammoscreenPipeline, load_image
from mammoscreen.io import _uint8


class Crop(torch.nn.Module):
    def __init__(self, box=(1, 2, 4, 3)):
        super().__init__()
        self.box = box

    def preprocess(self, image):
        return image

    def forward(self, pixels, shape):
        assert pixels.shape[1] == 1
        assert shape.tolist() == [[8, 6]]
        return torch.tensor([self.box], device=pixels.device)


class Screen(torch.nn.Module):
    def forward(self, samples, device):
        self.received = samples
        n = len(samples)
        return {"cancer": torch.full((n,), 0.25), "density": torch.full((n, 4), 0.25)}


def test_crop_and_breast_grouping():
    screen = Screen()
    pipeline = MammoscreenPipeline(Crop(), screen)
    image = np.arange(48, dtype=np.uint8).reshape(8, 6)
    result = pipeline([{"cc": image, "mlo": image}, {"mlo": image}])
    assert [set(s) for s in screen.received] == [{"cc", "mlo"}, {"mlo"}]
    np.testing.assert_array_equal(screen.received[0]["cc"], image[2:5, 1:5])
    assert result["cancer"].shape == (2,)
    assert result["density"].shape == (2, 4)
    assert result["cancer"].device.type == "cpu"


def test_clip_crop_and_reject_empty():
    image = np.zeros((8, 6), dtype=np.uint8)
    p = MammoscreenPipeline(Crop((-2, -1, 10, 12)), Screen())
    assert p.crop(image).shape == (8, 6)
    for box in [(0, 0, 0, 1), (9, 9, 1, 1)]:
        with pytest.raises(ValueError):
            MammoscreenPipeline(Crop(box), Screen()).crop(image)


@pytest.mark.parametrize("sample", [[], {}, [{"left": np.zeros((8, 6), dtype=np.uint8)}]])
def test_reject_invalid_views(sample):
    with pytest.raises(ValueError):
        MammoscreenPipeline(Crop(), Screen())(sample)


def test_reject_unscaled_arrays_and_color():
    p = MammoscreenPipeline(Crop(), Screen())
    for image in [np.zeros((8, 6), dtype=np.float32), np.zeros((8, 6, 3), dtype=np.uint8)]:
        with pytest.raises(ValueError):
            p.crop(image)


def test_raster_and_constant_conversion(tmp_path):
    import cv2

    image = np.arange(48, dtype=np.uint8).reshape(8, 6)
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), image)
    np.testing.assert_array_equal(load_image(path), image)
    assert not _uint8(np.full((8, 6), 1000, dtype=np.uint16)).any()
    with pytest.raises(ValueError):
        load_image(tmp_path / "missing.png")


def test_reject_multipage_tiff(tmp_path):
    import cv2

    path = tmp_path / "multiple.tiff"
    assert cv2.imwritemulti(str(path), [
        np.full((8, 6), 10, dtype=np.uint8), np.full((8, 6), 200, dtype=np.uint8)
    ])
    with pytest.raises(ValueError, match="single-frame"):
        load_image(path)


def test_square_preprocessing_matches_saved_training_padding(monkeypatch):
    import cv2
    from mammoscreen._models.screen import modeling

    monkeypatch.setattr(modeling, "create_model", lambda **kwargs: torch.nn.Identity())
    model = modeling.MammoModel("unused", (6, 6), False)
    image = np.tile(np.array([20, 180], dtype=np.uint8), (6, 1))
    result = model.preprocess({"cc": image}, device="cpu")[0]["cc"][0].numpy()
    # Original b386692c config.pkl records PadIfNeeded border_mode=4.
    expected = cv2.copyMakeBorder(image, 0, 0, 2, 2, cv2.BORDER_REFLECT_101)
    np.testing.assert_array_equal(result, expected)


def test_real_model_forward_averages_views_within_each_breast(monkeypatch):
    from mammoscreen._models.screen import modeling

    monkeypatch.setattr(modeling, "create_model", lambda **kwargs: torch.nn.Identity())
    model = modeling.MammoModel("unused", (6, 6), False, feature_dim=1).eval()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.], [0.], [1.], [2.], [3.]]))
        model.linear.bias.zero_()
    samples = [
        {"cc": torch.zeros(1, 2, 2), "mlo": torch.full((1, 2, 2), 255.)},
        {"cc": torch.full((1, 2, 2), 127.5)},
    ]
    with torch.inference_mode():
        result = model(samples)
    # Independent oracle: average probabilities (not logits), preserving breast order.
    density_views = np.exp(np.array([[0, -1, -2, -3], [0, 1, 2, 3]], dtype=float))
    density_views /= density_views.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(result["density"].numpy(),
                               [density_views.mean(axis=0), np.full(4, 0.25)], rtol=1e-6)
    np.testing.assert_allclose(result["cancer"].numpy(), [0.5, 0.5], rtol=1e-6)


def test_cli_serialization_and_loading_options(monkeypatch, capsys):
    import sys
    from mammoscreen import cli

    calls = {}
    image = np.zeros((8, 6), dtype=np.uint8)

    class Loader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["loader"] = (args, kwargs)
            return cls()

        def __call__(self, sample):
            assert set(sample) == {"mlo"}
            np.testing.assert_array_equal(sample["mlo"], image)
            return {"cancer": torch.tensor([0.5]), "density": torch.full((1, 4), 0.25)}

    monkeypatch.setattr(cli, "MammoscreenPipeline", Loader)
    monkeypatch.setattr(cli, "load_image", lambda path: image)
    monkeypatch.setattr(sys, "argv", ["mammoscreen", "--mlo", "view.png",
                                      "--crop-source", "crop", "--screen-source", "screen",
                                      "--local-files-only"])
    cli.main()
    import json
    assert json.loads(capsys.readouterr().out) == {
        "cancer": [0.5], "density": [[0.25, 0.25, 0.25, 0.25]]
    }
    assert calls["loader"] == (("cpu", "crop", "screen"), {"local_files_only": True})


def test_dicom_monochrome1_and_constant(tmp_path):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    path = tmp_path / "image.dcm"
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = 2, 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.array([[0, 1], [2, 3]], dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    np.testing.assert_array_equal(load_image(path), [[255, 170], [85, 0]])
    ds.PixelData = np.zeros((2, 2), dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    assert not load_image(path).any()

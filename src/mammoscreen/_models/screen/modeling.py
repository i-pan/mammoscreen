import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn

from transformers import PreTrainedModel
from timm import create_model
from typing import Mapping, Sequence, Tuple
from .configuration import MammoConfig


def _pad_to_aspect_ratio(img: np.ndarray, aspect_ratio: float) -> np.ndarray:
    """
    Pads to specified aspect ratio, only if current aspect ratio is
    greater.
    """
    h, w = img.shape[:2]
    if h / w > aspect_ratio:
        new_w = round(h / aspect_ratio)
        w_diff = new_w - w
        left_pad = w_diff // 2
        right_pad = w_diff - left_pad
        padding = ((0, 0), (left_pad, right_pad))
        if img.ndim == 3:
            padding = padding + ((0, 0),)
        img = np.pad(img, padding, mode="constant", constant_values=0)
    return img


def _to_torch_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    if x.ndim == 2:
        x = torch.from_numpy(x).unsqueeze(0)
    elif x.ndim == 3:
        x = torch.from_numpy(x)
        if torch.tensor(x.size()).argmin().item() == 2:
            # channels last -> first
            x = x.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected 2 or 3 dimensions, got {x.ndim}")
    return x.float().to(device)


class MammoModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        image_size: Tuple[int, int],
        pad_to_aspect_ratio: bool,
        feature_dim: int = 1280,
        dropout: float = 0.1,
        num_classes: int = 5,
        in_chans: int = 1,
    ):
        super().__init__()
        self.backbone = create_model(
            model_name=backbone,
            pretrained=False,
            num_classes=0,
            global_pool="",
            features_only=False,
            in_chans=in_chans,
        )
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(feature_dim, num_classes)

        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self.aspect_ratio = image_size[0] / image_size[1]
        if self.pad_to_aspect_ratio:
            self.resize = A.Resize(image_size[0], image_size[1], p=1)
        else:
            self.resize = A.Compose(
                [
                    A.LongestMaxSize(image_size[0], p=1),
                    # Match the preserved square-model training config; defaults
                    # changed from reflection to constant padding in Albumentations 2.
                    A.PadIfNeeded(
                        image_size[0], image_size[1], border_mode=cv2.BORDER_REFLECT_101, p=1
                    ),
                ],
                p=1,
            )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # [0, 255] -> [-1, 1]
        mini, maxi = 0.0, 255.0
        x = (x - mini) / (maxi - mini)
        x = (x - 0.5) * 2.0
        return x

    def preprocess(
        self,
        x: Mapping[str, np.ndarray] | Sequence[Mapping[str, np.ndarray]],
        device: str,
    ) -> Sequence[Mapping[str, torch.Tensor]]:
        # x is a dict (or list of dicts) with keys "cc" and/or "mlo"
        # though the actual keys do not matter
        if not isinstance(x, Sequence):
            assert isinstance(x, Mapping)
            x = [x]
        if self.pad_to_aspect_ratio:
            x = [
                {
                    k: _pad_to_aspect_ratio(v.copy(), self.aspect_ratio)
                    for k, v in sample.items()
                }
                for sample in x
            ]
        x = [
            {
                k: _to_torch_tensor(self.resize(image=v)["image"], device=device)
                for k, v in sample.items()
            }
            for sample in x
        ]
        return x

    def forward(
        self, x: Sequence[Mapping[str, torch.Tensor]]
    ) -> Mapping[str, torch.Tensor]:
        batch_tensor = []
        batch_indices = []
        for idx, sample in enumerate(x):
            for k, v in sample.items():
                batch_tensor.append(v)
                batch_indices.append(idx)

        batch_tensor = torch.stack(batch_tensor, dim=0)
        batch_tensor = self.normalize(batch_tensor)
        features = self.pooling(self.backbone(batch_tensor))
        b, d = features.shape[:2]
        features = features.reshape(b, d)
        logits = self.linear(features)
        # cancer
        logits0 = logits[:, 0].sigmoid()
        # density
        logits1 = logits[:, 1:].softmax(dim=1)
        # mean over views
        batch_indices = torch.tensor(batch_indices)
        logits0 = torch.stack(
            [logits0[batch_indices == i].mean(dim=0) for i in batch_indices.unique()]
        )
        logits1 = torch.stack(
            [logits1[batch_indices == i].mean(dim=0) for i in batch_indices.unique()]
        )
        return {"cancer": logits0, "density": logits1}


class MammoEnsemble(PreTrainedModel):
    config_class = MammoConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_models = config.num_models
        for i in range(self.num_models):
            setattr(
                self,
                f"net{i}",
                MammoModel(
                    config.backbone,
                    config.image_sizes[i],
                    config.pad_to_aspect_ratio[i],
                    config.feature_dim,
                    config.dropout,
                    config.num_classes,
                    config.in_chans,
                ),
            )

    @staticmethod
    def load_image_from_dicom(path: str) -> np.ndarray | None:
        try:
            from pydicom import dcmread
            from pydicom.pixels import apply_voi_lut
        except ModuleNotFoundError:
            print("`pydicom` is not installed, returning None ...")
            return None
        dicom = dcmread(path)
        arr = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr

        arr = arr - arr.min()
        arr = arr / arr.max()
        arr = (arr * 255).astype("uint8")
        return arr

    def forward(
        self,
        x: Mapping[str, np.ndarray] | Sequence[Mapping[str, np.ndarray]],
        device: str = "cpu",
    ) -> Mapping[str, torch.Tensor]:
        out = []
        for i in range(self.num_models):
            model = getattr(self, f"net{i}")
            x_pp = model.preprocess(x, device=device)
            out.append(model(x_pp))
        out = {k: torch.stack([o[k] for o in out]).mean(0) for k in out[0].keys()}
        return out

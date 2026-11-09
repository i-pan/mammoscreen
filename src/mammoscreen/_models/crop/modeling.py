import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy.typing import NDArray
from transformers import PreTrainedModel
from timm import create_model
from typing import Optional

from .configuration import MammoCropConfig

_PYDICOM_AVAILABLE = False
try:
    from pydicom import dcmread
    from pydicom.pixels import apply_voi_lut

    _PYDICOM_AVAILABLE = True
except ModuleNotFoundError:
    pass


class GeM(nn.Module):
    def __init__(
        self, p: int = 3, eps: float = 1e-6, dim: int = 2, flatten: bool = True
    ):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        assert dim in {2, 3}, f"dim must be one of [2, 3], not {dim}"
        self.dim = dim
        if self.dim == 2:
            self.func = F.adaptive_avg_pool2d
        elif self.dim == 3:
            self.func = F.adaptive_avg_pool3d
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assumes x.shape is (n, c, [t], h, w)
        x = self.func(x.clamp(min=self.eps).pow(self.p), output_size=1).pow(
            1.0 / self.p
        )
        return self.flatten(x)


class MammoCropModel(PreTrainedModel):
    config_class = MammoCropConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = create_model(
            model_name=config.backbone,
            pretrained=False,
            num_classes=0,
            global_pool="",
            features_only=False,
            in_chans=config.in_chans,
        )
        self.pooling = GeM(p=3, dim=2)
        self.dropout = nn.Dropout(p=config.dropout)
        self.linear = nn.Linear(config.feature_dim, config.num_classes)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # [0, 255] -> [-1, 1]
        mini, maxi = 0.0, 255.0
        x = (x - mini) / (maxi - mini)
        x = (x - 0.5) * 2.0
        return x

    @staticmethod
    def load_image_from_dicom(path: str) -> Optional[NDArray]:
        if not _PYDICOM_AVAILABLE:
            print("`pydicom` is not installed, returning None ...")
            return None
        dicom = dcmread(path)
        arr = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            # invert image if needed
            arr = arr.max() - arr

        arr = arr - arr.min()
        arr = arr / arr.max()
        arr = (arr * 255).astype("uint8")
        return arr

    @staticmethod
    def preprocess(x: NDArray) -> NDArray:
        return A.Resize(256, 256, p=1)(image=x)["image"]

    def forward(
        self, x: torch.Tensor, img_shape: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # if img_shape is provided, will provide rescaled coordinates
        # otherwise, provide normalized [0, 1] coordinates
        # coords format is xywh
        if img_shape is not None:
            assert (
                x.size(0) == img_shape.size(0)
            ), f"x.size(0) [{x.size(0)}] must equal img_shape.size(0) [{img_shape.size(0)}]"
            # img_shape = (batch_dim, 2)
            # img_shape[:, 0] = height, img_shape[:, 1] = width

        x = self.normalize(x)
        features = self.pooling(self.backbone(x))
        coords = self.linear(features).sigmoid()

        if img_shape is None:
            return coords

        rescaled_coords = coords.clone()
        rescaled_coords[:, 0] = rescaled_coords[:, 0] * img_shape[:, 1]
        rescaled_coords[:, 1] = rescaled_coords[:, 1] * img_shape[:, 0]
        rescaled_coords[:, 2] = rescaled_coords[:, 2] * img_shape[:, 1]
        rescaled_coords[:, 3] = rescaled_coords[:, 3] * img_shape[:, 0]
        return rescaled_coords.int()

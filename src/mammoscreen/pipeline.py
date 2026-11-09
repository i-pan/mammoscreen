"""Run the published crop model followed by the three-model ensemble."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

SCREEN_REVISION = "b619a889a8214f3b3f9e7427c2603608a44e6d0a"
CROP_REVISION = "bec2e302d47176d743a86d276d8402f876ad0f9f"


class MammoscreenPipeline:
    def __init__(self, crop_model, screening_model, device: str = "cpu"):
        self.device = torch.device(device)
        self.crop_model = crop_model.eval().to(self.device)
        self.screening_model = screening_model.eval().to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        device: str = "cpu",
        crop_source: str | Path = "ianpan/mammo-crop",
        screen_source: str | Path = "ianpan/mammoscreen",
        *,
        local_files_only: bool = False,
    ):
        """Load pinned HF weights, or local snapshot directories.

        The implementations are vendored here; no trust_remote_code is needed.
        """
        from ._models.crop.modeling import MammoCropModel
        from ._models.screen.modeling import MammoEnsemble

        def load(model_class, source, revision):
            options = {"local_files_only": local_files_only, "use_safetensors": True}
            if not Path(source).is_dir():
                options["revision"] = revision
            return model_class.from_pretrained(str(source), **options)

        return cls(
            load(MammoCropModel, crop_source, CROP_REVISION),
            load(MammoEnsemble, screen_source, SCREEN_REVISION),
            device,
        )

    @torch.inference_mode()
    def crop(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.ndim != 2 or 0 in image.shape:
            raise ValueError("Expected a nonempty 2D grayscale NumPy array")
        if image.dtype != np.uint8:
            raise ValueError("Expected uint8 pixels; use load_image for files")
        small = self.crop_model.preprocess(image)
        tensor = torch.from_numpy(np.ascontiguousarray(small)).to(self.device).float()[None, None]
        shape = torch.tensor([image.shape], device=self.device)
        coords = self.crop_model(tensor, shape)[0].detach().cpu().numpy()
        if len(coords) != 4 or not np.isfinite(coords).all():
            raise ValueError("Crop model returned invalid coordinates")
        x, y, width, height = map(int, coords)
        if width <= 0 or height <= 0:
            raise ValueError("Crop model returned an empty crop")
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(image.shape[1], x + width), min(image.shape[0], y + height)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Crop model returned a box outside the image")
        return np.ascontiguousarray(image[y0:y1, x0:x1])

    @torch.inference_mode()
    def __call__(self, samples: Mapping | Sequence[Mapping]) -> dict[str, torch.Tensor]:
        """Predict one breast per mapping, with CC and/or MLO images.

        Returns CPU tensors: cancer (N,), density (N, 4).
        """
        if isinstance(samples, Mapping):
            samples = [samples]
        if not isinstance(samples, Sequence) or not samples:
            raise ValueError("Provide at least one breast with CC and/or MLO views")
        for sample in samples:
            if not isinstance(sample, Mapping) or not sample or set(sample) - {"cc", "mlo"}:
                raise ValueError("Each breast must have cc and/or mlo keys")
        cropped = [{view: self.crop(image) for view, image in s.items()} for s in samples]
        output = self.screening_model(cropped, device=str(self.device))
        return {key: value.detach().cpu() for key, value in output.items()}

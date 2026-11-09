from transformers import PretrainedConfig
from typing import List, Tuple


class MammoConfig(PretrainedConfig):
    model_type = "mammo"

    def __init__(
        self,
        backbone: str = "tf_efficientnetv2_s",
        feature_dim: int = 1280,
        dropout: float = 0.1,
        num_classes: int = 5,
        in_chans: int = 1,
        num_models: int = 3,
        image_sizes: List[Tuple[int, int]] = [(2048, 1024), (1920, 1280), (1536, 1536)],
        pad_to_aspect_ratio: List[bool] = [True, True, False],
        **kwargs,
    ):
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.num_models = num_models
        assert len(image_sizes) == len(pad_to_aspect_ratio) == num_models, (
            f"length of `image_sizes` [{len(image_sizes)}] and `pad_to_aspect_ratio` "
            f"[{len(pad_to_aspect_ratio)}] must be equal to `num_models` [{num_models}]."
        )
        self.image_sizes = image_sizes
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        super().__init__(**kwargs)

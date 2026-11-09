from transformers import PretrainedConfig


class MammoCropConfig(PretrainedConfig):
    model_type = "mammo_crop"

    def __init__(
        self,
        backbone="mobilenetv3_small_100",
        feature_dim=1024,
        dropout=0.1,
        num_classes=4,
        in_chans=1,
        **kwargs,
    ):
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.dropout = dropout
        self.num_classes = num_classes
        self.in_chans = in_chans
        super().__init__(**kwargs)

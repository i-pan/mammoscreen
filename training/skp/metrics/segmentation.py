import numpy as np
import torch
import torchmetrics as tm

from einops import rearrange, reduce
from scipy.ndimage import center_of_mass, label as ndi_label
from skimage.morphology import remove_small_objects
from torch.nn.functional import one_hot
from torcheval.metrics.functional import binary_auprc
from typing import Dict

from skp.configs import Config


class MulticlassDiceScore(tm.Metric):
    # Each class is mutually exclusive

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p, t = out["logits"], batch["y"]
        if "mask_present" in batch:
            # ignore samples without a valid mask
            mask_present = batch["mask_present"]
            p, t = p[mask_present], t[mask_present]

        if "pseudolabel" in batch:
            # ignore mask pseudolabels
            p, t = p[~batch["pseudolabel"]], t[~batch["pseudolabel"]]

        num_classes = p.size(1)
        assert self.cfg.activation_fn == "softmax"
        p = p.argmax(1)

        p = one_hot(p, num_classes=num_classes)
        t = one_hot(t, num_classes=num_classes)

        if self.cfg.metric_invert_background:
            # assumes background class is index 0
            p[..., 0] = 1 - p[..., 0]
            t[..., 0] = 1 - t[..., 0]

        if self.cfg.metric_ignore_class0:
            p, t = p[..., 1:], t[..., 1:]

        if p.ndim == 5:
            s1 = "b x y z c -> b c x y z"
            s2 = "b c x y z -> b c"
        elif p.ndim == 4:
            s1 = "b h w c -> b c h w"
            s2 = "b c h w -> b c"
        else:
            raise Exception(f"p is not a valid shape {p.shape}")

        p = rearrange(p, s1).long()
        t = rearrange(t, s1).long()

        intersection = reduce(p * t, s2, "sum")
        denominator = reduce(p + t, s2, "sum")

        dice = (2 * intersection) / denominator

        self.dice_scores.append(dice)

    def compute(self) -> Dict[str, torch.Tensor]:
        dice_scores = torch.cat(self.dice_scores, dim=0)  # (N, C)
        metrics_dict = {}
        for c in range(dice_scores.shape[1]):
            # ignore NaN
            tmp_dice_scores = dice_scores[:, c].cpu().numpy()
            tmp_dice_scores = tmp_dice_scores[~np.isnan(tmp_dice_scores)]
            if self.cfg.metric_ignore_class0:
                c += 1
            metrics_dict[f"dice{c}"] = torch.tensor(np.mean(tmp_dice_scores))
        metrics_dict["dice_mean"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean(0)
        return metrics_dict


class MultilabelDiceScore(tm.Metric):
    # Pixel can be >1 class

    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.thresholds = torch.tensor(self.cfg.metric_thresholds or [0.5])
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p, t = out["logits"], batch["y"]
        if "mask_present" in batch:
            # ignore samples without a valid mask
            mask_present = batch["mask_present"]
            p, t = p[mask_present], t[mask_present]

        if "pseudolabel" in batch:
            # ignore mask pseudolabels
            p, t = p[~batch["pseudolabel"]], t[~batch["pseudolabel"]]

        if self.cfg.metric_labels_to_onehot:
            # if we are converting to onehot and using this metric
            # that means we have a multiclass classification problem
            # but we are using sigmoid act for specific reasons
            t = one_hot(t, num_classes=p.size(1))
            if p.ndim == 5:
                t = rearrange(t, "b x y z c -> b c x y z")
            elif p.ndim == 4:
                t = rearrange(t, "b h w c -> b c h w")

        # t should be one-hot encoded
        assert self.cfg.activation_fn == "sigmoid"
        p = p.sigmoid()

        # if only one class, add channel dimension
        if p.size(1) == 1 and p.ndim == t.ndim + 1:
            t = t.unsqueeze(1)

        assert p.ndim == t.ndim, (
            f"prediction [{p.ndim}] and label [{t.ndim}] tensors should have same # of dimensions, ensure labels are one-hot encoded"
        )

        if self.cfg.loss_params.get("invert_background", False):
            # assumes background class is index 0
            t[:, 0] = 1 - t[:, 0]
            # note, p already calibrated for inverted background
            # so no need to invert p background here

        p = torch.stack([p >= _th for _th in self.thresholds])
        t = torch.stack([t] * len(self.thresholds))
        if p.ndim == 6:
            s = "n b c x y z -> n b c"
        elif p.ndim == 5:
            s = "n b c h w -> n b c"
        else:
            raise Exception(f"p is not a valid shape {p.shape}")

        intersection = reduce(p * t, s, "sum")
        denominator = reduce(p + t, s, "sum")

        dice = (2 * intersection) / denominator

        self.dice_scores.append(dice)

    @staticmethod
    def _dice_ignore_nan(x: torch.Tensor) -> torch.Tensor:
        x = x.cpu().numpy()
        x = x[~np.isnan(x)]
        return torch.tensor(x.mean())

    def compute(self) -> Dict[str, torch.Tensor]:
        dice = torch.cat(
            self.dice_scores, dim=1
        )  # shape = (num_thresholds, num_samples, num_classes)
        dice_over_thresholds = torch.stack(
            [
                torch.stack(
                    [
                        self._dice_ignore_nan(dice[num_t, :, num_c])
                        for num_t in range(dice.shape[0])
                    ]
                )
                for num_c in range(dice.shape[2])
            ]
        )  # shape = (num_classes, num_thresholds)

        best_dice = dice_over_thresholds.amax(dim=1)
        best_thresholds = self.thresholds[
            dice_over_thresholds.argmax(dim=1).cpu().numpy()
        ]
        metrics_dict = {f"dice{idx}": _dice for idx, _dice in enumerate(best_dice)}
        metrics_dict["dice_mean"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean(0)
        metrics_dict.update(
            {f"th{idx}": _th for idx, _th in enumerate(best_thresholds)}
        )

        return metrics_dict


class MultimodalVesselSegDice(MultilabelDiceScore):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(cfg, dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.thresholds = torch.tensor(self.cfg.metric_thresholds or [0.5])
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)
        self.add_state("modality", default=[], dist_reduce_fx=None)
        self.int2modality = {0: "cta", 1: "mra", 2: "mrt1c", 3: "mrt2"}

    def calculate_dice_over_thresholds(
        self, dice: torch.Tensor, prefix: str = ""
    ) -> Dict[str, torch.Tensor]:
        dice_over_thresholds = torch.stack(
            [
                torch.stack(
                    [
                        self._dice_ignore_nan(dice[num_t, :, num_c])
                        for num_t in range(dice.shape[0])
                    ]
                )
                for num_c in range(dice.shape[2])
            ]
        )  # shape = (num_classes, num_thresholds)

        best_dice = dice_over_thresholds.amax(dim=1)
        best_thresholds = self.thresholds[
            dice_over_thresholds.argmax(dim=1).cpu().numpy()
        ]
        metrics_dict = {
            f"{prefix}dice{idx}": _dice for idx, _dice in enumerate(best_dice)
        }
        metrics_dict[f"{prefix}dice_mean"] = torch.stack(
            [v for v in metrics_dict.values()]
        ).mean(0)
        metrics_dict.update(
            {f"{prefix}th{idx}": _th for idx, _th in enumerate(best_thresholds)}
        )

        return metrics_dict

    def update(self, out: Dict, batch: Dict):
        super().update(out, batch)
        self.modality.append(batch["modality"])

    def compute(self) -> Dict[str, torch.Tensor]:
        dice = torch.cat(
            self.dice_scores, dim=1
        )  # shape = (num_thresholds, num_samples, num_classes)
        overall_metrics_dict = self.calculate_dice_over_thresholds(dice)
        modality = torch.cat(self.modality, dim=0)
        for each_modality in torch.unique(modality):
            each_modality = int(each_modality)
            prefix = f"{self.int2modality[each_modality]}_"
            tmp_metrics_dict = self.calculate_dice_over_thresholds(
                dice[:, modality == each_modality], prefix=prefix
            )
            overall_metrics_dict.update(tmp_metrics_dict)

        return overall_metrics_dict


class TotalSegmentatorForegroundDice(MultilabelDiceScore):
    def update(self, out: Dict, batch: Dict) -> None:
        # only makes sense if using sigmoid activation
        assert self.cfg.activation_fn == "sigmoid"

        p, t = out["logits"], batch["y"]

        # p.shape = (N, C, [T], H, W)
        # t.shape = (N, [T], H, W)

        # get foreground preds and gt
        if self.cfg.loss_params.get("invert_background", False):
            # foreground class is index 0
            p = p[:, 0]  # -> (N, [T], H, W)
            t = (t > 0).float()
        else:
            assert self.cfg.loss_params.get("ignore_background", False)
            # no specific background class
            # take max across classes
            p = p.amax(dim=1)
            t = (t > 0).float()

        p = p.sigmoid()
        assert p.ndim == t.ndim, (
            f"prediction [{p.ndim}] and label [{t.ndim}] tensors should have same # of dimensions"
        )

        p = torch.stack([p >= _th for _th in self.thresholds])
        t = torch.stack([t] * len(self.thresholds))
        if p.ndim == 5:
            s = "n b t h w -> n b 1"
        elif p.ndim == 4:
            s = "n b h w -> n b 1"
        else:
            raise Exception(f"p is not a valid shape {p.shape}")

        intersection = reduce(p * t, s, "sum")
        denominator = reduce(p + t, s, "sum")

        dice = (2 * intersection) / denominator

        self.dice_scores.append(dice)


# class MultilabelDiceScore(tm.Metric):
#     # Pixel can be >1 class

#     def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.cfg = cfg
#         self.add_state("dice_scores", default=[], dist_reduce_fx=None)

#     def update(self, out: Dict, batch: Dict) -> None:
#         p, t = out["logits"], batch["y"]

#         if self.cfg.metric_labels_to_onehot:
#             # if we are converting to onehot and using this metric
#             # that means we have a multiclass classification problem
#             # but we are using sigmoid act for specific reasons
#             t = one_hot(t, num_classes=p.size(1))
#             if p.ndim == 5:
#                 t = rearrange(t, "b x y z c -> b c x y z")
#             elif p.ndim == 4:
#                 t = rearrange(t, "b h w c -> b c h w")

#         # t should be one-hot encoded
#         assert self.cfg.activation_fn == "sigmoid"
#         p = p.sigmoid()
#         p = (p >= 0.5).long()

#         # if only one class, add channel dimension
#         if p.size(1) == 1 and p.ndim == t.ndim + 1:
#             t = t.unsqueeze(1)

#         assert (
#             p.ndim == t.ndim
#         ), f"prediction [{p.ndim}] and label [{t.ndim}] tensors should have same # of dimensions, ensure labels are one-hot encoded"

#         if self.cfg.loss_params.get("invert_background", False):
#             # assumes background class is index 0
#             t[:, 0] = 1 - t[:, 0]
#             # note, p already calibrated for inverted background
#             # so no need to invert p background here

#         if p.ndim == 5:
#             intersection = reduce(p * t, "b c x y z -> b c", "sum")
#             denominator = reduce(p + t, "b c x y z -> b c", "sum")
#         elif p.ndim == 4:
#             intersection = reduce(p * t, "b c h w -> b c", "sum")
#             denominator = reduce(p + t, "b c h w -> b c", "sum")

#         dice = (2 * intersection) / denominator

#         self.dice_scores.append(dice)

#     def compute(self) -> Dict[str, torch.Tensor]:
#         dice_scores = torch.cat(self.dice_scores, dim=0)  # (N, C)
#         metrics_dict = {}
#         for c in range(dice_scores.shape[1]):
#             # ignore NaN
#             tmp_dice_scores = dice_scores[:, c].cpu().numpy()
#             tmp_dice_scores = tmp_dice_scores[~np.isnan(tmp_dice_scores)]
#             metrics_dict[f"ml_dice{c}"] = torch.tensor(np.mean(tmp_dice_scores))
#         metrics_dict["ml_dice_mean"] = torch.stack([v for v in metrics_dict.values()]).mean(0)
#         return metrics_dict


def evaluate_segmentation(gt_mask, pred_mask, max_distance):
    """
    Calculates the number of true positives, false positives, and false negatives
    for 3D object segmentation.

    A true positive is a predicted object with its center within max_distance of a
    ground-truth object's center. Each ground-truth object can only be matched once.

    Args:
        gt_mask (np.ndarray): The 3D ground-truth binary mask (D, H, W).
        pred_mask (np.ndarray): The 3D prediction binary mask (D, H, W).
        max_distance (int): The maximum tolerated pixel distance between the
                             centers of a predicted and a ground-truth object for
                             it to be considered a true positive.

    Returns:
        tuple: A tuple containing the number of true positives,
               false positives, and false negatives.
    """
    # Label connected components in the ground-truth and prediction masks
    labeled_gt, num_gt_objects = ndi_label(gt_mask)
    labeled_pred, num_pred_objects = ndi_label(pred_mask)

    # If there are no objects in the ground-truth, all predicted objects are false positives
    if num_gt_objects == 0:
        return 0, num_pred_objects, 0

    # If there are no objects in the prediction, all ground-truth objects are false negatives
    if num_pred_objects == 0:
        return 0, 0, num_gt_objects

    # Calculate the center of mass for each object
    gt_centers = np.array(
        center_of_mass(gt_mask, labeled_gt, range(1, num_gt_objects + 1))
    )
    pred_centers = np.array(
        center_of_mass(pred_mask, labeled_pred, range(1, num_pred_objects + 1))
    )

    # Ensure centers are 2D arrays for distance calculation
    if gt_centers.ndim == 1:
        gt_centers = gt_centers.reshape(1, -1)
    if pred_centers.ndim == 1:
        pred_centers = pred_centers.reshape(1, -1)

    # Calculate the pairwise distances between ground-truth and predicted object centers
    dist_matrix = np.sqrt(
        np.sum(
            (gt_centers[:, np.newaxis, :] - pred_centers[np.newaxis, :, :]) ** 2,
            axis=-1,
        )
    )

    # Initialize tracking for matched ground-truth objects
    gt_matched = np.zeros(num_gt_objects, dtype=bool)
    true_positives = 0

    # Greedily match predicted objects to the closest available ground-truth object
    for i in range(num_pred_objects):
        # Find distances for the current predicted object
        distances = dist_matrix[:, i]

        # Get the indices of ground-truth objects sorted by distance
        sorted_gt_indices = np.argsort(distances)

        for gt_idx in sorted_gt_indices:
            # If the closest ground-truth object is within the distance threshold and not yet matched
            if distances[gt_idx] <= max_distance and not gt_matched[gt_idx]:
                true_positives += 1
                gt_matched[gt_idx] = True
                break  # Move to the next predicted object

    false_positives = num_pred_objects - true_positives
    false_negatives = num_gt_objects - np.sum(gt_matched)

    return true_positives, false_positives, false_negatives


class AneurysmRecallWithinXPixels(tm.Metric):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.min_aneurysm_size = self.cfg.min_aneurysm_size or 64
        self.max_pixel_tol = self.cfg.max_pixel_tol or 12
        self.add_state("matrix", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p, t = out["logits"], batch["y"]
        assert p.shape == t.shape, (
            f"prediction [{p.shape}] and label [{t.shape}] tensors are not same shape"
        )
        assert p.shape[0] == 1, f"expected 1 for p.shape[0], got {p.shape[0]}"

        p, t = p[0], t[0]
        # p, t are of shape (D, H, W)
        assert self.cfg.activation_fn == "sigmoid"
        p = p.sigmoid()

        # Convert to NumPy
        p, t = p.cpu().numpy(), t.cpu().numpy().astype(np.uint8)

        matrix = {}
        for each_thresh in self.cfg.thresholds:
            pb = (p >= each_thresh).astype(np.uint8)
            # Remove small objects
            if pb.sum() > 0:
                pb = remove_small_objects(
                    pb, min_size=self.min_aneurysm_size, connectivity=1
                )
            tp, fp, fn = evaluate_segmentation(t, pb, max_distance=self.max_pixel_tol)
            matrix[each_thresh] = tp, fp, fn

        self.matrix.append(matrix)

    def compute(self) -> Dict[str, torch.Tensor]:
        matrix_over_thresholds = {th: [] for th in self.cfg.thresholds}
        for threshold in self.cfg.thresholds:
            for mat in self.matrix:
                matrix_over_thresholds[threshold].append(mat[threshold])

        matrix_over_thresholds = {
            k: np.stack(v) for k, v in matrix_over_thresholds.items()
        }

        metrics_dict = {}
        for k, v in matrix_over_thresholds.items():
            metrics_dict[f"recall{k}"] = torch.tensor(
                np.mean(v[:, 0] / (v[:, 0] + v[:, 2]))
            )
            metrics_dict[f"precision{k}"] = torch.tensor(
                np.mean(v[:, 0] / (v[:, 0] + v[:, 1]))
            )

        metrics_dict["recall_max"] = torch.stack(
            [v for k, v in metrics_dict.items() if "recall" in k]
        ).amax()
        return metrics_dict


def pixel_level_metrics(
    gt_mask: torch.Tensor, pred_mask: torch.Tensor
) -> tuple[float, float]:
    """
    Calculates pixel-level recall and precision for 3D segmentation masks using PyTorch.

    Args:
        gt_mask (torch.Tensor): The 3D ground-truth binary mask (D, H, W).
        pred_mask (torch.Tensor): The 3D prediction binary mask (D, H, W).

    Returns:
        tuple[float, float]: A tuple containing the recall and precision values.
    """
    # Ensure the masks are boolean type for logical operations
    gt_mask = gt_mask.bool()
    pred_mask = pred_mask.bool()

    # True positives are pixels that are 1 in both masks
    true_positives = (pred_mask & gt_mask).sum().item()

    # False positives are pixels that are 1 in the prediction but 0 in the ground truth
    false_positives = (pred_mask & ~gt_mask).sum().item()

    # False negatives are pixels that are 0 in the prediction but 1 in the ground truth
    false_negatives = (~pred_mask & gt_mask).sum().item()

    # Calculate recall and precision
    # Recall = TP / (TP + FN)
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    # Precision = TP / (TP + FP)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )

    return recall, precision


def fbeta_score(recall, precision, beta=1.0):
    return ((1 + beta**2) * recall * precision) / (beta**2 * precision + recall + 1e-8)


class AneurysmPixelLevelPrecisionRecall(tm.Metric):
    def __init__(self, cfg: Config, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.add_state("matrix", default=[], dist_reduce_fx=None)
        self.add_state("auprc", default=[], dist_reduce_fx=None)

    def update(self, out: Dict, batch: Dict) -> None:
        p, t = out["logits"], batch["y"]
        assert p.shape == t.shape, (
            f"prediction [{p.shape}] and label [{t.shape}] tensors are not same shape"
        )
        assert p.shape[0] == 1, f"expected 1 for p.shape[0], got {p.shape[0]}"

        self.auprc.append(binary_auprc(p.ravel(), t.ravel()))

        p, t = p[0], t[0].long()
        # p, t are of shape (D, H, W)
        assert self.cfg.activation_fn == "sigmoid"
        p = p.sigmoid()

        matrix = []
        for each_thresh in self.cfg.thresholds:
            pb = (p >= each_thresh).long()
            recall, precision = pixel_level_metrics(t, pb)
            matrix.append(torch.tensor([each_thresh, recall, precision]))

        self.matrix.append(torch.stack(matrix).cuda())

    def compute(self) -> Dict[str, torch.Tensor]:
        matrix = torch.cat(self.matrix, dim=0)
        # matrix.shape = (num_samples * num_thresholds, 3)
        metrics_dict = {"auprc": torch.stack(self.auprc).mean()}
        for threshold in self.cfg.thresholds:
            metrics_dict[f"recall{threshold}"] = matrix[
                matrix[:, 0] == threshold, 1
            ].mean()
            metrics_dict[f"precision{threshold}"] = matrix[
                matrix[:, 0] == threshold, 2
            ].mean()

        # for threshold in self.cfg.thresholds:
        #     metrics_dict[f"f4_{threshold}"] = fbeta_score(
        #         metrics_dict[f"recall{threshold}"],
        #         metrics_dict[f"precision{threshold}"],
        #         beta=4,
        #     )

        metrics_dict["recall_max"] = torch.stack(
            [v for k, v in metrics_dict.items() if "recall" in k]
        ).amax()
        # metrics_dict["f4_max"] = torch.stack(
        #     [v for k, v in metrics_dict.items() if "f4" in k]
        # ).amax()

        return metrics_dict

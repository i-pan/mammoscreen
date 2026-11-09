"""
Custom losses for specific tasks (e.g., Kaggle competitions) that may or may not be
transferrable to other tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from skp.losses.classification import _check_shapes_equal


class AgeViewFemaleLoss(nn.Module):
    """
    Part of CheXmask training.
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.weights = params["weights"]

    def forward(self, out: Dict, batch: Dict) -> Dict:
        p, t = out["logits"], batch["y"]
        # only calculate view loss for CheXpert dataset
        # since NIH does not have any laterals
        # to avoid unwanted bias in the model
        valid_view = batch["valid_view"]
        age_loss = F.l1_loss(p[:, 0], t[:, 0].float())
        if valid_view.sum() > 0:
            view_loss = F.cross_entropy(p[valid_view, 1:4], t[valid_view, 1].long())
        else:
            view_loss = torch.tensor(0.0, device=p.device)
        female_loss = F.binary_cross_entropy_with_logits(p[:, 4], t[:, 2].float())
        loss_dict = {
            "age_loss": age_loss * self.weights[0],
            "view_loss": view_loss * self.weights[1],
            "female_loss": female_loss * self.weights[2],
        }
        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict


class L1CELoss(nn.Module):
    """
    Originally intended for bone age regression + classification.
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.l1_weight = params["l1_weight"]
        self.ce_weight = params["ce_weight"]
        self.l1_func = F.smooth_l1_loss if params.get("smooth_l1", False) else F.l1_loss

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_reg, p_cls, t_reg, t_cls = (
            out["logits0"],
            out["logits1"],
            batch["y"]["reg"],
            batch["y"]["cls"],
        )
        loss_dict = {}
        loss_dict["l1_loss"] = self.l1_func(p_reg, t_reg)
        loss_dict["ce_loss"] = F.cross_entropy(p_cls, t_cls)
        loss_dict["loss"] = (
            loss_dict["l1_loss"] * self.l1_weight
            + loss_dict["ce_loss"] * self.ce_weight
        )
        return loss_dict


class DoubleL1Loss(nn.Module):
    """
    Originally intended for bone age regression + classification.

    Considers outputs from 2 heads:
        1) Regression head (1 class)
        2) Classification head (240 classes, 1 for each month)
    """

    def __init__(self, params: Dict):
        super().__init__()
        self.reg_weight = params["reg_weight"]
        self.cls_weight = params["cls_weight"]
        self.l1_func = F.smooth_l1_loss if params.get("smooth_l1", False) else F.l1_loss

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_reg, p_cls, t = out["logits0"], out["logits1"], batch["y"]
        # logits -> probabilities
        p_cls = p_cls.softmax(dim=1)
        # probabilties over classes -> bone age scalar
        p_cls = p_cls * torch.arange(p_cls.size(1)).to(p_cls.device)
        p_cls = p_cls.sum(dim=1, keepdims=True)
        loss_dict = {}
        loss_dict["l1_loss_reg"] = self.l1_func(p_reg, t)
        loss_dict["l1_loss_cls"] = self.l1_func(p_cls, t)
        loss_dict["loss"] = (
            self.reg_weight * loss_dict["l1_loss_reg"]
            + self.cls_weight * loss_dict["l1_loss_cls"]
        ) / (self.reg_weight + self.cls_weight)
        return loss_dict


class MURA_BCE_CELoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.bce_weight = params.get("bce_weight", 1.0)
        self.ce_weight = params.get("ce_weight", 1.0)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        p0, t0 = p[:, 0], t[:, 0]
        p1, t1 = p[:, 1:], t[:, 1]
        loss_dict = {}
        loss_dict["bce_loss"] = F.binary_cross_entropy_with_logits(p0, t0.float())
        loss_dict["ce_loss"] = F.cross_entropy(p1, t1.long())
        loss_dict["loss"] = (
            loss_dict["bce_loss"] * self.bce_weight
            + loss_dict["ce_loss"] * self.ce_weight
        )
        return loss_dict


class MammoCancerWithAuxLosses(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.loss_weights = torch.tensor(params.get("loss_weights", [1.0] * 6))
        self.classes = [
            "cancer",
            "biopsy",
            "invasive",
            "difficult",
            "birads",
            "density",
        ]

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        if self.loss_weights.device != p.device:
            self.loss_weights = self.loss_weights.to(p.device)

        p0, t0 = p[:, :4], t[:, :4]  # cancer, biopsy, invasive, difficult_negative_case
        p1, t1 = p[:, [4, 5, 6]], t[:, 4]  # BIRADS - 3 classes
        p2, t2 = p[:, [7, 8, 9, 10]], t[:, 5]  # density - 4 classes
        loss_dict = {}

        bce_loss = F.binary_cross_entropy_with_logits(p0, t0.float(), reduction="none")
        bce_loss = self.loss_weights[:4] * bce_loss
        for i, c in enumerate(self.classes[:4]):
            loss_dict[f"{c}_loss"] = bce_loss[:, i].mean()

        birads_present = t1 != -1  # only present for one site
        if birads_present.sum() == 0:
            birads_loss = torch.tensor(0.0, device=p.device)
        else:
            birads_loss = F.cross_entropy(
                p1[birads_present], t1[birads_present].long(), reduction="none"
            )
        birads_loss = self.loss_weights[4] * birads_loss
        loss_dict["birads_loss"] = birads_loss.mean()

        density_present = t2 != -1  # only present for one site
        if density_present.sum() == 0:
            density_loss = torch.tensor(0.0, device=p.device)
        else:
            density_loss = F.cross_entropy(
                p2[density_present], t2[density_present].long(), reduction="none"
            )
        density_loss = self.loss_weights[5] * density_loss
        loss_dict["density_loss"] = density_loss.mean()

        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict


class ICHSeqMaskedBCELossWithAux(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.aux_weight = params.get("aux_weight", 0.4)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_seq, p_aux = out["logits_seq"], out["aux_logits_seq"]
        t_seq = batch["y_seq"]

        seq_loss = F.binary_cross_entropy_with_logits(p_seq, t_seq, reduction="none")
        aux_loss = F.binary_cross_entropy_with_logits(p_aux, t_seq, reduction="none")

        mask = batch["mask"]
        seq_loss = seq_loss[mask]
        aux_loss = aux_loss[mask]

        # upweight any by 2
        seq_loss = (seq_loss[:, 0] * 2 + seq_loss[:, 1:].sum(1)) / 7.0
        seq_loss = seq_loss.mean()
        aux_loss = (aux_loss[:, 0] * 2 + aux_loss[:, 1:].sum(1)) / 7.0
        aux_loss = aux_loss.mean()

        loss_dict = {"seq_loss": seq_loss, "aux_loss": aux_loss}
        loss_dict["loss"] = seq_loss + self.aux_weight * aux_loss

        return loss_dict


class ICHSeqClsMaskedLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.cls_weight = params.get("cls_weight", 1.0)
        self.seq_weight = params.get("seq_weight", 1.0)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_seq, p_cls = out["logits_seq"], out["logits_cls"]
        t_seq, t_cls = batch["y_seq"], batch["y_cls"]
        seq_loss = F.binary_cross_entropy_with_logits(p_seq, t_seq, reduction="none")
        seq_mask = batch["mask"]
        seq_loss = seq_loss[seq_mask]
        # upweight any by 2
        seq_loss = seq_loss[:, 0] * 2 + seq_loss[:, 1:].sum(1)
        seq_loss = seq_loss / 7.0
        seq_loss = seq_loss.mean()

        cls_loss = F.binary_cross_entropy_with_logits(p_cls, t_cls, reduction="none")
        # upweight any by 2
        cls_loss = cls_loss[:, 0] * 2 + cls_loss[:, 1:].sum(1)
        cls_loss = cls_loss / 7.0
        cls_loss = cls_loss.mean()

        loss_dict = {"seq_loss": seq_loss, "cls_loss": cls_loss}
        loss_dict["loss"] = self.cls_weight * cls_loss + self.seq_weight * seq_loss
        return loss_dict


class ICHLogLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p = out["logits"]
        t = batch["y"]
        assert p.size(1) == t.size(1) == 6
        loss = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        # loss.shape = (B, 6)
        sample_weights = torch.ones(loss.size(), device=p.device).float()
        # upweight any class by 2
        sample_weights[:, 0] = 2.0
        loss = loss.reshape(-1) * sample_weights.reshape(-1)
        loss = loss.sum() / sample_weights.sum()
        loss_dict = {"loss": loss}
        return loss_dict


class ICHLogLossSeqWithAux(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.aux_weight = params.get("aux_weight", 0.2)

    @staticmethod
    def calculate_loss(
        p: torch.Tensor, t: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        # loss.shape = (B, N, 6)
        B = loss.size(0)
        sample_weights = torch.ones(loss.size(), device=p.device).float()
        # upweight any class by 2
        sample_weights[:, :, 0] = 2.0
        sample_weights = sample_weights.reshape(B, -1)
        loss = loss.reshape(B, -1) * sample_weights
        mask = mask.reshape(B, -1)
        loss = loss[mask].sum() / sample_weights[mask].sum()
        return loss

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, p_aux = out["logits_seq"], out["aux_logits_seq"]
        t = batch["y_seq"]
        mask = batch["mask"].unsqueeze(2).expand(-1, -1, 6)
        assert p.size(2) == t.size(2) == 6
        seq_loss = self.calculate_loss(p, t, mask)
        aux_loss = self.calculate_loss(p_aux, t, mask)
        loss_dict = {}
        loss_dict["seq_loss"] = seq_loss
        loss_dict["aux_loss"] = aux_loss
        loss_dict["loss"] = seq_loss + self.aux_weight * aux_loss
        return loss_dict


class AneurysmSeqClsMaskedLoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.cls_weight = params.get("cls_weight", 1.0)
        self.seq_weight = params.get("seq_weight", 1.0)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_seq, p_cls = out["logits_seq"], out["logits_cls"]
        t_seq, t_cls = batch["y_seq"], batch["y_cls"]

        seq_loss = F.binary_cross_entropy_with_logits(
            p_seq, t_seq.float(), reduction="none"
        )
        seq_mask = batch["mask"]
        seq_loss = seq_loss[seq_mask]
        # Match comp metric
        seq_loss = seq_loss[:, 0] + seq_loss[:, 1:].mean(1)
        seq_loss = seq_loss.mean()

        # Match comp metric
        cls_loss = F.binary_cross_entropy_with_logits(
            p_cls, t_cls.float(), reduction="none"
        )
        cls_loss = cls_loss[:, 0] + cls_loss[:, 1:].mean(1)
        cls_loss = cls_loss.mean()

        loss_dict = {"seq_loss": seq_loss, "cls_loss": cls_loss}
        loss_dict["loss"] = self.cls_weight * cls_loss + self.seq_weight * seq_loss
        return loss_dict


class AneurysmBCE13(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p = out["logits"]
        t = batch["y"]
        assert p.size(1) == t.size(1) == 14
        loss = F.binary_cross_entropy_with_logits(p, t, reduction="none")
        # loss.shape = (B, 14)
        loss1 = loss[:, 0].mean()
        loss2 = loss[:, 1:].mean()
        loss_dict = {"loss": 0.5 * loss1 + 0.5 * loss2}
        return loss_dict


class CowSeqPlusBCE(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_seq = out["logits_seq"]
        t_seq = batch["y_cow"]
        loss_seq = F.binary_cross_entropy_with_logits(p_seq.float(), t_seq.float())

        p_cls = out["logits_cls"]
        p_modality, p_plane = p_cls[:, :4], p_cls[:, 4:]

        t_modality = F.one_hot(batch["y_modality"], num_classes=4)
        t_plane = F.one_hot(batch["y_plane"], num_classes=3)

        loss_modality = F.binary_cross_entropy_with_logits(
            p_modality.float(), t_modality.float()
        )
        loss_plane = F.binary_cross_entropy_with_logits(
            p_plane.float(), t_plane.float()
        )

        loss_dict = {
            "loss_seq": loss_seq,
            "loss_modality": loss_modality,
            "loss_plane": loss_plane,
            "loss": loss_seq + loss_modality + loss_plane,
        }
        return loss_dict


class CowSeqPlusBCE_CE(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.params = params

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p_seq = out["logits_seq"]
        t_seq = batch["y_cow"]
        loss_seq = F.binary_cross_entropy_with_logits(p_seq.float(), t_seq.float())

        p_cls = out["logits_cls"]
        p_modality, p_plane = p_cls[:, :4], p_cls[:, 4:]

        t_modality = batch["y_modality"]
        t_plane = batch["y_plane"]

        loss_modality = F.cross_entropy(p_modality.float(), t_modality.long())
        loss_plane = F.cross_entropy(p_plane.float(), t_plane.long())

        loss_dict = {
            "loss_seq": loss_seq,
            "loss_modality": loss_modality,
            "loss_plane": loss_plane,
            "loss": 2 * loss_seq + loss_modality + loss_plane,
        }
        return loss_dict


class BiomassWeightedL2WithAuxBCELoss(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        assert "weights" in params, "weights must be specified"
        self.weights = torch.tensor(params["weights"]).unsqueeze(0)
        self.l2_weight = params.get("l2_weight", 1.0)
        self.aux_bce_weight = params.get("aux_bce_weight", 0.4)
        assert self.weights.shape[0] == 1 and self.weights.ndim == 2

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        if self.weights.device != p.device:
            self.weights = self.weights.to(p.device)
        _check_shapes_equal(p, t)
        l2_loss = (
            (
                F.mse_loss(
                    p[:, :5].float(),
                    t[:, :5].float(),
                    reduction="none",
                )
                * self.weights
            )
            .sum(1)
            .mean()
        )
        bce_loss = F.binary_cross_entropy_with_logits(
            p[:, 5:].float(), t[:, 5:].float()
        )
        return {
            "l2_loss": l2_loss,
            "aux_bce_loss": bce_loss,
            "loss": l2_loss * self.l2_weight + bce_loss * self.aux_bce_weight,
        }


class BiomassWeightedL2WithAuxBCELossNDVI(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        assert "weights" in params, "weights must be specified"
        self.weights = torch.tensor(params["weights"]).unsqueeze(0)
        self.l2_weight = params.get("l2_weight", 1.0)
        self.aux_bce_weight = params.get("aux_bce_weight", 1.0)
        self.aux_ndvi_weight = params.get("aux_ndvi_weight", 1.0)
        assert self.weights.shape[0] == 1 and self.weights.ndim == 2

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        if self.weights.device != p.device:
            self.weights = self.weights.to(p.device)
        _check_shapes_equal(p, t)
        l2_loss = (
            (
                F.mse_loss(
                    p[:, :5].float(),
                    t[:, :5].float(),
                    reduction="none",
                )
                * self.weights
            )
            .sum(1)
            .mean()
        )
        bce_loss = F.binary_cross_entropy_with_logits(
            p[:, 5:-1].float(), t[:, 5:-1].float()
        )
        ndvi_loss = F.l1_loss(p[:, -1].sigmoid().float(), t[:, -1].float())
        return {
            "l2_loss": l2_loss,
            "aux_bce_loss": bce_loss,
            "aux_ndvi_loss": ndvi_loss,
            "loss": l2_loss * self.l2_weight
            + bce_loss * self.aux_bce_weight
            + ndvi_loss * self.aux_ndvi_weight,
        }


class BiomassL2(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        assert p.shape[1] == t.shape[1] == 5
        _check_shapes_equal(p, t)
        N = len(p)
        p = torch.cat([p[:, i] for i in range(5)], dim=0)
        t = torch.cat([t[:, i] for i in range(5)], dim=0)
        sample_weights = [0.1] * (N * 3) + [0.2] * N + [0.5] * N
        sample_weights = torch.tensor(sample_weights, device=p.device).float()
        loss = (
            F.mse_loss(p.float(), t.float(), reduction="none") * sample_weights
        ).mean()
        return {"loss": loss}


class BiomassL2CloverDeadPresent(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        # Create clover present and dead present labels
        t = torch.cat([t, t[:, 0].unsqueeze(1) > 0, t[:, 1].unsqueeze(1) > 0], dim=1)
        assert p.shape[1] == t.shape[1] == 7
        _check_shapes_equal(p, t)

        N = len(p)
        p2 = torch.cat([p[:, i] for i in range(5)], dim=0)
        t2 = torch.cat([t[:, i] for i in range(5)], dim=0)
        sample_weights = [0.1] * (N * 3) + [0.2] * N + [0.5] * N
        sample_weights = torch.tensor(sample_weights, device=p.device).float()
        mse_loss = (
            F.mse_loss(p2.float(), t2.float(), reduction="none") * sample_weights
        ).mean()
        present_loss = F.binary_cross_entropy_with_logits(
            p[:, 5:].float(), t[:, 5:].float()
        )
        return {
            "mse_loss": mse_loss,
            "present_loss": present_loss,
            "loss": mse_loss + present_loss,
        }


class BiomassL2CloverDeadPresentState(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        p, t, p_state, t_state = p[:, :7], t[:, :5], p[:, 7:], t[:, 5:]
        # Create clover present and dead present labels
        t = torch.cat([t, t[:, 0].unsqueeze(1) > 0, t[:, 1].unsqueeze(1) > 0], dim=1)
        assert p.shape[1] == t.shape[1] == 7
        _check_shapes_equal(p, t)

        N = len(p)
        p2 = torch.cat([p[:, i] for i in range(5)], dim=0)
        t2 = torch.cat([t[:, i] for i in range(5)], dim=0)
        sample_weights = [0.1] * (N * 3) + [0.2] * N + [0.5] * N
        sample_weights = torch.tensor(sample_weights, device=p.device).float()
        mse_loss = (
            F.mse_loss(p2.float(), t2.float(), reduction="none") * sample_weights
        ).mean()
        present_loss = F.binary_cross_entropy_with_logits(
            p[:, 5:].float(), t[:, 5:].float()
        )
        state_loss = F.binary_cross_entropy_with_logits(
            p_state.float(), t_state.float()
        )
        return {
            "mse_loss": mse_loss,
            "present_loss": present_loss,
            "state_loss": state_loss,
            "loss": mse_loss + present_loss + state_loss,
        }


class Biomass3Target(nn.Module):
    # For training on:
    #   1- Dry_Clover_g
    #   2- Dry_Dead_g
    #   3- Dry_Green_g
    def __init__(self, params: Dict):
        super().__init__()
        self.ignore_zeros = params.get("ignore_zeros", False)

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        _check_shapes_equal(p, t)
        mask1 = torch.tensor([True] * len(p), device=p.device)
        mask2 = torch.tensor([True] * len(p), device=p.device)
        mask3 = torch.tensor([True] * len(p), device=p.device)
        if self.ignore_zeros:
            mask1[t[:, 0] == 0] = False
            mask2[t[:, 1] == 0] = False
            mask3[t[:, 2] == 0] = False
        dry_clover_loss = F.mse_loss(p[mask1, 0].float(), t[mask1, 0].float())
        dry_dead_loss = F.mse_loss(p[mask2, 1].float(), t[mask2, 1].float())
        dry_green_loss = F.mse_loss(p[mask3, 2].float(), t[mask3, 2].float())
        loss = dry_clover_loss + dry_dead_loss + dry_green_loss
        return {
            "loss": loss,
            "dry_clover_loss": dry_clover_loss,
            "dry_dead_loss": dry_dead_loss,
            "dry_green_loss": dry_green_loss,
        }


class BiomassL2AuxMultiTarget(nn.Module):
    def __init__(self, params: Dict):
        super().__init__()
        self.aux_height_loss_weight = params.get("aux_height_loss_weight", 1.0)
        self.aux_binary_loss_weight = params.get("aux_binary_loss_weight", 1.0)

    def forward_biomass(self, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert p.shape[1] == t.shape[1] == 5
        _check_shapes_equal(p, t)
        N = len(p)
        p = torch.cat([p[:, i] for i in range(5)], dim=0)
        t = torch.cat([t[:, i] for i in range(5)], dim=0)
        sample_weights = [0.1] * (N * 3) + [0.2] * N + [0.5] * N
        sample_weights = torch.tensor(sample_weights, device=p.device).float()
        loss = (
            F.mse_loss(p.float(), t.float(), reduction="none") * sample_weights
        ).mean()
        return loss

    def forward(self, out: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        p, t = out["logits"], batch["y"]
        biomass_loss = self.forward_biomass(p[:, :5], t[:, :5])
        height_loss = F.l1_loss(p[:, 5].float(), t[:, 5].float())
        binary_loss = F.binary_cross_entropy_with_logits(
            p[:, 6:].float(), t[:, 6:].float()
        )
        loss_dict = {
            "biomass_loss": biomass_loss,
            "height_loss": height_loss,
            "binary_loss": binary_loss,
            "loss": biomass_loss
            + self.aux_height_loss_weight * height_loss
            + self.aux_binary_loss_weight * binary_loss,
        }
        return loss_dict

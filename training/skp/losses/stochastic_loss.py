import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSupervisionDiceBCELoss(nn.Module):
    def __init__(self, params):
        """
        Args:
            params (dict): Dictionary containing configuration parameters:
                - 'weight_dice' (float, default=1.0): Weight for Dice component.
                - 'weight_bce' (float, default=1.0): Weight for BCE component.
                - 'epsilon' (float, default=1e-6): Stability epsilon.
                - 'squared_pred' (bool, default=False): If True, use squares in Dice denominator.
                - 'to_onehot_y' (bool, default=False): If True, convert targets to one-hot.
                - 'ignore_background_channel' (bool, default=False): If True (and to_onehot_y is True),
                   assumes 0 is background, removes it, and expects preds to have C-1 channels.
                - 'class_weights' (torch.Tensor, optional): Tensor of shape (C,) for class balancing.
                - 'bce_pos_weight' (float, default=1.0): Weight for positive examples in BCE.
                - 'batch_dice' (bool, default=False): If True, calculate Dice over the whole batch (pseudovolume).
                   If False, calculate per sample and average.
                - 'ds_weights' (list, optional): List of weights for deep supervision levels.
                   Order: [original, ds1, ds2, ...].
        """
        super().__init__()
        self.w_dice = params.get("weight_dice", 1.0)
        self.w_bce = params.get("weight_bce", 1.0)
        self.eps = params.get("epsilon", 1e-6)
        self.squared_pred = params.get("squared_pred", False)
        self.to_onehot_y = params.get("to_onehot_y", False)
        self.ignore_background = params.get("ignore_background_channel", False)
        self.batch_dice = params.get("batch_dice", False)
        self.ds_weights = params.get("ds_weights", None)

        # Store class weights and BCE pos weight as buffer so they move to GPU with the module
        class_weights = params.get("class_weights", None)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights, persistent=False)
        else:
            self.class_weights = None

        bce_pos_weight = params.get("bce_pos_weight", 1.0)
        self.register_buffer(
            "bce_pos_weight", torch.tensor(bce_pos_weight), persistent=False
        )

    def forward(self, out, batch, mode="train"):
        """
        Args:
            out (dict): Must contain 'logits'. Optional keys: 'logits_ds1', 'logits_ds2', etc.
            batch (dict): Must contain 'y' (ground truth).
        """
        target_orig = batch["y"]

        # Identify available Deep Supervision keys in order: logits, logits_ds1, logits_ds2...
        ds_keys = ["logits"] + sorted(
            [k for k in out.keys() if k.startswith("logits_ds")],
            key=lambda x: int(x.replace("logits_ds", ""))
            if x.replace("logits_ds", "").isdigit()
            else 999,
        )

        # Validate weights (for training)
        if self.ds_weights is not None and mode == "train":
            assert (
                len(self.ds_weights) == len(ds_keys)
            ), f"Number of ds_weights ({len(self.ds_weights)}) must match number of outputs ({len(ds_keys)})."
            weights = self.ds_weights
        else:
            # Default to 1.0 for main, 0.0 for others if not specified (or 1.0 for all, usually strictly specified)
            # Here we assume equal weighting or just main if not provided.
            weights = [1.0] * len(ds_keys)

        total_loss = {"loss": 0.0}
        if self.w_dice > 0:
            total_loss["dice_loss"] = 0.0
        if self.w_bce > 0:
            total_loss["bce_loss"] = 0.0

        for i, key in enumerate(ds_keys):
            pred_logits = out[key]
            current_weight = weights[i]

            if current_weight == 0:
                continue

            if key == "logits":
                # Full resolution head will use stochastic weight sampling
                class_indices = out["class_indices"]
                # Ensure class_indices is in ascending order
                # This is mainly for if one-hot conversion is needed
                assert all(
                    class_indices[i] < class_indices[i + 1]
                    for i in range(len(class_indices) - 1)
                )
                if self.to_onehot_y:
                    target_resampled = self._convert_subset_one_hot(
                        target_orig, class_indices, self.ignore_background
                    )
                else:
                    # Assume ignore_background is already accounted by the user for if the target is already one-hot encoded
                    target_resampled = target_orig[:, class_indices]
            else:
                # Determine spatial target shape from current prediction
                target_shape = pred_logits.shape[2:]  # (D, H, W) or (H, W)

                if target_orig.shape[-3:] != target_shape:
                    # Use nearest neighbors downsampling since we are first downsampling the target
                    # before one-hot conversion to save memory versus initially doing one-hot encoding
                    # on the full resolution target and downsampling from there
                    mode = "nearest"

                    if self.to_onehot_y:
                        # Add channel dimension, then squeeze it off later
                        target_orig = target_orig.unsqueeze(1)

                    target_resampled = F.interpolate(
                        target_orig.float(),
                        size=target_shape,
                        mode=mode,
                    )

                    if self.to_onehot_y:
                        target_orig = target_orig.squeeze(1)
                        target_resampled = target_resampled.squeeze(1)

                if self.to_onehot_y:
                    target_resampled = self._convert_to_onehot(
                        target_resampled, pred_logits
                    )

            # Calculate loss for this level
            single_scale_loss_dict = self._compute_single_scale_loss(
                pred_logits, target_resampled
            )
            for k, v in single_scale_loss_dict.items():
                total_loss[k] += current_weight * v

        # Scale by sum of weights
        for k, v in total_loss.items():
            total_loss[k] = v / sum(weights)

        return total_loss

    def _convert_to_onehot(self, target, pred):
        """
        Converts indices to one-hot.
        Handles the logic for ignoring the background channel.
        """
        # Assume target is (B, 1, ...) or (B, ...) containing indices
        if target.ndim == pred.ndim:
            target = target.squeeze(1)

        # Determine number of classes for one-hot
        if self.ignore_background:
            # If we ignore background, the pred has C channels (classes 1..C).
            # The target has indices 0..C. We map 0->Background, 1..C->Classes.
            # So we need pred.shape[1] + 1 classes for the encoding.
            n_classes = pred.shape[1] + 1
        else:
            n_classes = pred.shape[1]

        # One hot: (B, spatial...) -> (B, spatial..., C)
        target_onehot = F.one_hot(target.long(), num_classes=n_classes)

        # Permute to (B, C, spatial...)
        dims = (0, -1) + tuple(range(1, target_onehot.ndim - 1))
        target_onehot = target_onehot.permute(dims).float()

        # If ignore background, strip the first channel (index 0)
        if self.ignore_background:
            target_onehot = target_onehot[:, 1:, ...]

        return target_onehot

    def _convert_subset_one_hot(
        self, target: torch.Tensor, class_indices: list, ignore_background: bool = False
    ) -> torch.Tensor:
        """
        One-hot encodes a target tensor based on a specific subset of classes.

        This function efficiently compares the target volume against the requested
        indices using broadcasting, avoiding slow loops.

        Args:
            target (torch.Tensor): Ground truth tensor of integers.
                                Shape: (B, D, H, W) or (B, H, W).
            class_indices (list[int]): A sorted list of class indices to extract.
                                    Length L.
            ignore_background (bool): If True, shifts the lookup logic so that
                                    Index 0 in the list maps to Label 1 in the target.
                                    (e.g., Use this if your list is [0, 1] referring
                                    to 'First Organ', 'Second Organ', but in the
                                    TotalSegmentator labels they are 1 and 2).

        Returns:
            torch.Tensor: One-hot encoded Float tensor.
                        Shape: (B, L, D, H, W).
        """
        # 1. Prepare Indices
        # Convert list to tensor and move to same device as target (e.g., GPU)
        if not isinstance(class_indices, torch.Tensor):
            indices_tensor = torch.tensor(
                class_indices, device=target.device, dtype=target.dtype
            )
        else:
            indices_tensor = class_indices.to(target.device)

        # 2. Apply Offset Logic
        if ignore_background:
            # User implies the list is 0-indexed relative to "Positive Classes",
            # so we shift +1 to match the "Background=0" convention of the target.
            indices_tensor = indices_tensor + 1

        # 3. Shape Manipulation for Broadcasting
        # We want to compare: Target [B, 1, D, H, W]  ==  Indices [1, L, 1, 1, 1]

        # Reshape target: Add a channel dimension at dim 1
        target_expanded = target.unsqueeze(1)

        # Reshape indices: [1, L, 1, 1, 1...]
        # This dynamically creates enough singleton dimensions to match the target
        view_shape = [1, len(indices_tensor)] + [1] * (target.ndim - 1)
        indices_expanded = indices_tensor.view(*view_shape)

        # 4. Vectorized Comparison
        # This generates a Boolean tensor [B, L, D, H, W] where True indicates a match
        mask = target_expanded == indices_expanded

        # 5. Return as Float (Standard for Dice/CE Loss)
        return mask.float()

    def _compute_single_scale_loss(self, logits, target):
        loss_dict = {"loss": 0.0}

        # Check shapes
        if logits.shape != target.shape:
            raise ValueError(
                f"Shape mismatch after processing: Pred {logits.shape} vs Target {target.shape}. "
                "Check 'to_onehot_y' and 'ignore_background_channel' params."
            )

        # Dice Loss
        if self.w_dice > 0:
            probs = torch.sigmoid(logits)
            dice = self._get_dice(probs, target)
            loss_dict["dice_loss"] = dice
            loss_dict["loss"] += self.w_dice * dice

        # BCE Loss
        if self.w_bce > 0:
            bce = self._get_bce(logits, target)
            loss_dict["bce_loss"] = bce
            loss_dict["loss"] += self.w_bce * bce

        return loss_dict

    def _get_dice(self, probs, target):
        """
        Calculates Soft Dice Loss.
        """
        # Dimensions to sum over
        # If batch_dice is True: sum over (B, spatial...) -> ONE score per class
        # If batch_dice is False: sum over (spatial...) -> (B) scores per class, then average B

        ndim = probs.ndim
        if self.batch_dice:
            # Sum over Batch (0) and Spatial (2, ...)
            dims = (0,) + tuple(range(2, ndim))
        else:
            # Sum over Spatial only
            dims = tuple(range(2, ndim))

        # Numerator
        intersection = (probs * target).sum(dim=dims)

        # Denominator
        if self.squared_pred:
            cardinality = (probs.pow(2) + target.pow(2)).sum(dim=dims)
        else:
            cardinality = (probs + target).sum(dim=dims)

        dice_score = (2.0 * intersection + self.eps) / (cardinality + self.eps)

        dice_loss = 1.0 - dice_score

        # Apply Class Weights if provided
        if self.class_weights is not None:
            # dice_loss shape: (C,) if batch_dice else (B, C)
            # class_weights shape: (C,)
            if not self.batch_dice:
                # broadcast weights to batch
                weights = self.class_weights.view(1, -1)
                dice_loss = dice_loss * weights
                return dice_loss.mean()  # Average over B and C
            else:
                dice_loss = dice_loss * self.class_weights
                return dice_loss.mean()  # Average over C
        else:
            return dice_loss.mean()

    def _get_bce(self, logits, target):
        """
        Calculates Binary Cross Entropy with Logits.
        """
        # If class weights are present, we use the weight arg in BCE
        # BCE weights usually expected as (C) or (B, C, ...)

        # Note: standard BCE weights are usually for positive/negative balance.
        # Here we are implementing Class Balancing.
        # F.binary_cross_entropy_with_logits supports 'weight' (per element) and 'pos_weight' (class imbalance).
        # We will manually multiply by class weights to ensure consistent behavior with Dice logic.

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=self.bce_pos_weight
        )

        # Average over spatial dims to get (B, C)
        # We assume independent pixels, usually mean over spatial is taken.
        ndim = logits.ndim
        spatial_dims = tuple(range(2, ndim))
        bce_loss = bce_loss.mean(dim=spatial_dims)  # Now (B, C)

        if self.class_weights is not None:
            weights = self.class_weights.view(1, -1)
            bce_loss = bce_loss * weights

        return bce_loss.mean()

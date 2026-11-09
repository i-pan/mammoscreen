import numpy as np
import torch.nn as nn

try:
    import webdataset as wds
except ModuleNotFoundError:
    print("webdataset not installed")
    print("If you want to use webdataset, install it with `pip install webdataset`")

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from skp.configs.base import Config
from skp.tasks import samplers


def build_dataloader(cfg: Config, dataset: Dataset, mode: str) -> DataLoader:
    if cfg.wds:
        if mode == "train":
            loader = wds.WebLoader(
                dataset, batch_size=None, num_workers=cfg.num_workers
            ).with_epoch(cfg.num_iterations_per_epoch)
        else:
            loader = wds.WebLoader(
                dataset, batch_size=None, num_workers=cfg.num_workers
            )
        return loader

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    dataloader_params = {}
    dataloader_params["num_workers"] = cfg.num_workers
    dataloader_params["drop_last"] = mode == "train"
    dataloader_params["shuffle"] = mode == "train"
    dataloader_params["pin_memory"] = cfg.pin_memory or False
    dataloader_params["persistent_workers"] = cfg.persistent_workers or False
    dataloader_params["collate_fn"] = dataset.collate_fn

    if mode == "train":
        dataloader_params["batch_size"] = cfg.batch_size
    else:
        dataloader_params["batch_size"] = cfg.val_batch_size or cfg.batch_size * 2

    sampler = None
    if cfg.sampler and cfg.sampler != "" and mode == "train":
        sampler = getattr(samplers, cfg.sampler)(dataset=dataset, cfg=cfg)

    if sampler:
        dataloader_params["shuffle"] = False
        if cfg.args["strategy"] == "ddp":
            sampler = samplers.DistributedSamplerWrapper(sampler)
        print(f"Using sampler {sampler} for training ...")
        dataloader_params["sampler"] = sampler
    elif cfg.args["strategy"] == "ddp":
        dataloader_params["shuffle"] = False
        dataloader_params["sampler"] = DistributedSampler(
            dataset, shuffle=mode == "train"
        )

    loader = DataLoader(dataset, **dataloader_params, worker_init_fn=worker_init_fn)

    return loader


class ParameterGrouper:
    def __init__(self, groups: list):
        """
        Logic for splitting model parameters into different optimizer groups based on their names.

        Args:
            groups (list[dict]): A list of dictionaries defining the grouping rules.
                Each dict can contain:
                - 'match' (str): The substring to look for in the parameter name (e.g., "encoder").
                - 'lr_scale' (float): Multiplier for the global base_lr.
                - 'lr' (float): Absolute learning rate (overrides lr_scale).
                - 'weight_decay' (float): Specific weight decay (optional).

        IMPORTANT - MATCHING LOGIC (ORDER MATTERS):
        -------------------------------------------
        The matching is performed sequentially (First Match Wins).
        Once a parameter matches a group's substring, it is assigned to that group
        and the code stops checking subsequent groups.

        You must list MORE SPECIFIC rules BEFORE general rules.

        Example:
            If you want the 'encoder' to have LR=1e-5, but 'encoder.layer4' to have LR=1e-3:

            CORRECT ORDER:
            [
                {'match': 'layer4',  'lr': 1e-3},  # Specific override first
                {'match': 'encoder', 'lr': 1e-5}   # General rule second
            ]

            INCORRECT ORDER:
            [
                {'match': 'encoder', 'lr': 1e-5},  # Catches everything in encoder (including layer4)
                {'match': 'layer4',  'lr': 1e-3}   # Never reached
            ]
        """
        self.groups = groups

    def __call__(self, model: nn.Module, base_lr: float, base_wd: float):
        """
        Categorizes the model's parameters and returns the list of groups expected by PyTorch optimizers.
        """
        # 1. Bucket parameters
        categorized_params = {i: [] for i in range(len(self.groups))}
        default_params = []

        # 2. Iterate over all named parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Find which group this parameter belongs to
            assigned = False
            for i, group_cfg in enumerate(self.groups):
                # Check if substring is in the full parameter name
                if group_cfg["match"] in name:
                    categorized_params[i].append(param)
                    assigned = True
                    break  # Stop checking after first match (Priority Logic)

            if not assigned:
                default_params.append(param)

        # 3. Construct the optimizer input list
        optimizer_groups = []

        # Add specific groups
        for i, params in categorized_params.items():
            if not params:
                continue

            cfg = self.groups[i]

            # Determine LR: Absolute or Scaled
            if "lr" in cfg:
                final_lr = cfg["lr"]
            else:
                final_lr = base_lr * cfg.get("lr_scale", 1.0)

            # Determine WD: Override or Base
            final_wd = cfg.get("weight_decay", base_wd)

            optimizer_groups.append(
                {
                    "params": params,
                    "lr": final_lr,
                    "weight_decay": final_wd,
                    "name": cfg["match"],  # Helpful for debugging
                }
            )

        # Add default group (leftovers)
        if default_params:
            optimizer_groups.append(
                {
                    "params": default_params,
                    "lr": base_lr,
                    "weight_decay": base_wd,
                    "name": "default",
                }
            )

        return optimizer_groups

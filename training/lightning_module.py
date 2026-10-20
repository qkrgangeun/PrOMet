"""Lightning wrappers used to train the EGNN (loss terms as in the original training code)."""
from typing import Type

import torch
import torch.nn as nn
from lightning import LightningDataModule, LightningModule  # type: ignore
from torch.utils.data import DataLoader

from promet.constants import metal_counts, metal_counts_focus
from dataset import PreprocessedDataSet


class LigMetModel(LightningModule):
    def __init__(self, model: Type[nn.Module], model_config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model = model(**model_config)

        # class-balanced weights w_c ~ 1 / M_c (last entry of metal_counts is the "no metal" class)
        self.register_buffer(
            "metal_weight",
            torch.tensor([10000 / metal_counts.get(m, 10000) for m in metal_counts], dtype=torch.float32))
        self.register_buffer(
            "metal_weight_focus",
            torch.tensor([10000 / metal_counts_focus.get(m, 10000) for m in metal_counts_focus], dtype=torch.float32))
        self.register_buffer("pos_weight", torch.tensor([2], dtype=torch.float32))

        self.loss_fns = nn.ModuleDict({
            "BCE": nn.BCEWithLogitsLoss(pos_weight=self.pos_weight),  # occupancy loss
            "CE": nn.CrossEntropyLoss(weight=self.metal_weight),  # type loss (metals + none)
            "CEfocus": nn.CrossEntropyLoss(weight=self.metal_weight_focus),  # auxiliary type loss, d <= 2 A
        })

    def forward(self, G):
        return self.model(G)

    def compute_loss(self, pred, label, type_pred):
        logs = {}
        loss = self.loss_fns["BCE"](pred.squeeze(-1), label[..., 0])
        logs["BCE Loss"] = loss.item()

        ce_loss = self.loss_fns["CE"](type_pred, label[..., 1].long())
        logs["CE Loss"] = ce_loss.item()
        loss = loss + ce_loss

        mask = label[..., 0] > 0.5  # soft label > 0.5 <=> within 2 A of a metal
        if mask.any():
            focus = self.loss_fns["CEfocus"](type_pred[mask][..., :-1], label[..., 1].long()[mask])
            logs["CE Focus Loss"] = focus.item()
            loss = loss + 2 * focus
        return loss, logs

    def _step(self, batch, stage):
        G, label, info = batch
        pred, type_pred, _ = self(G.to(self.device))
        grididx = torch.where(G.ndata["grid_mask"] > 0)[0]
        loss, logs = self.compute_loss(pred[grididx], label, type_pred[grididx])
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=stage == "val", sync_dist=stage == "val")
        for k, v in logs.items():
            self.log(f"{stage}_{k}", v, on_epoch=True, sync_dist=stage == "val")
        return loss, pred[grididx], type_pred[grididx], label

    def training_step(self, batch, batch_idx):
        loss, *_ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, type_pred, label = self._step(batch, "val")
        label_05 = label[..., 0] > 0.5
        pred_05 = torch.sigmoid(pred.squeeze(-1)) > 0.5
        tp = torch.logical_and(label_05, pred_05).sum().item()
        self.log("precision", tp / (pred_05.sum().item() + 1e-6), on_epoch=True, sync_dist=True)
        self.log("recall", tp / (label_05.sum().item() + 1e-6), on_epoch=True, sync_dist=True)
        if label_05.any():
            top1 = type_pred[label_05][..., :-1].argmax(dim=1)
            acc = (top1 == label[..., 1].long()[label_05]).float().mean().item()
            self.log("type_accuracy", acc, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class LigMetDataModule(LightningDataModule):
    def __init__(self, train_data_file: str, val_data_file: str, preprocessed: dict,
                 train_loader_params: dict, val_loader_params: dict, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_data_file = train_data_file
        self.val_data_file = val_data_file
        self.preprocessed = preprocessed
        self.train_loader_params = train_loader_params
        self.val_loader_params = val_loader_params

    def setup(self, stage=None):
        self.train_dataset = PreprocessedDataSet(data_file=self.train_data_file, **self.preprocessed)
        self.valid_dataset = PreprocessedDataSet(data_file=self.val_data_file, **self.preprocessed)

    # Lightning wraps these in a DistributedSampler when training with DDP.
    def train_dataloader(self):
        params = {**self.train_loader_params, "shuffle": True}
        return DataLoader(self.train_dataset, collate_fn=self.train_dataset.collate, **params)

    def val_dataloader(self):
        params = {**self.val_loader_params, "shuffle": False}
        return DataLoader(self.valid_dataset, collate_fn=self.valid_dataset.collate, **params)

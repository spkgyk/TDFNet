import torch
import warnings
import subprocess
import pytorch_lightning as pl

from collections.abc import MutableMapping
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")


def get_gpu_utilization():
    cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    utilization = subprocess.check_output(cmd, shell=True)
    utilization = utilization.decode("utf-8").strip().split("\n")
    utilization = max([int(x.replace(" MiB", "")) for x in utilization]) / 1000
    return utilization


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class System(pl.LightningModule):
    default_monitor: str = "val_loss"

    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        scheduler=None,
        config=None,
        train_video_model=False,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        self.train_video_model = train_video_model
        self.save_hyperparameters(self.config_to_hparams(self.config))

    def forward(self, wav: torch.Tensor, mouth: torch.Tensor = None):
        if self.video_model == None:
            return self.audio_model(wav)
        else:
            if not self.train_video_model:
                with torch.no_grad():
                    mouth_emb = self.video_model(mouth.type_as(wav))
            else:
                mouth_emb = self.video_model(mouth.type_as(wav))
            return self.audio_model(wav, mouth_emb)

    def common_step(self, batch, batch_nb, is_train=True):
        if self.video_model == None:
            if self.config["training"]["online_mix"] == True:
                inputs, targets, _ = self.online_mixing_collate(batch)
            else:
                inputs, targets, _ = batch
            est_targets = self(inputs)
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            if is_train:
                loss = self.loss_func["train"](est_targets, targets)
            else:
                loss = self.loss_func["val"](est_targets, targets)
            return loss
        elif self.video_model != None:
            inputs, targets, target_mouths, _ = batch
            est_targets = self(inputs, target_mouths)
            if targets.ndim == 2:
                targets = targets.unsqueeze(1)
            if is_train:
                loss = self.loss_func["train"](est_targets, targets)
            else:
                loss = self.loss_func["val"](est_targets, targets)
            return loss

    def training_step(self, batch, batch_nb):
        loss = self.common_step(batch, batch_nb)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("memory", get_gpu_utilization(), on_epoch=True, prog_bar=True, sync_dist=True)
        return {"loss": loss}

    def training_step_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_loss = torch.mean(self.all_gather(avg_loss))
        self.logger.experiment.add_scalar("train_sisnr", -train_loss, self.current_epoch)

    def validation_step(self, batch, batch_nb):
        loss = self.common_step(batch, batch_nb, is_train=False)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {"val_loss": loss}

    def validation_step_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_loss = torch.mean(self.all_gather(avg_loss))
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True, prog_bar=True, sync_dist=True)
        self.logger.experiment.add_scalar("learning_rate", self.optimizer.param_groups[0]["lr"], self.current_epoch)
        self.logger.experiment.add_scalar("val_sisnr", -val_loss, self.current_epoch)

    def configure_optimizers(self):
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def online_mixing_collate(batch):
        inputs, targets = batch
        batch, n_src, _ = targets.shape

        energies = torch.sum(targets**2, dim=-1, keepdim=True)
        new_src = []
        for i in range(targets.shape[1]):
            new_s = targets[torch.randperm(batch), i, :]
            new_s = new_s * torch.sqrt(energies[:, i] / (new_s**2).sum(-1, keepdims=True))
            new_src.append(new_s)

        targets = torch.stack(new_src, dim=1)
        inputs = targets.sum(1)
        return inputs, targets

    def on_epoch_end(self):
        if self.config["sche"]["patience"] > 0 and self.config["training"]["divide_lr_by"] != None:
            if self.current_epoch % self.config["sche"]["patience"] == 0 and self.current_epoch != 0:
                new_lr = self.config["optim"]["lr"] / (
                    self.config["training"]["divide_lr_by"] ** (self.current_epoch // self.config["sche"]["patience"])
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr

    @staticmethod
    def config_to_hparams(dic):
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic

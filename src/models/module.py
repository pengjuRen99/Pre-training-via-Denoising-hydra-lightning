from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torch.nn.functional import mse_loss, l1_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.components.priors import BasePrior
from src.models.components.model import create_model, load_model


class PlateauScheduler(ReduceLROnPlateau):
    def __init__(self, factor, patience):
        # from torch._six import inf

        self.factor = factor
        self.patience = patience
        self.threshold = 1e-4
        self.mode = "min"
        self.threshold_mode = "rel"
        self.best = float("Inf")
        self.num_bad_epochs = None
        self.eps = 1e-8
        self.last_epoch = 0

    def step(self, metrics, epoch=None):
        current = float(metrics)
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.num_bad_epochs = 0
            return self.factor

        return 1.0



class LNNP(LightningModule):
    """
    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        extras: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        net: torch.nn.Module,
        compile: bool,
        prior_model: BasePrior,
        mean: float,
        std: float,
    ) -> None:
        """Initialize a `LNNP`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        if self.net.load_model:
            self.model = load_model(self.net.load_model, args=self.net)
        elif self.net.pretrained_model:
            self.model = load_model(self.net.pretrained_model, args=self.net, mean=mean, std=std)
        else:
            self.model = create_model(self.net, prior_model=prior_model, mean=mean, std=std)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()
        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()
        # initialize orther lr_scheduler
        self.last_epoch = 0
        self.lr_gen_scheduler = PlateauScheduler(
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch=None) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.
        """
        return self.model(z, pos, batch=batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], loss_fn: Any = None, stage: str = "train"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        with torch.set_grad_enabled(stage == "train" or self.hparams.net.derivative):
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            pred, noise_pred, deriv = self(batch.z, batch.pos, batch.batch)

        denoising_is_on = ("pos_target" in batch) and (self.hparams.extras.denoising_weight > 0) and (noise_pred is not None)

        loss_y, loss_dy, loss_pos = 0, 0, 0
        if self.hparams.net.derivative:
            if "y" not in batch:
                # "use" both outputs of the model's forward function but discard the first
                # to only use the derivative and avoid 'Expected to have finished reduction
                # in the prior iteration before starting a new one.', which otherwise get's
                # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
                deriv = deriv + pred.sum() * 0

            # force/derivative loss
            loss_dy = loss_fn(deriv, batch.dy)

            if stage in ["train", "val"] and self.hparams.extras.ema_alpha_dy < 1:
                if self.ema[stage + "_dy"] is None:
                    self.ema[stage + "_dy"] = loss_dy.detach()
                # apply exponential smoothing over batches to dy
                loss_dy = (
                    self.hparams.extras.ema_alpha_dy * loss_dy
                    + (1 - self.hparams.extras.ema_alpha_dy) * self.ema[stage + "_dy"]
                )
                self.ema[stage + "_dy"] = loss_dy.detach()

            if self.hparams.extras.force_weight > 0:
                self.losses[stage + "_dy"].append(loss_dy.detach())

        if "y" in batch:
            if (noise_pred is not None) and not denoising_is_on:
                # "use" both outputs of the model's forward (see comment above).
                pred = pred + noise_pred.sum() * 0

            if batch.y.ndim == 1:
                batch.y = batch.y.unsqueeze(1)

            # energy/prediction loss
            loss_y = loss_fn(pred, batch.y)

            if stage in ["train", "val"] and self.hparams.extras.ema_alpha_y < 1:
                if self.ema[stage + "_y"] is None:
                    self.ema[stage + "_y"] = loss_y.detach()
                # apply exponential smoothing over batches to y
                loss_y = (
                    self.hparams.extras.ema_alpha_y * loss_y
                    + (1 - self.hparams.extras.ema_alpha_y) * self.ema[stage + "_y"]
                )
                self.ema[stage + "_y"] = loss_y.detach()

            if self.hparams.extras.energy_weight > 0:
                self.losses[stage + "_y"].append(loss_y.detach())

        if denoising_is_on:
            if "y" not in batch:
                # "use" both outputs of the model's forward (see comment above).
                noise_pred = noise_pred + pred.sum() * 0
                
            normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
            loss_pos = loss_fn(noise_pred, normalized_pos_target)
            self.losses[stage + "_pos"].append(loss_pos.detach())

        # total loss
        loss = loss_y * self.hparams.extras.energy_weight + loss_dy * self.hparams.extras.force_weight + loss_pos * self.hparams.extras.denoising_weight

        self.losses[stage].append(loss.detach())

        # Frequent per-batch logging for training
        if stage == 'train':
            train_metrics = {k + "_per_step": v[-1] for k, v in self.losses.items() if (k.startswith("train") and len(v) > 0)}
            train_metrics['lr_per_step'] = self.trainer.optimizers[0].param_groups[0]["lr"]
            train_metrics['step'] = self.trainer.global_step   
            train_metrics['batch_pos_mean'] = batch.pos.mean().item()
            self.log_dict(train_metrics, sync_dist=True)

        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        return self.model_step(batch, mse_loss, "train")

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, *args) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if len(args) == 0 or (len(args) > 0 and args[0] == 0):
            # validation step
            return self.model_step(batch, mse_loss, "val")
        # test step
        return self.model_step(batch, l1_loss, "test")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": self.current_epoch,
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()

            # if prediction and derivative are present, also log them separately
            if len(self.losses["train_y"]) > 0 and len(self.losses["train_dy"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
                result_dict["train_loss_dy"] = torch.stack(
                    self.losses["train_dy"]
                ).mean()
                result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
                result_dict["val_loss_dy"] = torch.stack(self.losses["val_dy"]).mean()

                if len(self.losses["test"]) > 0:
                    result_dict["test_loss_y"] = torch.stack(
                        self.losses["test_y"]
                    ).mean()
                    result_dict["test_loss_dy"] = torch.stack(
                        self.losses["test_dy"]
                    ).mean()

            if len(self.losses["train_y"]) > 0:
                result_dict["train_loss_y"] = torch.stack(self.losses["train_y"]).mean()
            if len(self.losses['val_y']) > 0:
              result_dict["val_loss_y"] = torch.stack(self.losses["val_y"]).mean()
            if len(self.losses["test_y"]) > 0:
                result_dict["test_loss_y"] = torch.stack(
                    self.losses["test_y"]
                ).mean()

            # if denoising is present, also log it
            if len(self.losses["train_pos"]) > 0:
                result_dict["train_loss_pos"] = torch.stack(
                    self.losses["train_pos"]
                ).mean()

            if len(self.losses["val_pos"]) > 0:
                result_dict["val_loss_pos"] = torch.stack(
                    self.losses["val_pos"]
                ).mean()

            if len(self.losses["test_pos"]) > 0:
                result_dict["test_loss_pos"] = torch.stack(
                    self.losses["test_pos"]
                ).mean()

            self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        return self.model_step(batch, l1_loss, "test")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.extras.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.extras.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.optimizer.keywords['lr']
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                return{
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                }
            # Fallback for other schedulers, assuming they are step-based.
            # lr_schedule=cosine
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}
    

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
            "train_y": [],
            "val_y": [],
            "test_y": [],
            "train_dy": [],
            "val_dy": [],
            "test_dy": [],
            "train_pos": [],
            "val_pos": [],
            "test_pos": [],
        }

    def _reset_ema_dict(self):
        self.ema = {"train_y": None, "val_y": None, "train_dy": None, "val_dy": None}


if __name__ == "__main__":
    _ = LNNP(None, None, None, None, None, None, None, None, None)

from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from os.path import join
from torch.utils.data import Subset
from pytorch_lightning.utilities import rank_zero_warn
from torch_scatter import scatter
from tqdm import tqdm
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import components
from src.data.utils import make_splits, MissingEnergyException


class DataModule(LightningDataModule):
    """
    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/PCQM4MV2/",
        dataset_name: str = "PCQM4MV2",
        batch_size: int = 64,
        inference_batch_size: int=70,

        splits: str = None,
        train_val_test_split: Tuple[int, int, int] = [None, 5_000, 10_000],
        
        num_workers: int = 0,
        pin_memory: bool = False,

        data_arg: str = None,
        coord_files: str = None,
        embed_files: str = None,
        energy_files: str = None,
        force_files: str = None,
        energy_weight: float = 0.0,
        force_weight: float = 1.0,
        position_noise_scale: float = 0.0,
        denoising_weight: float = 1.0,
        denoising_only: bool = True,
        standardize: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset: Optional[Data] = None
        self.data_train: Optional[Data] = None
        self.data_val: Optional[Data] = None
        self.data_test: Optional[Data] = None

        self.batch_size_per_device = batch_size
        self._mean, self._std = None, None
        self._saved_dataloaders = dict()


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.hparams.dataset_name == "Custom":
            self.dataset = components.Custom(
                self.hparams.coord_files,
                self.hparams.embed_files,
                self.hparams.energy_files,
                self.hparams.force_files,
            )
        else:
            if self.hparams.position_noise_scale > 0.:
                def transform(data):
                    noise = torch.randn_like(data.pos) * self.hparams.position_noise_scale
                    data.pos_target = noise
                    data.pos = data.pos + noise
                    return data
            else:
                transform = None
            
            dataset_factory = lambda t: getattr(components, self.hparams.dataset_name)(
                self.hparams.data_dir,
                dataset_arg = self.hparams.data_arg,
                transform=t,
            )

            # Noisy version of dataset
            self.dataset_maybe_noisy = dataset_factory(transform)
            # Clean version of dataset
            self.dataset = dataset_factory(None)

        # 为了一部分是有噪声的，一部分是无噪声的，因此用索引的方式分割
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(self.dataset),
            self.hparams.train_val_test_split[0],
            self.hparams.train_val_test_split[1],
            self.hparams.train_val_test_split[2],
            seed=42,
            filename=join(self.hparams.data_dir, "splits.npz"),
            splits=self.hparams.splits,
        )
        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )

        self.train_dataset = Subset(self.dataset_maybe_noisy, self.idx_train)
        if self.hparams.denoising_only:
            self.val_dataset = Subset(self.dataset_maybe_noisy, self.idx_val)
            self.test_dataset = Subset(self.dataset_maybe_noisy, self.idx_test)
        else:
            self.val_dataset = Subset(self.dataset, self.idx_val)
            self.test_dataset = Subset(self.dataset, self.idx_test)
        
        if self.hparams.standardize:
            self._standardize()

    def train_dataloader(self):
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        loaders = [self._get_dataloader(self.val_dataset, "val")]
        if (
            len(self.test_dataset) > 0
            # and self.trainer.current_epoch % self.hparams.test_interval == 0
        ):
            loaders.append(self._get_dataloader(self.test_dataset, "test"))
        return loaders

    def test_dataloader(self):
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._get_dataloader(self.test_dataset, "test")
    
    @property
    def atomref(self):
        if hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std
    
    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = (
            store_dataloader and not self.trainer.reload_dataloaders_every_n_epochs
        )
        if stage in self._saved_dataloaders and store_dataloader:
            # storing the dataloaders like this breaks calls to trainer.reload_train_val_dataloaders
            # but makes it possible that the dataloaders are not recreated on every testing epoch
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams.inference_batch_size
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,               
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
        return dl

    def _standardize(self):
        def get_energy(batch, atomref):
            if batch.y is None:
                raise MissingEnergyException()

            if atomref is None:
                return batch.y.clone()

            # remove atomref energies from the target energy
            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone()

        data = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            # only remove atomref energies if the atomref prior is used
            atomref = self.atomref if self.hparams["prior_model"] == "Atomref" else None
            # extract energies from the data
            ys = torch.cat([get_energy(batch, atomref) for batch in data])
        except MissingEnergyException:
            rank_zero_warn(
                "Standardize is true but failed to compute dataset mean and "
                "standard deviation. Maybe the dataset only contains forces."
            )
            return

        # compute mean and standard deviation
        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
    


    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = DataModule()
    _.prepare_data()
    _.setup("fit")


import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from torch_geometric.data import download_url, extract_zip
import os


class QM9SP(QM9_geometric):     # 三类光谱，需要一个list，[uv, ir, raman]
    def __init__(self, root, transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(qm9_target_dict.values())}.'
        )

        self.raw_url = 'https://figshare.com/ndownloader/articles/24235333/versions/3'

        self.label = dataset_arg
        if dataset_arg == "alpha":   # set this value as placeholder during pre-training
            self.label = "isotropic_polarizability"
        elif dataset_arg in ["U", "U0"]:
            self.label = "energy_" + dataset_arg

        label2idx = dict(zip(qm9_target_dict.values(), qm9_target_dict.keys()))
        self.label_idx = label2idx[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            transform = Compose([transform, self._filter_label])

        self.root = root

        super(QM9SP, self).__init__(root, transform=transform)

    @property
    def processed_file_names(self) -> str:
        # return "data_with_uv_ir_raman.pt"
        return "data_with_uv_ir_raman_offical.pt"

    def get_atomref(self, max_z=100):
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def _filter_label(self, batch):
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    def download(self):
        file_path = download_url(self.raw_url, self.raw_dir, filename='3')
        extract_zip(file_path, self.raw_dir)
        os.unlink(file_path)

    def process(self):
        # 要提前下载QM9数据集，默认地址在../data/QM9/raw/qm9_v3.pt       使用的qm9V3
        # qm9sp
        data_qm9sp = torch.load(os.path.join(self.raw_dir, "qm9s.pt"))           # root: ../data/QM9SP
        # qm9
        print(os.path.dirname(self.root))
        data_list_qm9 = torch.load(os.path.join(os.path.dirname(self.root), "QM9/raw/qm9_v3.pt"))                # root: ../data/QM9/raw/qm9_v3.pt
        # print


if __name__ == "__main__":
    dataset = QM9SP(root="/home/RenPengju/code/MultiModal_Spectra/Pretrain_task/Pre-training-via-Denoising-hydra-lightning/data/QM9SP", dataset_arg="homo")
    print(dataset)
    print(dataset[0])

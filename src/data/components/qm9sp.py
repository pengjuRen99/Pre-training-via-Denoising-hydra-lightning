import torch
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.nn.models.schnet import qm9_target_dict
from torch_geometric.data import download_url, extract_zip
import os
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm 


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
        # return "data_with_uv_ir_raman_offical.pt"
        return "data_with_uv_ir_raman_collate.pt"

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
        pass
        # file_path = download_url(self.raw_url, self.raw_dir, filename='3')
        # extract_zip(file_path, self.raw_dir)
        # os.unlink(file_path)

    def process(self):
        # 要提前下载QM9数据集，默认地址在../data/QM9/raw/qm9_v3.pt       使用的qm9V3
        # qm9sp
        # edge_index, pos, number, smile, z, quadrupole, octapole, npacharge, dipole, polar, hyperpolar, energy, Hij, Hi, dedipole, depolar, train_dipole, tran_energy,  
        data_qm9sp = torch.load(os.path.join(self.raw_dir, "qm9s.pt"), weights_only=False)           # root: ../data/QM9SP
        # print(len(data_qm9sp))      # 129817
        # qm9
        # x, edge_index, edge_attr, y, pos, z, smiles, name, idx 
        # qm9_v3中没有smiles，data_v3中是包含的
        data_qm9 = torch.load(os.path.join(os.path.dirname(self.root), "QM9/raw/qm9_v3.pt"), weights_only=False)                # root: ../data/QM9/raw/qm9_v3.pt
        # print(len(data_list_qm9))    # 130831  # 
        # spectra
        # index, 129817*702
        UV_spectra = pd.read_csv(os.path.join(self.raw_dir, "uv_boraden.csv"))
        IR_spectra = pd.read_csv(os.path.join(self.raw_dir, "ir_boraden.csv"))
        Raman_spectra = pd.read_csv(os.path.join(self.raw_dir, "raman_boraden.csv"))
        print("Preprocessing QM9 data for fast lookup...")
        # need augment: x, edge_index, edge_attr, y, pos, z, name, idx, uv, ir, raman
        qm9_dict = {item['name']: item for item in data_qm9}
        data_list = []
        print("Processing and merging QM9SP data...")
        for i, qm9sp_entry in enumerate(tqdm(data_qm9sp, desc="Processing molecules")):
            name_key = "gdb_" + str(qm9sp_entry.number)
            if name_key in qm9_dict:
                qm9_entry = qm9_dict[name_key]
                uv_tensor = torch.from_numpy(UV_spectra.iloc[i].values[1:]).float()
                ir_tensor = torch.from_numpy(IR_spectra.iloc[i].values[1:]).float()
                raman_tensor = torch.from_numpy(Raman_spectra.iloc[i].values[1:]).float()
                data = Data(
                    x=qm9_entry['x'],
                    edge_index=qm9_entry['edge_index'],
                    edge_attr=qm9_entry['edge_attr'],
                    y=qm9_entry['y'],
                    pos=qm9_entry['pos'],
                    z=qm9_entry['z'],
                    name=qm9_entry['name'],
                    idx=qm9_entry['idx'],
                    uv=uv_tensor.reshape(-1, 701),
                    ir=ir_tensor.reshape(-1, 3501),
                    raman=raman_tensor.reshape(-1, 3501),
                )
                data_list.append(data)
        print(f"Successfully processed and matched {len(data_list)} molecules.")
        print("Collating data into a single graph object...")
        data, slices = self.collate(data_list)
        print(f"Saving processed data to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])
        print("Processing finished.")


if __name__ == "__main__":
    dataset = QM9SP(root="/home/RenPengju/code/MultiModal_Spectra/Pretrain_task/Pre-training-via-Denoising-hydra-lightning/data/QM9SP", dataset_arg="homo")
    print(dataset)
    print(dataset[0])

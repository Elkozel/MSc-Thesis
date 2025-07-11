from typing import Any, List
import pandas as pd
import numpy as np
import torch
from datasets.LANL_local import LANLL
from torch_geometric.data import HeteroData


class LANLH(LANLL):
    """
    A heterogeneous graph representation of the UWF22 dataset.
    """
    def __init__(self, 
                 data_dir, 
                 bin_size: int = 20, 
                 batch_size: int = 350, 
                 from_time: int = 0, 
                 to_time: int = 70000, 
                 lanl_URL: str | None = None, 
                 transforms: List = [], 
                 dataset_name: str = "LANL"):
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, lanl_URL, transforms, dataset_name)

    def df_to_data(self, df: pd.DataFrame): # type: ignore
        data = HeteroData()
        data.time = torch.from_numpy(df["time"].to_numpy(dtype=float)).to(dtype=torch.float64)
        data.x = torch.zeros((len(self.hostmap), 0))
        data.edge_index = torch.from_numpy(df[["source computer", "destination computer"]].to_numpy().T).to(torch.int64)
        data.edge_attr = torch.nan_to_num(torch.from_numpy(df[[
            "authentication type",
            "logon type",
            "authentication orientation",
            "success/failure"
        ]].astype(float).to_numpy()).to(torch.float32))
        data.y = torch.from_numpy(df["malicious"].to_numpy()).to(torch.int64)

        return data
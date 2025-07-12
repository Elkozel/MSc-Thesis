from multiprocessing.pool import ThreadPool
import os
import inspect
import math
import logging
from typing import Any, Generator, Literal, Optional, Union
import requests
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from ordered_set import OrderedSet
from torch.utils.data import random_split, TensorDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch
import lightning as L
from transforms.EnrichHost import EnrichHost
from datasets.UWF22_local import UWF22L


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UWF24L(UWF22L):

    download_data = [
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet/2024-02-25%20-%202024-03-03/part-00000-8b838a85-76eb-4896-a0b6-2fc425e828c2-c000.snappy.parquet",
            "raw_file": "0.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet/2024-03-03%20-%202024-03-10/part-00000-0955ed97-8460-41bd-872a-7375a7f0207e-c000.snappy.parquet",
            "raw_file": "1.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet/2024-03-10%20-%202024-03-17/part-00000-071774ae-97f3-4f31-9700-8bfcdf41305a-c000.snappy.parquet",
            "raw_file": "2.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet/2024-03-17%20-%202024-03-24/part-00000-5f556208-a1fc-40a1-9cc2-a4e24c76aeb3-c000.snappy.parquet",
            "raw_file": "3.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet/2024-03-24%20-%202024-03-31/part-00000-ea3a47a3-0973-4d6b-a3a2-8dd441ee7901-c000.snappy.parquet",
            "raw_file": "4.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet/2024-10-27%20-%202024-11-03/part-00000-69700ccb-c1c1-4763-beb7-cd0f1a61c268-c000.snappy.parquet",
            "raw_file": "5.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet/2024-11-03%20-%202024-11-10/part-00000-f078acc1-ab56-40a6-a6e1-99d780645c57-c000.snappy.parquet",
            "raw_file": "6.parquet"
        }
    ]

    def __init__(self, 
                 data_dir: str,
                 bin_size: int = 20,
                 batch_size: int = 350,
                 from_time: int = 0,
                 to_time: int = 21758350, # (Relative) timestamp of last event is 21758350.529811144
                 transforms: list = [],
                 batch_split: list = [0.6, 0.25, 0.15],
                 dataset_name: str = "UWF24"):
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, batch_split, dataset_name)

        self.ts_first_event = 1709092837.805641 # This allows us to easily make time relative

        # Lastly, save those hyperparams
        self.save_hyperparameters()
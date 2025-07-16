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

class UWF24FallL(UWF22L):

    download_data = [
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-09-15%20-%202024-09-22/part-00000-71f94c04-0853-433f-ad56-45d80000fa4d-c000.snappy.parquet",
            "raw_file": "0.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-09-22%20-%202024-09-29/part-00000-47ad3994-9e36-4638-90fa-9651c6c60ad3-c000.snappy.parquet",
            "raw_file": "1.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-10-27%20-%202024-11-03/part-00000-e0469268-a9e0-458d-beec-4e95db2677fc-c000.snappy.parquet",
            "raw_file": "2.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-11-03%20-%202024-11-10/part-00000-a67cd4d5-0aa5-4378-87f4-a61f467d855d-c000.snappy.parquet",
            "raw_file": "3.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-11-10%20-%202024-11-17/part-00000-7f01bb0b-25b4-4353-a7c2-7418346e3cbb-c000.snappy.parquet",
            "raw_file": "4.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-11-17%20-%202024-11-24/part-00000-5bcfc093-4a1f-4fc2-af09-14f133aa5b13-c000.snappy.parquet",
            "raw_file": "5.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-11-24%20-%202024-12-01/part-00000-24521abf-f2eb-4cb7-b243-257846310bb6-c000.snappy.parquet",
            "raw_file": "6.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-12-01%20-%202024-12-08/part-00000-3c25b9c9-b28c-4a19-b79a-3fadb292d329-c000.snappy.parquet",
            "raw_file": "7.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall24-2/parquet/2024-12-08%20-%202024-12-15/part-00000-fdff949b-33b2-4ebc-b926-a59d462d7ea4-c000.snappy.parquet",
            "raw_file": "8.parquet"
        }
    ]

    def __init__(self, 
                 data_dir: str,
                 bin_size: int = 20,
                 batch_size: int = 350,
                 from_time: int = 0,
                 to_time: int = 6835980, # (Relative) timestamp of last event is 6835979.18627286
                 transforms: list = [],
                 batch_split: list = [0.6, 0.25, 0.15],
                 dataset_name: str = "UWF24Fall"):
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, batch_split, dataset_name)

        self.ts_first_event = 1726952812.207993 # This allows us to easily make time relative

        # Lastly, save those hyperparams
        self.save_hyperparameters()
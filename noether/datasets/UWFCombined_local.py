from multiprocessing.pool import ThreadPool
import os
import inspect
import math
import logging
from typing import Any, Generator, List, Literal, Optional, Union
import requests
import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from ordered_set import OrderedSet
from torch.utils.data import random_split, TensorDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch
import lightning as L
from transforms.EnrichHost import EnrichHost

from noether.datasets.UWF import UWF22L
from datasets.UWF22Fall_local import UWF22FallL
from datasets.UWF24_local import UWF24L
from datasets.UWF24Fall_local import UWF24FallL
from itertools import chain


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UWFCombined(L.LightningDataModule):

    all_datasets: List[L.LightningDataModule] = []
    def __init__(self, 
                 data_dir: str,
                 bin_size: int = 20,
                 batch_size: int = 350,
                 transforms: list = [],
                 batch_split: list = [0.6, 0.25, 0.15],
                 dataset_name: str = "UWFCombined"):
        super().__init__()
        self.all_datasets.append(UWF22L(data_dir, 
                                        bin_size, 
                                        batch_size, 
                                        transforms=transforms, 
                                        batch_split=batch_split, 
                                        dataset_name=dataset_name))
        self.all_datasets.append(UWF22FallL(data_dir, 
                                        bin_size, 
                                        batch_size, 
                                        transforms=transforms, 
                                        batch_split=batch_split, 
                                        dataset_name=dataset_name))
        self.all_datasets.append(UWF24L(data_dir, 
                                        bin_size, 
                                        batch_size, 
                                        transforms=transforms, 
                                        batch_split=batch_split, 
                                        dataset_name=dataset_name))
        self.all_datasets.append(UWF24FallL(data_dir, 
                                        bin_size, 
                                        batch_size, 
                                        transforms=transforms, 
                                        batch_split=batch_split, 
                                        dataset_name=dataset_name))

        self.save_hyperparameters()
        self.node_features = 6
        self.edge_features = 14
        self.num_classes = 11

    def prepare_data(self):
        for dataset in self.all_datasets:
            dataset.prepare_data()

    def setup(self, stage: str):
        for dataset in self.all_datasets:
            dataset.setup(stage)

    def train_dataloader(self):
        dataset = CustomBatchDataset(self, stage="train")
        return dataset
    
    def val_dataloader(self):
        dataset = CustomBatchDataset(self, stage="val")
        return dataset
    
    def test_dataloader(self):
        dataset = CustomBatchDataset(self, stage="test")
        return dataset

class CustomBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_module, stage: Literal['train', 'val', 'test'], estimated_len: int = 0):
        self.data_module = data_module
        self.stage = 0 if stage == "train" else 1 if stage == "val" else 2
        self._length = self.calulate_length() if estimated_len == 0 else estimated_len

    def calulate_length(self):
        all_batch_masks = [self.data_module.batch_mask[file["raw_file"]] for file in self.data_module.download_data]
        sum_batch_masks = [torch.sum(mask == self.stage) for mask in all_batch_masks]
        return sum(sum_batch_masks)
    
    def __iter__(self):
        return self.data_module.batch_generator(self.stage)

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError("Length is not defined for streaming dataset.")
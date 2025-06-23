# https://csr.lanl.gov/data-fence/1750630801/cCYJvf9djgkHXw9h_OxGBC8wYyQ=/cyber1/auth.txt.gz

from multiprocessing.pool import ThreadPool
import os
import math
import numpy as np
import logging
from typing import Any, Hashable, List, Literal, Optional, Union
import requests
import torch
from tqdm import tqdm
import pandas as pd
from ordered_set import OrderedSet
from utils.SimpleBatch import SimpleBatch
from transforms.EnrichHost import EnrichHost
from torch.utils.data import random_split, TensorDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch
import lightning as L


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LANLL(L.LightningDataModule):

    def __init__(self, 
                 data_dir,
                 bin_size: int = 20,
                 batch_size: int = 350,
                 download: bool = False, 
                 lanl_URL: Optional[str] = None,
                 file_data: str = "filedata.json",
                 transforms: list = []):
        super().__init__()
        self.data_dir = data_dir
        self.bin_size = bin_size
        self.batch_size = batch_size
        self.download = download
        self.file_data = os.path.join(data_dir, file_data)
        self.transforms = transforms
        self.edge_features = 4
        self.node_features = 1
        self.num_classes = 2
        if self.download:
            assert lanl_URL is not None, "Download URL should be set if download is True"
            self.auth_file = {
                "url": os.path.join(lanl_URL, "auth.txt.gz"),
                "file": os.path.join(data_dir, "auth.txt.gz")
            }
            self.redteam_file = {
                "url": os.path.join(lanl_URL, "redteam.txt.gz"),
                "file": os.path.join(data_dir, "redteam.txt.gz")
            }
            self.download_all_files()
    
    def check_files(self):
        return os.path.exists(self.file_data)
    
    def download_file(self, url, filepath, tqdm_pos=0):
        # Send a GET request to the URL with streaming enabled so we can read it in chunks
        response = requests.get(url, stream=True)

        # Get the total file size in bytes from the response headers
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # Read in blocks of 1 KB

        # Initialize a progress bar using tqdm
        with tqdm(
            desc=f"Downloading {os.path.basename(filepath)}",  # Display filename in progress bar
            total=total_size,
            unit="B",  # Show progress in bytes
            unit_scale=True,  # Automatically scale units (e.g., KB, MB)
            position=tqdm_pos  # Position for tqdm when multiple bars are displayed
        ) as progress_bar:
            # Open the destination file in binary write mode
            with open(filepath, "wb") as file:
                # Write the file in chunks
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))  # Update the progress bar
                    file.write(data)  # Write chunk to file

        # If the downloaded size doesn't match the expected size, raise an error
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")

    def download_all_files(self):
        download_links = [self.auth_file, self.redteam_file]
        # Check if all files listed in self.download_data already exist in data_dir
        if all([os.path.exists(os.path.join(self.data_dir, link["file"])) for link in download_links]):
            return  # If so, skip downloading

        # Create a thread pool with 8 worker threads
        with ThreadPool(2) as pool:
            # Prepare arguments: (url, destination path, tqdm bar position) for each file
            args = [(
                file_data["url"], 
                os.path.join(self.data_dir, file_data["file"]),
                idx  # tqdm progress bar position (starts at 1)
            ) for idx, file_data in enumerate(download_links, 1)]

            # Use map to apply download_file to all argument tuples in parallel
            pool.map(lambda x: self.download_file(*x), args)

    def csv_auth_to_pickle(self, 
                      file: str,
                      file_data={}, 
                      sec_per_file: int = 3600):
        leftover = pd.DataFrame([])
        columns = [
            'time',
            'source user@domain',
            'destination user@domain',
            'source computer',
            'destination computer',
            'authentication type',
            'logon type',
            'authentication orientation',
            'success/failure'
        ]
        # Maps for each column are also created automatically
        keyword_map = {col: OrderedSet([]) for col in columns if col != 'time'}

        # Read the CSV file in chunks
        for part in tqdm(
            pd.read_csv(file, sep=',', compression='gzip', chunksize=1000000, iterator=True, names=columns, header=None),
            desc="Splitting CSV"):
            # Merge leftover from previous chunk if any
            if not leftover.empty:
                part = pd.concat([leftover, part], ignore_index=True)

            # Ensure 'time' is numeric
            part['time'] = pd.to_numeric(part['time'], errors='coerce')
            part.dropna(subset=['time'], inplace=True)
            part.sort_values(by='time', inplace=True)

            # Update keyword map
            for col in keyword_map:
                keyword_map[col].update(part[col].dropna().unique()) # type: ignore

            # Apply the keyword map
            for col in keyword_map:
                part[col] = part[col].map(keyword_map[col].index)

            # Bin the data
            part['bin'] = (part['time'] // sec_per_file) * sec_per_file
            grouped = part.groupby('bin')
            leftover = pd.DataFrame([])

            for bin_start, group in grouped:
                # If this is the last group, it may be incomplete — keep as leftover
                if bin_start + sec_per_file > part[0].max(): # type: ignore
                    leftover = group
                    continue

                # Save the bin to a pickle file
                fname = f"{bin_start}.pkl"
                group.drop(columns='bin').to_pickle(fname)

                # Record in file_data
                file_data[bin_start] = {
                    'file': fname,
                    'min_ts': float(group["time"].min()),
                    'max_ts': float(group["time"].max()),
                    'rows': len(group)
                }

        # Optionally process leftover after loop
        if not leftover.empty:
            last_bin_start = (leftover[0].min() // sec_per_file) * sec_per_file
            fname = f"{last_bin_start}_leftover.pkl"
            leftover.drop(columns='bin').to_pickle(fname)
            file_data[last_bin_start] = {
                'file': fname,
                'min_ts': float(leftover["time"].min()),
                'max_ts': float(leftover["time"].max()),
                'rows': len(leftover)
            }

        return file_data, keyword_map
            
    def prepare_data(self):
        if self.check_files():
            return
        
        # Download files
        self.download_all_files()
        
        # Create pickle files
        file_data, maps = self.csv_auth_to_pickle(self.auth_file["file"])

        # save the file data and the maps
        pd.DataFrame(file_data).to_json(self.file_data)
        for col, map in maps:
            pd.DataFrame(map).to_pickle(os.path.join(self.data_dir, f"{col}.pkl"))


    def generate_split(self):
        # Reset the file mask
        self.file_masks = {}

        # Each file has the following data:
        # {
        #     'file': fname,
        #     'min_ts': float(leftover["time"].min()),
        #     'max_ts': float(leftover["time"].max()),
        #     'rows': len(leftover)
        # }
        file_data = pd.read_json(self.file_data)
        file_data["num_batches"] = (file_data["max_ts"] - file_data["min_ts"]) // self.bin_size

        # Randomly split the indices into training (60%), validation (25%), and test (15%) sets
        generator = torch.Generator().manual_seed(42)
        num_batches = file_data["num_batches"]

        for _, row in file_data.iterrows():
            fname = row['file']
            num_batches = int(row['num_batches'])

            if num_batches == 0:
                continue  # skip empty files

            # Create full batch index tensor for this file
            full_idx = torch.arange(num_batches)
            idx = TensorDataset(full_idx)

            _, val, test = random_split(idx, [0.6, 0.25, 0.15], generator=generator)

            # Create mask: 0 = train, 1 = val, 2 = test
            batch_mask = torch.zeros(num_batches, dtype=torch.uint8)
            batch_mask[val.indices] = 1
            batch_mask[test.indices] = 2

            self.file_masks[fname] = batch_mask
            
        return self.file_masks
        
    def setup(self, stage: str):
        # Generate the batch masks
        self.generate_split()

    def transform_data(self, data: Union[Data, HeteroData]) -> BaseData:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def merge_with_redteam(self, df: pd.DataFrame):
        redteam_header = [
            "time",
            "user@domain",
            "source computer",
            "destination computer"
        ]
        redteam = pd.read_csv(
            self.redteam_file["file"], 
            sep=',', 
            compression='gzip',
            names=redteam_header, 
            header=None)
        
        # First, do a merge on source_user
        merge_source = pd.merge(
            df,
            redteam,
            left_on=['time', 'source_user', 'source_computer', 'destination_computer'],
            right_on=['time', 'user', 'source_computer', 'destination_computer'],
            how='left',
            indicator='source_match'
        )

        # Then, do a merge on destination_user
        merge_dest = pd.merge(
            df,
            redteam,
            left_on=['time', 'destination_user', 'source_computer', 'destination_computer'],
            right_on=['time', 'user', 'source_computer', 'destination_computer'],
            how='left',
            indicator='dest_match'
        )

        # Add flags
        merge_source['source_malicious'] = merge_source['source_match'] == 'both'
        merge_dest['destination_malicious'] = merge_dest['dest_match'] == 'both'

        df['malicious'] = merge_source['source_malicious'] | merge_dest['destination_malicious']

        return df

    def df_to_data(self, df: pd.DataFrame, bin_ranges: List[Any]):
        df["bin"] = (df['time'] // self.bin_size) * self.bin_size
        hostmap: pd.DataFrame = pd.read_pickle(os.path.join(self.data_dir, "source computer.pkl"))
        df = self.merge_with_redteam(df)

        data = Data()
        data.time = torch.from_numpy(df["time"].to_numpy(dtype=float)).to(dtype=torch.float64)
        data.x = torch.from_numpy(hostmap.to_numpy()).to(torch.float32)
        data.edge_index = torch.from_numpy(df[["source computer", "destination computer"]].to_numpy().T).to(torch.int64)
        data.edge_attr = torch.nan_to_num(torch.from_numpy(df[[
            "authentication type",
            "logon type",
            "authentication orientation",
            "success/failure"
        ]].astype(float).to_numpy()).to(torch.float32))
        data.y = torch.from_numpy(df["malicious"].to_numpy()).to(torch.int64)

        # Generate bins
        for range in bin_ranges:
            if range["empty"]:
                yield Data(
                    time=torch.empty(size=[0]).to(torch.int64),
                    edge_index=torch.empty(size=(2, 0)).to(torch.int64),
                    edge_attr=torch.empty(size=(0, self.edge_features)).to(torch.float32),
                    y=torch.empty(size=[0]).to(torch.int64),
                    x=data.x
                )
            else:
                start_idx = int(range["start"])
                end_idx = int(range["end"])
                yield Data(
                    time=data.time[start_idx:end_idx],
                    edge_index=data.edge_index[:, start_idx:end_idx],
                    edge_attr=data.edge_attr[start_idx:end_idx],
                    y=data.y[start_idx:end_idx], # type: ignore
                    x=data.x
                )

    def batch_generator(self, stage: Literal['train', 'val', 'test']):
        file_data = pd.read_json(self.file_data)
        for _, row in file_data.iterrows():
            pkl_file = row['file']
            from_ts = row["min_ts"]
            to_ts = row["max_ts"]
            batch_mask = self.file_masks[pkl_file]

            df = pd.read_pickle(os.path.join(self.data_dir, pkl_file))
            data = list(self.df_to_data(df, list(range(from_ts, to_ts, self.bin_size))))
            data_transformed: List[BaseData] = list(map(self.transform_data, data))
            for batch_num, batch_i in enumerate(range(0, len(data_transformed), self.batch_size)):
                batch = Batch.from_data_list(data_transformed[batch_i:batch_i+self.batch_size])
                stage_num = 0
                if stage == "val":
                    stage_num = 1
                elif stage == "test":
                    stage_num = 2
                if batch_mask[batch_num] != stage_num:
                    continue

                # if batch.num_graphs <= self.rnn_window:
                #     continue
                yield batch

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
        all_masks = list(self.data_module.file_masks.values())
        lengts = [torch.sum(mask == self.stage) for mask in all_masks]
        return sum(lengts)
    
    def __iter__(self):
        return self.data_module.batch_generator(self.stage)

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError("Length is not defined for streaming dataset.")
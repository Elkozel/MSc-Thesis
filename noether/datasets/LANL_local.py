# https://csr.lanl.gov/data-fence/1750630801/cCYJvf9djgkHXw9h_OxGBC8wYyQ=/cyber1/auth.txt.gz

from multiprocessing.pool import ThreadPool
import os
import math
import logging
from typing import Any, Generator, Literal, Optional, Union
import requests
import torch
from tqdm import tqdm
import pandas as pd
from ordered_set import OrderedSet
from torch.utils.data import random_split, TensorDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch
import lightning as L


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LANLL(L.LightningDataModule):

    def __init__(self, 
                 data_dir: str,
                 bin_size: int = 20,
                 batch_size: int = 350,
                 from_time: int = 0,
                 to_time: int = 1209600, # First 14 days
                 lanl_URL: Optional[str] = None,
                 transforms: list = [],
                 dataset_name: str = "LANL"):
        super().__init__()
        self.data_dir = data_dir
        self.bin_size = bin_size
        self.from_time = from_time
        self.to_time = to_time
        self.batch_size = batch_size
        self.transforms = transforms
        self.dataset_name = dataset_name
        self.edge_features = 4
        self.node_features = 0
        self.num_classes = 2

        self.save_hyperparameters()

        self.auth_file = {}
        self.auth_file["file"] = os.path.join(data_dir, "auth.txt.gz")
        self.redteam_file = {}
        self.redteam_file["file"] = os.path.join(data_dir, "redteam.txt.gz")
        
        if lanl_URL is not None:
            self.auth_file["url"] =  os.path.join(lanl_URL, "auth.txt.gz")
            self.redteam_file["url"] = os.path.join(lanl_URL, "redteam.txt.gz")

        os.makedirs(self.data_dir, exist_ok=True)
    
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

        assert self.auth_file is not None, "Dataset files could not be found, \
            please specify a download URL to download them (see lanl_URL argument)"
        assert self.redteam_file is not None, "Dataset files could not be found, \
            please specify a download URL to download them (see lanl_URL argument)"
        # Create a thread pool with 2 worker threads
        with ThreadPool(2) as pool:
            # Prepare arguments: (url, destination path, tqdm bar position) for each file
            args = [(
                file_data["url"], 
                file_data["file"],
                idx  # tqdm progress bar position (starts at 1)
            ) for idx, file_data in enumerate(download_links, 1)]

            # Use map to apply download_file to all argument tuples in parallel
            pool.map(lambda x: self.download_file(*x), args)

    def prepare_data(self):
        self.download_all_files()

    def generate_split(self):
        # Randomly split the indices into training (60%), validation (25%), and test (15%) sets
        generator = torch.Generator().manual_seed(42)
        from_bin = (self.from_time // self.bin_size) * self.bin_size
        to_bin = (self.to_time // self.bin_size) * self.bin_size
        bin_range = range(from_bin, to_bin, self.bin_size)
        num_batches = math.ceil(len(bin_range) / self.batch_size)

        # Create full batch index tensor for this file
        full_idx = torch.arange(num_batches)
        idx = TensorDataset(full_idx)

        _, val, test = random_split(idx, [0.6, 0.25, 0.15], generator=generator)

        # Create mask: 0 = train, 1 = val, 2 = test
        self.batch_mask = torch.zeros(num_batches, dtype=torch.uint8)
        self.batch_mask[val.indices] = 1
        self.batch_mask[test.indices] = 2

        return self.batch_mask
        
    def setup(self, stage: str):
        # Generate the batch masks
        self.generate_split()

        # Load redteam
        redteam_header = [
            "time",
            "user@domain",
            "source computer",
            "destination computer"
        ]
        self.redteam = pd.read_csv(
            self.redteam_file["file"], 
            sep=',', 
            compression='gzip',
            names=redteam_header, 
            header=None).drop_duplicates()

    def transform_data(self, data: Union[Data, HeteroData]) -> BaseData:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def generate_bins(self) -> Generator[pd.DataFrame, Any, None]:
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
        self.keyword_map = {col: OrderedSet([]) for col in columns if col != 'time'}
        self.keyword_map.pop("source computer", None)
        self.keyword_map.pop("destination computer", None)
        self.hostmap = OrderedSet([])

        self.keyword_map.pop("source user@domain", None)
        self.keyword_map.pop("destination user@domain", None)
        self.usermap = OrderedSet([])

        # Read the CSV file in chunks
        for part in pd.read_csv(
            self.auth_file["file"], 
            sep=',', 
            compression='gzip',
            chunksize=100000, 
            iterator=True, 
            names=columns, 
            header=None):

            # Ensure 'time' is numeric
            part['time'] = pd.to_numeric(part['time'], errors='coerce')
            part.dropna(subset=['time'], inplace=True)
            part.sort_values(by='time', inplace=True)

            # Skip until the start time has been reached
            if part["time"].max() < self.from_time:
                continue

            # Find only data between the time given
            part = part[(part["time"] >= self.from_time) & (part["time"] < self.to_time)]
            if part.empty:
                leftover = pd.DataFrame([])
                break
            

            # Merge the data with redteam (attaching labels)
            part = self.merge_with_redteam(part)

            # Update keyword map
            for col in self.keyword_map:
                self.keyword_map[col].update(part[col].dropna().unique()) # type: ignore
            self.hostmap.update(part["source computer"].dropna().unique()) # type: ignore
            self.hostmap.update(part["destination computer"].dropna().unique()) # type: ignore
            self.usermap.update(part["source user@domain"].dropna().unique()) # type: ignore
            self.usermap.update(part["destination user@domain"].dropna().unique()) # type: ignore

            # Apply the keyword map
            for col in self.keyword_map:
                part[col] = part[col].map(self.keyword_map[col].index)
            part["source computer"] = part["source computer"].map(self.hostmap.index)
            part["destination computer"] = part["destination computer"].map(self.hostmap.index)
            part["source user@domain"] = part["source user@domain"].map(self.usermap.index)
            part["destination user@domain"] = part["destination user@domain"].map(self.usermap.index)

            # Merge leftover from previous chunk if any
            if not leftover.empty:
                part = pd.concat([leftover, part], ignore_index=True)

            # Bin the data
            part['bin'] = (part['time'] // self.bin_size) * self.bin_size
            grouped = list(part.groupby('bin'))

            # Assume that the last bin is incomplete
            # if it is complete, it will be sent with the next batch
            _, leftover = grouped.pop()

            for _, group in grouped:
                yield group

        # Optionally process leftover after loop
        if not leftover.empty:
            grouped = leftover.groupby('bin')

            for _, group in grouped:
                yield group

    def generate_batches(self, batch_type: int | None = None):
        batch_num = 0
        batch: list[pd.DataFrame] = []
        for bin in self.generate_bins():
            # Collect bins
            if len(batch) != self.batch_size:
                batch.append(bin)
                continue

            # Create the batch and send it
            if batch_type is None:
                yield batch, int(self.batch_mask[batch_num])
            elif self.batch_mask[batch_num] == batch_type:
                yield batch, int(self.batch_mask[batch_num])
            batch = []
            batch_num = batch_num + 1
        
        # Handle leftover bins
        if len(batch) > 0:
            yield batch, int(self.batch_mask[batch_num])
    
    def merge_with_redteam(self, df: pd.DataFrame):
        
        # First, do a merge on source_user
        merge_source = pd.merge(
            df,
            self.redteam,
            left_on=['time', 'source user@domain', 'source computer', 'destination computer'],
            right_on=['time', 'user@domain', 'source computer', 'destination computer'],
            how='left',
            validate='many_to_one',
            indicator='source_match'
        ).reset_index(drop=True)

        # Then, do a merge on destination_user
        merge_dest = pd.merge(
            df,
            self.redteam,
            left_on=['time', 'destination user@domain', 'source computer', 'destination computer'],
            right_on=['time', 'user@domain', 'source computer', 'destination computer'],
            how='left',
            validate='many_to_one',
            indicator='dest_match'
        ).reset_index(drop=True)

        # Add flags
        merge_source['source_malicious'] = merge_source['source_match'] == 'both'
        merge_dest['destination_malicious'] = merge_dest['dest_match'] == 'both'
        df = df.reset_index(drop=True)
        
        df["malicious"] = False
        df["malicious"] = merge_source['source_malicious'] | merge_dest['destination_malicious']

        return df

    def df_to_data(self, df: pd.DataFrame):
        data = Data()
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

    def batch_generator(self, stage: Literal['train', 'val', 'test']):
        stage_id = 0
        if stage == "val":
            stage_id = 1
        elif stage == "test":
            stage_id = 2

        for batch, _ in self.generate_batches(stage_id):
            batch = [self.df_to_data(bin) for bin in batch]
            transformed_batch = [self.transform_data(bin) for bin in batch]
            yield Batch.from_data_list(transformed_batch)

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
        lengts = torch.sum(self.data_module.batch_mask == self.stage)
        return int(lengts)
    
    def __iter__(self):
        return self.data_module.batch_generator(self.stage)

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError("Length is not defined for streaming dataset.")
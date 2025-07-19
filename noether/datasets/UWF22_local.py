from multiprocessing.pool import ThreadPool
import os
import inspect
import math
import logging
from typing import Any, Callable, Generator, List, Literal, Optional, Union
import requests
import torch
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from ordered_set import OrderedSet
from torch.utils.data import DataLoader
from torch.utils.data import random_split, TensorDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch
import lightning as L
from transforms.EnrichHost import EnrichHost


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UWF22L(L.LightningDataModule):

    download_data = [
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-12%20-%202021-12-19/part-00000-7c2e9adb-5430-4792-a42b-10ff5bbd46e8-c000.snappy.parquet",
            "raw_file": "0.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-19%20-%202021-12-26/part-00000-3f86626a-1225-47f9-a5a2-0170b737e404-c000.snappy.parquet",
            "raw_file": "1.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-26%20-%202022-01-02/part-00000-b1a9fc13-8068-4a5d-91b2-871438709e81-c000.snappy.parquet",
            "raw_file": "2.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-02%20-%202022-01-09/part-00000-26e9208e-7819-451b-b23f-2e47f6d1e834-c000.snappy.parquet",
            "raw_file": "3.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-09%20-%202022-01-16/part-00000-36240b61-b84f-4164-a873-d7973e652780-c000.snappy.parquet",
            "raw_file": "4.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-16%20-%202022-01-23/part-00000-cbf26680-106d-40e7-8278-60520afdbb0e-c000.snappy.parquet",
            "raw_file": "5.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-02-06%20-%202022-02-13/part-00000-df678a79-4a73-452b-8e72-d624b2732f17-c000.snappy.parquet",
            "raw_file": "6.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-02-13%20-%202022-02-20/part-00000-1da06990-329c-4e38-913a-0f0aa39b388d-c000.snappy.parquet",
            "raw_file": "7.parquet"
        }
    ]

    def __init__(self, 
                 data_dir: str,
                 bin_size: int = 20,
                 batch_size: int = 350,
                 from_time: int = 0,
                 to_time: int = 16452990000, # Max timestamp is 1645298196.163731
                 transforms: list = [],
                 batch_split: list = [0.6, 0.25, 0.15],
                 dataset_name: str = "UWF22",
                 account_for_duration: bool = True):
        super().__init__()
        self.data_dir = os.path.join(data_dir, dataset_name)
        self.from_time = from_time
        self.to_time = to_time
        self.bin_size = bin_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.account_for_duration = account_for_duration
        self.batch_mask = {}
        self.batch_split = batch_split
        self.ts_first_event = 1639746045.213779 # This allows us to easily make time relative

        self.save_hyperparameters()
        self.node_features = 6
        self.edge_features = 14
        self.num_classes = 11

    
    def download_file(self, url, filepath, tqdm_pos=0):
        # Check if the file is already downloaded
        if os.path.exists(filepath):
            return  # If so, skip downloading
        
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
        # Create a thread pool with 8 worker threads
        with ThreadPool(8) as pool:
            # Prepare arguments: (url, destination path, tqdm bar position) for each file
            args = [(
                file_data["url"], 
                os.path.join(self.data_dir, file_data["raw_file"]),
                idx  # tqdm progress bar position (starts at 1)
            ) for idx, file_data in enumerate(self.download_data, 1)]

            # Use map to apply download_file to all argument tuples in parallel
            pool.map(lambda x: self.download_file(*x), args)

    def prepare_data(self):
        # Make sure the data dir is there
        os.makedirs(self.data_dir, exist_ok=True)
        # Download files if nesessary
        self.download_all_files()

    def generate_split_file(self, filename: str):
        df = pd.read_parquet(os.path.join(self.data_dir, filename), columns=['ts'])

        # Make time relative
        df["ts"] = df["ts"] - self.ts_first_event
        
        # Filter only the timestamps we care about
        df = df[(df["ts"] >= self.from_time) & (df["ts"] < self.to_time)]
        if df.empty:
            self.batch_mask[filename] = torch.empty(0)
            return

        # Calculate bins
        from_time = math.floor(df["ts"].min())
        to_time = math.floor(df["ts"].max())

        # Randomly split the indices into training (60%), validation (25%), and test (15%) sets
        generator = torch.Generator().manual_seed(42)
        from_bin = (from_time // self.bin_size) * self.bin_size
        to_bin = (to_time // self.bin_size) * self.bin_size
        bin_range = range(from_bin, to_bin, self.bin_size)
        num_batches = math.floor(len(bin_range) / self.batch_size) + 1

        # Create full batch index tensor for this file
        full_idx = torch.arange(num_batches)
        idx = TensorDataset(full_idx)

        _, val, test = random_split(idx, self.batch_split, generator=generator)

        # Create mask: 0 = train, 1 = val, 2 = test
        self.batch_mask[filename] = torch.zeros(num_batches, dtype=torch.uint8)
        self.batch_mask[filename][val.indices] = 1
        self.batch_mask[filename][test.indices] = 2

        return self.batch_mask[filename]
        
    def setup(self, stage: str):
        for file in self.download_data:
            filename = file["raw_file"]
            # Generate the batch masks
            self.generate_split_file(filename)

    def transform_data(self, data: Union[Data, HeteroData]) -> BaseData:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def expand_duration(self, df: pd.DataFrame):
        if not self.account_for_duration:
            df["conn_status"] = "closing"
            return df
        
        def expand(x):
            duration = x["duration"]
            end_ts = x["ts"]
            start_ts = float(end_ts - duration)
            
            # Otherwise, determine how many adjustments will need to be made
            all_ts = np.arange(start_ts, end_ts, self.bin_size)[:-1] # cut the last one (we already have the original event)
            all_ts = np.append(all_ts, end_ts) # add the original event

            open_conn_states = ["open" for _ in range(all_ts.size - 2)]
            all_conn_states = ["started"] + open_conn_states + ["closing"]

            return pd.Series([all_ts, all_conn_states], index=['ts', 'conn_status'])
        
        df["conn_status"] = "closing"
        df["ts"] = df["ts"].astype("object")
        mask = (df["duration"] > self.bin_size)
        df.loc[mask, ["ts", "conn_status"]] = df[mask].apply(expand, axis=1)
        df = df.explode(["ts", "conn_status"])

        # Recast to a float
        df["ts"] = df["ts"].astype("float")

        # Remove events with negative ts
        df = df[df["ts"] >= 0.0]

        # Sort by time
        df = df.sort_values(by=['ts'])  # Sort records chronologically
        df = df.reset_index(drop=True)  # Reset index after sorting

        return df
    
    def generate_bins_from_file(self, filename, keyword_map = None) -> Generator[pd.DataFrame, Any, None]:
        df = pd.read_parquet(filename)
        df = self.expand_duration(df)
        columns = df.columns

        # Maps for each column are also created automatically
        if keyword_map is None:
            self.keyword_map = {col: OrderedSet([]) for col in columns if col != 'ts'}
            self.keyword_map.pop("src_ip_zeek", None)
            self.keyword_map.pop("dest_ip_zeek", None)
            self.hostmap = OrderedSet([])

            self.keyword_map.pop("src_port_zeek", None)
            self.keyword_map.pop("dest_port_zeek", None)
            self.servicemap = OrderedSet([])

        # Make time relative
        df["abs_ts"] = df["ts"]
        df["ts"] = df["ts"] - self.ts_first_event

        # Ensure 'time' is numeric
        df['ts'] = pd.to_numeric(df['ts'], errors='coerce')
        df.dropna(subset=['ts'], inplace=True)
        df.sort_values(by='ts', inplace=True)

        # Find only data between the time given
        df = df[(df["ts"] >= self.from_time) & (df["ts"] < self.to_time)]
        if df.empty:
            return

        # Update keyword map
        for col in self.keyword_map:
            self.keyword_map[col].update(df[col].fillna("None").unique()) # type: ignore
        self.hostmap.update(df["src_ip_zeek"].fillna("None").unique()) # type: ignore
        self.hostmap.update(df["dest_ip_zeek"].fillna("None").unique()) # type: ignore
        self.servicemap.update(df[["src_ip_zeek", "src_port_zeek"]]
                               .drop_duplicates()
                               .reset_index(drop=True)
                               .apply(lambda x: (x.iloc[0], x.iloc[1]), axis=1)) # type: ignore
        self.servicemap.update(df[["dest_ip_zeek", "dest_port_zeek"]]
                               .drop_duplicates()
                               .reset_index(drop=True)
                               .apply(lambda x: (x.iloc[0], x.iloc[1]), axis=1)) # type: ignore

        # Apply the keyword map
        for col in self.keyword_map:
            df[col] = df[col].fillna("None").map(self.keyword_map[col].index)
        df["src_ip_id"] = df["src_ip_zeek"].map(self.hostmap.index)
        df["dest_ip_id"] = df["dest_ip_zeek"].map(self.hostmap.index)
        df["src_service_id"] = df[["src_ip_zeek", "src_port_zeek"]] \
                                .apply(lambda x: (x.iloc[0], x.iloc[1]), axis=1).map(self.servicemap.index)
        df["dest_service_id"] = df[["dest_ip_zeek", "dest_port_zeek"]] \
                                .apply(lambda x: (x.iloc[0], x.iloc[1]), axis=1).map(self.servicemap.index)

        # Bin the data
        df['bin'] = (df['ts'] // self.bin_size) * self.bin_size
        grouped = list(df.groupby('bin'))

        previous_start = float(df["bin"].min())
        for bin_start, group in grouped:
            # Sometimes there are bins without any data, in this case we have to "simulate"
            # an empty bin, as it will not be in the groupby. Thus, we send empty dataframes
            # until we "reach" the target bin
            while bin_start - previous_start > self.bin_size: # type: ignore
                previous_start += self.bin_size # type: ignore
                yield pd.DataFrame([], columns=df.columns, dtype="int64")
            
            previous_start = bin_start
            yield group

    def generate_batches(self, batch_type: int | None = None):
        for file in self.download_data:        
            batch_num = 0
            filename = os.path.join(self.data_dir, file["raw_file"])
            batch: list[pd.DataFrame] = []
            batch_mask = self.batch_mask[file["raw_file"]]

            for bin in self.generate_bins_from_file(filename):
                # Collect bins
                if len(batch) != self.batch_size:
                    batch.append(bin)
                    continue

                # Create the batch and send it
                if batch_type is None:
                    yield batch, int(batch_mask[batch_num])
                elif batch_mask[batch_num] == batch_type:
                    yield batch, int(batch_mask[batch_num])
                batch = []
                batch_num += 1
            
            # Handle leftover bins
            if len(batch) > 0:
                yield batch, int(batch_mask[batch_num])

    def df_to_data(self, df: pd.DataFrame):
        hostmap_df = pd.DataFrame([
            EnrichHost.enrich_host(host, id) for id, host in enumerate(self.hostmap)
        ])

        data = Data()
        data.time = torch.from_numpy(df["ts"].to_numpy(dtype=float)).to(dtype=torch.float64)
        data.x = torch.from_numpy(hostmap_df[[
                "internal",
                "broadcast",
                "multicast",
                "ipv4",
                "ipv6",
                "valid",
        ]].to_numpy()).to(torch.float32)
        data.edge_index = torch.from_numpy(df[["src_ip_id", "dest_ip_id"]].to_numpy().T).to(torch.int64)
        data.edge_attr = torch.nan_to_num(torch.from_numpy(df[[
                "conn_status",
                "src_port_zeek",
                "dest_port_zeek",
                "orig_bytes",
                "orig_pkts",
                "resp_bytes",
                "resp_pkts",
                "missed_bytes",
                "local_orig",
                "local_resp",
                "duration",
                "proto",
                "service",
                "conn_state"
                # TODO: Add info about the port
                # TODO: Add history
        ]].astype(float).to_numpy()).to(torch.float32))
        data.history = df[[
                "history"
                # TODO: Add info about the port
                # TODO: Add history
        ]].to_numpy()
        data.y = torch.from_numpy(df["label_tactic"].to_numpy()).to(torch.int64)

        return data

    def batch_generator(self, stage: Literal['train', 'val', 'test']):
        stage_id = 0
        if stage == "val":
            stage_id = 1
        elif stage == "test":
            stage_id = 2

        for batch, _ in self.generate_batches(stage_id):
            batch = [self.df_to_data(bin) for bin in batch]
            batch = [self.transform_data(bin) for bin in batch]
            yield Batch.from_data_list(batch)

    def train_dataloader(self):
        stage = 0 # Training

        all_batch_masks = torch.cat([self.batch_mask[file["raw_file"]] for file in self.download_data])
        num_batches = int((all_batch_masks == stage).count_nonzero())

        dataset = GeneratorDataset(
            generator_fn=lambda: self.batch_generator("train"),
            length=num_batches
        )

        return DataLoader(dataset,
                          collate_fn=lambda x: x[0])
    
    def val_dataloader(self):
        stage = 1 # Validation

        all_batch_masks = torch.cat([self.batch_mask[file["raw_file"]] for file in self.download_data])
        num_batches = int((all_batch_masks == stage).count_nonzero())

        dataset = GeneratorDataset(
            generator_fn=lambda: self.batch_generator("val"),
            length=num_batches
        )

        return DataLoader(dataset,
                          collate_fn=lambda x: x[0])
    
    def test_dataloader(self):
        stage = 2 # Testing

        all_batch_masks = torch.cat([self.batch_mask[file["raw_file"]] for file in self.download_data])
        num_batches = int((all_batch_masks == stage).count_nonzero())

        dataset = GeneratorDataset(
            generator_fn=lambda: self.batch_generator("test"),
            length=num_batches
        )

        return DataLoader(dataset,
                          collate_fn=lambda x: x[0])
    
class GeneratorDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator_fn: Callable, length: int):
        super().__init__()
        self.generator_fn = generator_fn
        self._length = length

    def __iter__(self):
        return self.generator_fn()

    def __len__(self):
        return self._length
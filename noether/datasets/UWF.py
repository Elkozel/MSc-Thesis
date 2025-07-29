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

class UWF22(L.LightningDataModule):

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
    factorize_cols = ["service", "proto", "conn_state", "conn_status", "label_tactic"]

    def __init__(self, 
                 data_dir: str,
                 bin_size: int = 20,
                 batch_size: int = 350,
                 from_time: int = 0,
                 to_time: int = 16452990000, # Max timestamp is 1645298196.163731
                 transforms: list = [],
                 num_workers: int = 0,
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
        self.num_workers = num_workers
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

    def generate_split_file(self, df: pd.DataFrame):
        df = self.expand_duration(df)

        # Make time relative
        df["ts"] = df["ts"] - self.ts_first_event
        
        # Filter only the timestamps we care about
        df = df[(df["ts"] >= self.from_time) & (df["ts"] < self.to_time)]
        if df.empty:
            return torch.empty(0)

        # Calculate bins
        from_bin = df["ts"].min() // self.bin_size
        to_bin = df["ts"].max() // self.bin_size
        from_batch = from_bin // self.batch_size
        to_batch = to_bin // self.batch_size

        # Randomly split the indices into training (60%), validation (25%), and test (15%) sets
        generator = torch.Generator().manual_seed(42)
        num_batches = int(to_batch - from_batch + 1)

        # Create full batch index tensor for this file
        full_idx = torch.arange(num_batches)
        idx = TensorDataset(full_idx)

        _, val, test = random_split(idx, self.batch_split, generator=generator)

        # Create mask: 0 = train, 1 = val, 2 = test
        batch_mask = torch.zeros(num_batches, dtype=torch.uint8)
        batch_mask[val.indices] = 1
        batch_mask[test.indices] = 2

        return batch_mask
        
    def setup(self, stage: str):
        for file in self.download_data:
            filename = file["raw_file"]
            df = pd.read_parquet(os.path.join(self.data_dir, filename), columns=["ts", "duration"])

            # Generate the batch masks
            batch_mask = self.generate_split_file(df)
            self.batch_mask[filename] = batch_mask

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
    
    def bin_df(self, df: pd.DataFrame, keyword_map = None) -> pd.DataFrame:
        df = self.expand_duration(df)

        # Maps for each column are also created automatically
        if keyword_map is None:
            self.keyword_map = {col: OrderedSet([]) for col in self.factorize_cols}
            self.hostmap = OrderedSet([])
            self.servicemap = OrderedSet([])

        # Ensure that the label mask starts with "none"
        self.keyword_map["label_tactic"].update(["none"])

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
            return pd.DataFrame([], columns=df.columns, dtype="int64")

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
        df['bin'] = df['ts'] // self.bin_size
        # Make bin relative
        df["bin"] = df["bin"] - df["bin"].min()
        
        return df

    def fill_in_gaps_in_batch(self, batch, starting_bin, columns):
        previous_bin = starting_bin
        final_batch = []
        for bin, group in batch:
            while bin - previous_bin > 1:
                previous_bin += 1
                final_batch.append(pd.DataFrame([], columns=columns, dtype="int64"))

            final_batch.append(group)
            previous_bin = bin

        while len(final_batch) < self.batch_size:
            final_batch.append(pd.DataFrame([], columns=columns, dtype="int64"))

        return final_batch
        
    def generate_batches(self):
        for file in self.download_data:
            filename = os.path.join(self.data_dir, file["raw_file"])
            df = pd.read_parquet(filename)
            batch: list[pd.DataFrame] = []
            batch_mask = self.batch_mask[file["raw_file"]]

            df = self.bin_df(df)
            df["batch"] = df["bin"] // self.batch_size
            grouped_batches = df.groupby("batch", observed=False)

            for batch, group in grouped_batches: # type: ignore
                bins = group.groupby("bin")
                corrected_batch = self.fill_in_gaps_in_batch(bins, batch*self.batch_size, df.columns)
                yield corrected_batch, int(batch_mask[int(batch)]) # type: ignore

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

        for batch, batch_stage in self.generate_batches():
            if batch_stage != stage_id:
                continue

            batch = [self.df_to_data(bin) for bin in batch]
            batch = [self.transform_data(bin) for bin in batch]
            yield Batch.from_data_list(batch)

    def train_dataloader(self):
        dataset = GeneratorDataset(
            dataset = self,
            stage = "train"
        )

        return DataLoader(dataset,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: x[0])
    
    def val_dataloader(self):
        dataset = GeneratorDataset(
            dataset = self,
            stage = "val"
        )

        return DataLoader(dataset,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: x[0])
    
    def test_dataloader(self):
        dataset = GeneratorDataset(
            dataset = self,
            stage = "test"
        )

        return DataLoader(dataset,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: x[0])
    
class GeneratorDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: UWF22, stage: Literal['train', 'val', 'test']):
        super().__init__()
        self.dataset = dataset
        self.stage = stage

        if stage == "train":
            self.stage_num = 0
        elif stage == "val":
            self.stage_num = 1
        elif stage == "test":
            self.stage_num = 2
        else:
            raise Exception(f"Stage {stage} not found")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-worker case: return full iterator
            return self.dataset.batch_generator(self.stage) # type: ignore
        else:
            # Multi-worker case: split workload
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            worker_files = [
                file
                for file_id, file in enumerate(self.dataset.download_data)
                if file_id % num_workers == worker_id
            ]

            self.dataset.download_data = worker_files

            return self.dataset.batch_generator(self.stage) # type: ignore
        

    def __len__(self):        
        all_batch_masks = torch.cat([
            self.dataset.batch_mask[file["raw_file"]] 
            for file in self.dataset.download_data])
        num_batches = int((all_batch_masks == self.stage_num).sum())

        return num_batches
    
class UWF22Fall(UWF22):

    download_data = [
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2021-12-12%20-%202021-12-19/part-00000-d512890f-d1e9-49d5-a136-f87f0183cb4d-c000.snappy.parquet",
            "raw_file": "0.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2021-12-19%20-%202021-12-26/part-00000-d28b031b-bff1-4e16-853a-9b7d896627e7-c000.snappy.parquet",
            "raw_file": "1.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2021-12-26%20-%202022-01-02/part-00000-94d13437-ae00-4a8c-9f38-edd0196cfdee-c000.snappy.parquet",
            "raw_file": "2.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-01-02%20-%202022-01-09/part-00000-745e350a-da9e-4619-bd52-8cc23bb41ad5-c000.snappy.parquet",
            "raw_file": "3.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-08-28%20-%202022-09-04/part-00000-9a46dd05-4b06-4a39-a45b-5c8460b6c37b-c000.snappy.parquet",
            "raw_file": "4.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-09-04%20-%202022-09-11/part-00000-ea53b0e8-d346-44e3-9a87-1f60ac35c610-c000.snappy.parquet",
            "raw_file": "5.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-09-11%20-%202022-09-18/part-00000-f9afaec0-242e-41e7-906d-a42681515d75-c000.snappy.parquet",
            "raw_file": "6.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-09-18%20-%202022-09-25/part-00000-9ac876be-c07d-4a18-878d-959efa26f484-c000.snappy.parquet",
            "raw_file": "7.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-09-25%20-%202022-10-02/part-00000-be6d0798-554d-4c7a-9fef-d4c07aa0ce19-c000.snappy.parquet",
            "raw_file": "8.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-10-02%20-%202022-10-09/part-00000-2b76f9cc-0710-45e4-9e33-98ad5808ee79-c000.snappy.parquet",
            "raw_file": "9.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-10-09%20-%202022-10-16/part-00000-b2b625bc-5816-4586-b977-35f9ed4487fd-c000.snappy.parquet",
            "raw_file": "10.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-10-16%20-%202022-10-23/part-00000-9aeb279c-81c6-4481-9b30-d35d4d194fea-c000.snappy.parquet",
            "raw_file": "11.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-10-23%20-%202022-10-30/part-00000-23fdcfa3-9dd3-4c72-886c-e945bfcf92e1-c000.snappy.parquet",
            "raw_file": "12.parquet"
        }
    ]

    def __init__(self, 
                 data_dir: str, 
                 bin_size: int = 20, 
                 batch_size: int = 350, 
                 from_time: int = 0,
                 to_time: int = 26816821, # (Relative) timestamp of last event is 26816820.542023897
                 transforms: list = [],
                 num_workers: int = 0,
                 batch_split: list = [0.6, 0.25, 0.15], 
                 dataset_name: str = "UWF22Fall",
                 account_for_duration: bool = True):
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, num_workers, batch_split, dataset_name, account_for_duration)

        self.ts_first_event = 1639746045.251239 # This allows us to easily make time relative

        # Lastly, save those hyperparams
        self.save_hyperparameters()

class UWF24(UWF22):

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
                 num_workers: int = 0,
                 batch_split: list = [0.6, 0.25, 0.15],
                 dataset_name: str = "UWF24",
                 account_for_duration: bool = True):
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, num_workers, batch_split, dataset_name, account_for_duration)

        self.ts_first_event = 1709092837.805641 # This allows us to easily make time relative

        # Lastly, save those hyperparams
        self.save_hyperparameters()

class UWF24Fall(UWF22):

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
                 num_workers: int = 0,
                 batch_split: list = [0.6, 0.25, 0.15],
                 dataset_name: str = "UWF24Fall",
                 account_for_duration: bool = True):
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, num_workers, batch_split, dataset_name, account_for_duration)

        self.ts_first_event = 1726952812.207993 # This allows us to easily make time relative

        # Lastly, save those hyperparams
        self.save_hyperparameters()
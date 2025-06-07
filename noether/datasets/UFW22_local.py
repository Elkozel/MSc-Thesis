from multiprocessing.pool import ThreadPool
import os
import math
import logging
from typing import List, Literal
import requests
import torch
from tqdm import tqdm
import pandas as pd
from utils.SimpleBatch import SimpleBatch
from transforms.EnrichHost import EnrichHost
from torch.utils.data import random_split, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch
import lightning as L


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UFW22L(L.LightningDataModule):
    download_data = [
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-12%20-%202021-12-19/part-00000-7c2e9adb-5430-4792-a42b-10ff5bbd46e8-c000.snappy.parquet",
            "raw_file": "0.parquet",
            "pkl_file": "0.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-19%20-%202021-12-26/part-00000-3f86626a-1225-47f9-a5a2-0170b737e404-c000.snappy.parquet",
            "raw_file": "1.parquet",
            "pkl_file": "1.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-26%20-%202022-01-02/part-00000-b1a9fc13-8068-4a5d-91b2-871438709e81-c000.snappy.parquet",
            "raw_file": "2.parquet",
            "pkl_file": "2.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-02%20-%202022-01-09/part-00000-26e9208e-7819-451b-b23f-2e47f6d1e834-c000.snappy.parquet",
            "raw_file": "3.parquet",
            "pkl_file": "3.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-09%20-%202022-01-16/part-00000-36240b61-b84f-4164-a873-d7973e652780-c000.snappy.parquet",
            "raw_file": "4.parquet",
            "pkl_file": "4.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-16%20-%202022-01-23/part-00000-cbf26680-106d-40e7-8278-60520afdbb0e-c000.snappy.parquet",
            "raw_file": "5.parquet",
            "pkl_file": "5.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-02-06%20-%202022-02-13/part-00000-df678a79-4a73-452b-8e72-d624b2732f17-c000.snappy.parquet",
            "raw_file": "6.parquet",
            "pkl_file": "6.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-02-13%20-%202022-02-20/part-00000-1da06990-329c-4e38-913a-0f0aa39b388d-c000.snappy.parquet",
            "raw_file": "7.parquet",
            "pkl_file": "7.pkl"
        }
    ]
    conn_state_map = {
        "S0": 0,     # SYN seen, no SYN-ACK reply — connection attempt, no response.
        "S1": 1,     # Connection established, but not yet closed.
        "SF": 2,     # Normal connection: handshake completed and properly closed.
        "REJ": 3,    # Connection attempt rejected (RST sent).
        "S2": 4,     # Connection established, originator sent FIN, no response from responder.
        "S3": 5,     # Connection established, responder sent FIN, no response from originator.
        "RSTO": 6,   # Connection established, originator sent RST to abort.
        "RSTR": 7,   # Connection established, responder sent RST to abort.
        "RSTOS0": 8, # Originator sent SYN then RST — no SYN-ACK seen from responder.
        "RSTRH": 9,  # Responder sent SYN-ACK then RST — no SYN from originator seen.
        "SH": 10,    # Originator sent SYN followed by FIN — no SYN-ACK from responder.
        "SHR": 11,   # Responder sent SYN-ACK followed by FIN — no SYN from originator.
        "OTH": 12    # No SYN seen — just midstream traffic, possibly partial connection.
    }
    proto_map = {
        "icmp": 0,
        "tcp": 1,
        "udp": 2
    }
    tactic_map = {
        "none": 0,
        "Defense Evasion": 1,
        "Discovery": 2,
        "Exfiltration": 3,
        "Initial Access": 4,
        "Lateral Movement": 5,
        "Credential Access": 6,
        "Persistence": 7,
        "Privilege Escalation": 8,
        "Reconnaissance": 9,
        "Resource Development": 10
    }
    service_map = {
    }
    
    def __init__(self, data_dir, bin_size = 20, batch_size = 350, dataset_name = "UFW22", transforms = [], rnn_window = 30):
        super().__init__()# 91697
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "data.pkl")
        self.hostmap_file = os.path.join(data_dir, "hostmap.pkl")
        self.bin_size = bin_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.rnn_window = rnn_window

        self.save_hyperparameters()
        self.node_features = 6
        self.edge_features = 0
        self.file_stats = {}
        self.file_masks = {}
        self.hostmap = None
    
    def check_files(self):
        return os.path.exists(os.path.join(self.data_dir, self.hostmap_file)) and \
               all([os.path.exists(os.path.join(self.data_dir, link["pkl_file"])) for link in self.download_data])
    
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
        # Check if all files listed in self.download_data already exist in data_dir
        if all([os.path.exists(os.path.join(self.data_dir, link["raw_file"])) for link in self.download_data]):
            return  # If so, skip downloading

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
            
    def generate_hostmap(self):
        if os.path.exists(os.path.join(self.data_dir, self.hostmap_file)):
            return pd.read_pickle(os.path.join(self.data_dir, self.hostmap_file))

        # Build a list of full file paths for all raw data files
        all_files = [os.path.join(self.data_dir, file["raw_file"]) for file in self.download_data]
        
        # Read all Parquet files and concatenate them into a single DataFrame
        all_dfs = pd.concat([pd.read_parquet(file) for file in all_files], copy=False)

        # Extract unique source and destination IP addresses from the combined DataFrame
        all_hosts = set(all_dfs['src_ip_zeek'].unique()) \
            .union(set(all_dfs['dest_ip_zeek'].unique()))  # Combine into one set of all unique hosts

        # Create a DataFrame by enriching each unique host with additional info and assigning it an ID
        hostmap_df = pd.DataFrame([
            EnrichHost.enrich_host(host, id) for id, host in enumerate(all_hosts)
        ])

        # Save the resulting hostmap DataFrame as a pickle file for later use
        hostmap_df = hostmap_df.set_index("ip")
        hostmap_df.to_pickle(os.path.join(self.data_dir, self.hostmap_file))

        # compute minimal time ()
        min_time = float(all_dfs['ts'].min())

        # Return the hostmap DataFrame
        return hostmap_df, min_time

    def parquet_to_pickle(self, parquet_file, pickle_file, hostmap, min_time, tqdm_pos=0):
        # Initialize a progress bar with a total of 5 steps
        with tqdm(position=tqdm_pos, total=5, desc=f"Generating {os.path.basename(pickle_file)}") as pbar:
            
            # Step 1: Load the Parquet file into a DataFrame
            pbar.set_description(f"Loading parquet file")
            df = pd.read_parquet(parquet_file)
            pbar.update(1)

            # Step 2: Map categorical columns to more readable or standardized values
            pbar.set_description(f"Applying maps")
            df["conn_state"] = df["conn_state"].map(self.conn_state_map)
            df["proto"] = df["proto"].map(self.proto_map)
            df["label_tactic"] = df["label_tactic"].map(self.tactic_map)
            pbar.update(1)
            
            # Step 3: Map IP addresses to unique host IDs using the provided hostmap
            pbar.set_description(f"Applying hostmap")
            df['src_ip_id'] = df['src_ip_zeek'].map(hostmap["host_id"])
            df['dest_ip_id'] = df['dest_ip_zeek'].map(hostmap["host_id"])
            pbar.update(1)

            # Step 4: Adjust and normalize time data
            pbar.set_description(f"Making the time relative")
            df = df.rename(columns={"ts": "ts_abs"})  # Rename original timestamp to keep it
            df['ts'] = df['ts_abs'] - min_time  # Create new time column relative to min_time
            df = df.sort_values(by=['ts'])  # Sort records chronologically
            df = df.reset_index(drop=True)  # Reset index after sorting
            pbar.update(1)

            # Step 5: Save the transformed DataFrame as a pickle file
            pbar.set_description(f"Saving everything")
            df.to_pickle(pickle_file)  # Persist DataFrame for later use
            os.remove(parquet_file) # Remove the raw parquet file
            pbar.update(1)

        # Return the processed DataFrame
        return df
            
    def prepare_data(self):
        if self.check_files():
            return
        
        # Download files
        self.download_all_files()

        # Hostmap
        hostmap_df, min_time = self.generate_hostmap()

        # Generate pickle files
        with ThreadPool(8) as pool:
            args = [(
                os.path.join(self.data_dir, file["raw_file"]), 
                os.path.join(self.data_dir, file["pkl_file"]), 
                hostmap_df, 
                min_time, 
                pos
                ) for pos, file in enumerate(self.download_data, 1)]
            pool.map(lambda x: self.parquet_to_pickle(*x), args)
    
    def pickle_statistics(self, pickle_file: str):
        # Load the pickle file into a DataFrame
        df: pd.DataFrame = pd.read_pickle(os.path.join(self.data_dir, pickle_file))

        # Total number of elements in the DataFrame (not just rows)
        records = df.size

        # Find the minimum and maximum timestamps ('ts' is in seconds)
        start_ts = int(df["ts"].min())
        end_ts = int(df["ts"].max())

        # Calculate number of time bins based on bin size
        num_bins = len(range(start_ts, end_ts + 1, self.bin_size))

        # Store computed statistics in a dictionary associated with the file
        self.file_stats[pickle_file] = {
            "records": records,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "num_bins": num_bins,
            "num_batches": math.ceil(num_bins / self.batch_size)  # Number of batches is total bins divided by bin size
        }

        # Return statistics for the current file
        return self.file_stats[pickle_file]


    def generate_split(self, pickle_file: str):
        # Retrieve statistics for the given pickle file
        file_stats = self.file_stats[pickle_file]

        # Randomly split the indices into training (60%), validation (25%), and test (15%) sets
        generator = torch.Generator().manual_seed(42)
        idx = TensorDataset(torch.arange(0, file_stats["num_batches"]))
        _, val, test = random_split(idx, [0.6, 0.25, 0.15], generator=generator)

        # Initialize a tensor of zeros representing batch assignments
        batch_mask = torch.zeros(file_stats["num_batches"])

        # Assign value 1 for validation batches
        batch_mask[val.indices] = 1
        # Assign value 2 for test batches (training remains 0)
        batch_mask[test.indices] = 2

        # Return the batch assignment mask
        self.file_masks[pickle_file] = batch_mask
        return self.file_masks[pickle_file]
        
    def setup(self, stage: str):
        self.hostmap = pd.read_pickle(os.path.join(self.data_dir, self.hostmap_file))

        for file in self.download_data:
            self.pickle_statistics(file["pkl_file"])
            self.generate_split(file["pkl_file"])

    def df_to_data(self, df: pd.DataFrame, hostmap: pd.DataFrame):
        data = Data()
        data.time = torch.from_numpy(df["ts"].to_numpy()).to(dtype=torch.int64)
        data.x = torch.from_numpy(hostmap[[
                "internal",
                "broadcast",
                "multicast",
                "ipv4",
                "ipv6",
                "valid",
        ]].to_numpy()).to(torch.float32)
        data.edge_index = torch.from_numpy(df[["src_ip_id", "dest_ip_id"]].to_numpy().T).to(torch.int64)
        data.y = torch.from_numpy(df["label_tactic"].to_numpy()).to(torch.int64)

        return data

    def transform_data(self, data: Data) -> Data:
        for transform in self.transforms:
            data = transform(data)
        return data

    def bin_data(self, df_data: pd.DataFrame, bins, data: Data):
        # A couple of assertions
        assert data.time is not None
        assert data.edge_index is not None
        assert data.y is not None
        assert isinstance(data.y, torch.Tensor)

        # Cut the data into bins
        df_data["bin"] = pd.cut(df_data["ts"], bins, ordered=True, labels=False, right=False)
        bin_ranges = df_data.groupby("bin").apply(
            lambda g: pd.Series({
                "bin": g.name,
                "start": g.index[0],
                "end": g.index[-1] + 1
            }),
            include_groups=False  
        ).to_dict(orient="index")
        
        # Generate bins
        for bin_id, range in bin_ranges.items():
            start_idx = int(range["start"])
            end_idx = int(range["end"])
            yield Data(
                time=data.time[start_idx:end_idx],
                edge_index=data.edge_index[:, start_idx:end_idx],
                y=data.y[start_idx:end_idx],
                x=data.x
            )

    def batch_generator(self, stage: Literal['train', 'val', 'test']):
        assert self.hostmap is not None

        for file in self.download_data:
            pkl_file = file["pkl_file"]
            file_stats = self.file_stats[pkl_file]
            batch_mask = self.file_masks[pkl_file]

            df = pd.read_pickle(os.path.join(self.data_dir, pkl_file))
            data = self.df_to_data(df, self.hostmap)
            data_binned = list(self.bin_data(df, range(file_stats["start_ts"], file_stats["end_ts"], self.bin_size), data))
            data_transformed: List[BaseData] = list(map(self.transform_data, data_binned))
            for batch_num, batch_i in enumerate(range(0, len(data_transformed), self.batch_size)):
                batch = SimpleBatch.from_list(data_transformed[batch_i:batch_i+self.batch_size])
                stage_num = 0
                if stage == "val":
                    stage_num = 1
                elif stage == "test":
                    stage_num = 2
                if batch_mask[batch_num] != stage_num:
                    continue

                if batch.num_graphs <= self.rnn_window:
                    continue
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
        total_length = torch.cat(all_masks)
        return int(torch.sum(total_length == self.stage))
    
    def __iter__(self):
        return self.data_module.batch_generator(self.stage)

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError("Length is not defined for streaming dataset.")

if __name__ == "__main__":
    DATASET_DIR = "/data/datasets/UWF22"
    
    dataset = UFW22L(DATASET_DIR)
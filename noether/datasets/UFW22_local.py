from multiprocessing.pool import ThreadPool
import os
import logging
from random import randrange
import time
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map, process_map
import pandas as pd
from transforms.EnrichHost import EnrichHost
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, Dataset, DataLoader
import lightning as L


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UFW22L(L.LightningDataModule):
    download_data = [
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-12%20-%202021-12-19/part-00000-7c2e9adb-5430-4792-a42b-10ff5bbd46e8-c000.snappy.parquet",
            "file": "0.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-19%20-%202021-12-26/part-00000-3f86626a-1225-47f9-a5a2-0170b737e404-c000.snappy.parquet",
            "file": "1.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2021-12-26%20-%202022-01-02/part-00000-b1a9fc13-8068-4a5d-91b2-871438709e81-c000.snappy.parquet",
            "file": "2.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-02%20-%202022-01-09/part-00000-26e9208e-7819-451b-b23f-2e47f6d1e834-c000.snappy.parquet",
            "file": "3.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-09%20-%202022-01-16/part-00000-36240b61-b84f-4164-a873-d7973e652780-c000.snappy.parquet",
            "file": "4.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-01-16%20-%202022-01-23/part-00000-cbf26680-106d-40e7-8278-60520afdbb0e-c000.snappy.parquet",
            "file": "5.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-02-06%20-%202022-02-13/part-00000-df678a79-4a73-452b-8e72-d624b2732f17-c000.snappy.parquet",
            "file": "6.pkl"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekData22/parquet/2022-02-13%20-%202022-02-20/part-00000-1da06990-329c-4e38-913a-0f0aa39b388d-c000.snappy.parquet",
            "file": "7.pkl"
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
    
    def __init__(self, data_dir, bin_size = 20, dataset_name = "UFW22", transforms = [], train_split = [0, 2649600], val_split = [4719500, 4872400], test_split = [4925700, 5552200], rnn_window = 30):
        super().__init__()
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "data.pkl")
        self.hostmap_file = os.path.join(data_dir, "hostmap.pkl")
        self.bin_size = bin_size
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.rnn_window = rnn_window

        self.save_hyperparameters("dataset_name", "train_split", "val_split", "test_split", "rnn_window")
        self.node_features = 6
        self.edge_features = 0
    
    def check_files(self):
        return os.path.exists(os.path.join(self.data_dir, self.hostmap_file)) and \
               all([os.path.exists(os.path.join(self.data_dir, link["file"])) for link in self.download_data])
    
    def download(self, args):
        download_link, position = args
        with tqdm(position=position, total=5, desc=f"Downloading {download_link['file']}") as progress:
            df = pd.read_parquet(download_link["url"])
            progress.update(1)
            df["conn_state"] = df["conn_state"].map(self.conn_state_map)
            progress.update(1)
            df["proto"] = df["proto"].map(self.proto_map)
            progress.update(1)
            df["label_tactic"] = df["label_tactic"].map(self.tactic_map)
            progress.update(1)
            df.to_pickle(os.path.join(self.data_dir, download_link["file"]))
            progress.update(1)

        return df
    
    def prepare_hostmap(self, all_df):
        pbar = tqdm(total=3, desc=f"Generating hostmap")

        pbar.set_description(f"Generating hostmap")
        all_hosts = set(all_df['src_ip_zeek'].unique()) \
            .union(set(all_df['dest_ip_zeek'].unique()))
        pbar.update(1)
            
        pbar.set_description(f"Enriching nodemap")
        hostmap = pd.DataFrame([EnrichHost.enrich_host(host, id) for id, host in enumerate(all_hosts)])
        hostmap = hostmap.set_index('ip')['host_id']
        pbar.update(1)

        pbar.set_description(f"Saving hostmap")
        hostmap.to_pickle(os.path.join(self.data_dir, self.hostmap_file))
        pbar.update(1)
        
        return hostmap
    
    def additional_processing(self, args):
        df, file, min_time, hostmap, position = args
        with tqdm(total=4, desc=f"Preparing file {file}", position=position) as pbar:
            pbar.set_description(f"Applying hostmap")
            def apply_hostmap(df: pd.DataFrame):
                df['src_ip_id'] = df['src_ip_zeek'].map(hostmap)
                df['dest_ip_id'] = df['dest_ip_zeek'].map(hostmap)
                return df
            df = apply_hostmap(df)
            pbar.update(1)
            
            # Fixing time
            pbar.set_description(f"Making the time relative")
            def make_time_relative(df: pd.DataFrame):
                df = df.rename(columns={"ts": "ts_abs"})
                df['ts'] = df['ts_abs'] - min_time
                return df
            df = make_time_relative(df)
            pbar.update(1)
            
            pbar.set_description(f"Sorting data by time")
            def sort_by_time(df: pd.DataFrame):
                df = df.sort_values(by=['ts'])
                return df
            df = sort_by_time(df)
            pbar.update(1)
            
            # Save everything
            pbar.set_description(f"Saving everything")
            df.to_pickle(os.path.join(self.data_dir, file))
            pbar.update(1)
            
    def prepare_data(self):
        # Check if preparation has not been done before
        if self.check_files():
            return
        
        # Download files
        with ThreadPool(11) as pool:
            args = [(url, pos) for pos, url in enumerate(self.download_data, 1)]
            results = pool.map(self.download, args)

        # Hostmap
        all_df = pd.concat(results, copy=False)
        hostmap = self.prepare_hostmap(all_df)
        min_time = float(all_df['ts'].min())

        # Additional processing of the data
        with ThreadPool(11) as pool:
            all_df_zip = zip(results, self.download_data)
            args = [(df, url["file"], min_time, hostmap, pos) for pos, (df, url) in enumerate(all_df_zip, 1)]
            pool.map(self.additional_processing, args)

    def df_to_data(self, df: pd.DataFrame, hostmap: pd.DataFrame):
        device = "cuda"
        data = Data()
        data.time = torch.from_numpy(df["ts"].to_numpy()).to(device, dtype=torch.int64)
        data.x = torch.from_numpy(hostmap[[
                "internal",
                "broadcast",
                "multicast",
                "ipv4",
                "ipv6",
                "valid",
        ]].to_numpy()).to(device, dtype=torch.float32)
        data.edge_index = torch.from_numpy(df[["src_ip_id", "dest_ip_id"]].to_numpy().T).to(device, dtype=torch.int64)
        data.y = torch.from_numpy(df["label_tactic"].to_numpy()).to(device, dtype=torch.int64)

        return data

    def transform_data(self, data: Data):
        for transform in self.transforms:
            data = transform(data)
        return data

    def bin_data(self, df_data: pd.DataFrame, bins, data: Data):
        # Cut the data into bins
        df_data["bin"] = pd.cut(df_data["ts"], bins, labels=False)
        bin_ranges = df_data.groupby("bin").apply(lambda g: (g.index[0], g.index[-1] + 1)).to_dict()
        
        # Generate bins
        for bin_id, (start_idx, end_idx) in bin_ranges.items():
            yield Data(
                time=data.time[start_idx:end_idx],
                edge_index=data.edge_index[:, start_idx:end_idx],
                y=data.y[start_idx:end_idx],
                x=data.x
            )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # logger.info(f"Transfering batch {batch} to device {device}")
        return batch

        
    def setup_df(self, hostmap, filename):
        # logger.info(f"Load data")
        df: pd.DataFrame = pd.read_pickle(os.path.join(self.data_dir, filename))

        # logger.info(f"Making Data from df")
        data = self.df_to_data(df, hostmap)

        # logger.info(f"Binning data")
        from_ts = int(df["ts"].min())
        to_ts = int(df["ts"].max())+1
        data_binned = list(self.bin_data(df, range(from_ts, to_ts, self.bin_size), data))

        # logger.info(f"Transforming data")
        data_binned = list(map(self.transform_data, data_binned))
        
        # logger.info(f"Creating dataset")
        dataset = MovingWindowDataset(data_binned, window_size=self.rnn_window)

        # logger.info(f"Creating split")
        generator = torch.Generator().manual_seed(42)
        datasets = random_split(dataset, [0.6, 0.25, 0.15], generator=generator)

        return datasets
        
    def setup(self, stage: str):
        logger.info("Load hostmap")
        hostmap = pd.read_pickle(os.path.join(self.data_dir, self.hostmap_file))
        logger.info("Loading datafiles")
        args = [(hostmap, download_link["file"]) for download_link in self.download_data]
        datasets = thread_map(lambda x: self.setup_df(*x), args, max_workers=11)  # adjust max_workers as needed

        train, val, test = list(zip(*datasets))

        self.train_data = ConcatDataset(train)
        self.val_data = ConcatDataset(val)
        self.test_data = ConcatDataset(test)
    
    def train_dataloader(self):
        return self.train_data
    
    def val_dataloader(self):
        return DataLoader(self.val_data)
    
    def test_dataloader(self):
        return DataLoader(self.test_data)
    
class MovingWindowDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return max(len(self.data) - self.window_size + 1, 0)
    
    def gen_item(self, idx: int):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        return x, y

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            return [self.gen_item(id) for id in idx]
        
        return self.gen_item(idx)

if __name__ == "__main__":
    DATASET_DIR = "/data/datasets/UWF22"
    
    dataset = UFW22L(DATASET_DIR)
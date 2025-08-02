# https://csr.lanl.gov/data-fence/1750630801/cCYJvf9djgkHXw9h_OxGBC8wYyQ=/cyber1/auth.txt.gz

from multiprocessing.pool import ThreadPool
import os
import math
import logging
from typing import Any, Generator, Literal, Optional, Union
import requests
import torch
from tqdm import tqdm
import polars as pl
from ordered_set import OrderedSet
from torch.utils.data import random_split, TensorDataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.data import BaseData
from torch_geometric.data.batch import Batch
import lightning as L


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


auth_type_enum = pl.Enum(["MICROSOFT_AUTHENTICATION_PACKAG", "ACRONIS_RELOGON_AUTHENTICATION_PACKAGE", "MICROSOFT_AUTHENTICATIO", "MICROSOFT_AUTHENTICATION_PACKAGE_", "MICROSOFT_AUTHENTICATION_PACKA", "MICROSOFT_AUTHENTICATION_PAC", "MICROSOFT_AUTHENTICA", "MICROSOFT_AUTHENTICATION_PACKAGE_V1_0", "MICROSOFT_AUTHENTICATION_PACKAGE_V1", "Negotiate", "MICROSOFT_AUTHENTICATION_PACK", "Wave", "MICROSOFT_AUTHENTICATI", "MICROSOFT_AUTHENTICATION", "MICROSOFT_AUTHENTICATION_PACKAGE", "MICROSOFT_AUTHENTICATION_P", "?", "MICROSOFT_AUTHENTICATION_PACKAGE_V1_", "MICROSOFT_AUTHENTICATION_PACKAGE_V", "NETWARE_AUTHENTICATION_PACKAGE_V1_0", "N", "CygwinLsa", "MICROSOFT_AUTHENTICAT", "MICROSOFT_AUTHENTICATION_", "Kerberos", "NTLM", "MICROSOFT_AUTHENTICATION_PA", "Setuid", "TivoliAP"])
logon_type_enum = pl.Enum(["Batch", "Network", "Service", "NewCredentials", "RemoteInteractive", "?", "CachedInteractive", "NetworkCleartext", "Unlock", "Interactive"])
auth_orient_enum = pl.Enum(["TGT", "ScreenUnlock", "TGS", "AuthMap", "ScreenLock", "LogOff", "LogOn"])
result_enum = pl.Enum(["Fail", "Success"])


class LANLL(L.LightningDataModule):

    def __init__(self, 
                 data_dir: str,
                 bin_size: int = 20,
                 batch_size: int = 350,
                 from_time: int = 0,
                 to_time: int = 1209600, # First 14 days
                 lanl_URL: Optional[str] = None,
                 transforms: list = [],
                 batch_split: list = [0.6, 0.25, 0.15],
                 dataset_name: str = "LANL"):
        super().__init__()
        self.data_dir = os.path.join(data_dir, dataset_name)
        self.bin_size = bin_size
        self.from_time = from_time
        self.to_time = to_time
        self.batch_size = batch_size
        self.transforms = transforms
        self.batch_split = batch_split
        self.dataset_name = dataset_name
        self.edge_features = 4
        self.node_features = 0
        self.num_classes = 2

        self.save_hyperparameters()

        self.auth_file = {
            "csv": os.path.join(self.data_dir, "auth.txt.gz"),
            "parquet": os.path.join(self.data_dir, "auth.parquet")
        }
        self.redteam_file = {
            "csv": os.path.join(self.data_dir, "redteam.txt.gz"),
            "parquet": os.path.join(self.data_dir, "redteam.parquet")
        }
        
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
        if all([os.path.exists(os.path.join(self.data_dir, link["csv"])) for link in download_links]):
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
                file_data["csv"],
                idx  # tqdm progress bar position (starts at 1)
            ) for idx, file_data in enumerate(download_links, 1)]

            # Use map to apply download_file to all argument tuples in parallel
            pool.map(lambda x: self.download_file(*x), args)

    def csv_to_parquet(self):
        if not os.path.exists(self.redteam_file["parquet"]):
            redteam_header = [
                "time",
                "user@domain",
                "source computer",
                "destination computer"
            ]
            df = pl.scan_csv(self.redteam_file["csv"],
                        has_header=False,
                        new_columns=redteam_header)\
            .sink_parquet(
                self.redteam_file["parquet"],
                row_group_size=500_000
            )
            df = pl.read_parquet(self.redteam_file["parquet"])

        if not os.path.exists(self.auth_file["parquet"]):
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
            pl.scan_csv(self.auth_file["csv"],
                        has_header=False,
                        low_memory=True,
                        new_columns=columns)\
            .with_columns(
                pl.col("authentication type").cast(auth_type_enum).alias("authentication type"),
                pl.col("logon type").cast(logon_type_enum).alias("logon type"),
                pl.col("authentication orientation").cast(auth_orient_enum).alias("authentication orientation"),
                pl.col("success/failure").cast(result_enum).alias("success/failure"),
            ).sink_parquet( 
                self.auth_file["parquet"],
                compression="uncompressed",
                row_group_size=2_000_000
            )
    
    def generate_maps(self):
        # Check if the maps are already generated
        if os.path.exists(os.path.join(self.data_dir, "users.parquet")) and \
            os.path.exists(os.path.join(self.data_dir, "hosts.parquet")):
            return  # If so, skip
        
        auth = pl.scan_parquet(self.auth_file["parquet"])

        collected_auth = auth.select(
            "source computer",
            "destination computer",
            "source user@domain",
            "destination user@domain"
        )
        
        source_hosts = auth\
            .with_columns(
                host = pl.col("source computer")
            ).select("host").unique()
        destination_hosts = collected_auth\
            .with_columns(
                host = pl.col("destination computer")
            ).select("host").unique()
        hosts = pl.concat([source_hosts, destination_hosts]).unique()

        source_users = collected_auth\
            .with_columns(
                user = pl.col("source user@domain")
            ).select("user").unique()
        destination_users = collected_auth\
            .with_columns(
                user = pl.col("destination user@domain")
            ).select("user").unique()
        users = pl.concat([source_users, destination_users]).unique()

        hosts.sink_parquet(os.path.join(self.data_dir, "hosts.parquet"))
        users.sink_parquet(os.path.join(self.data_dir, "users.parquet"))

    def prepare_data(self):
        with tqdm(total=3) as pbar:
            pbar.set_description("Downloading files")
            self.download_all_files()
            pbar.update(1)
            pbar.set_description("Converting to parquet")
            self.csv_to_parquet()
            pbar.update(1)
            pbar.set_description("Generating maps")
            self.generate_maps()
            pbar.update(1)

    def generate_split(self, df: pl.LazyFrame) -> torch.Tensor:        
        # Filter only the timestamps we care about
        df = df.filter(
            (pl.col("time") >= self.from_time) & (pl.col("time") < self.to_time)
        )
        # Calculate bins
        df = df.with_columns(
            batch = pl.col("time") // self.bin_size // self.batch_size
        ).select("batch")

        collected_df = df.collect()
        if collected_df.is_empty():
            return torch.empty(0)

        batches = collected_df.select("batch").unique().to_series()
        num_batches = batches.len()

        # Randomly split the indices into training (60%), validation (25%), and test (15%) sets
        generator = torch.Generator().manual_seed(42)

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
        df = pl.scan_parquet(self.auth_file["parquet"])

        # Generate the batch masks
        self.batch_mask = self.generate_split(df)
        
        # Load the masks
        self.hostmap = pl.read_parquet(os.path.join(self.data_dir, "hosts.parquet"))
        self.hostenum = pl.Enum(self.hostmap.to_series())
        self.usermap = pl.read_parquet(os.path.join(self.data_dir, "users.parquet"))
        self.userenum = pl.Enum(self.usermap.to_series())

        # Load redteam
        redteam_header = [
            "time",
            "user@domain",
            "source computer",
            "destination computer"
        ]
        self.redteam = pl.scan_csv(
            self.redteam_file["csv"],
            separator=",",
            new_columns=redteam_header,
            has_header=False).with_columns(
                pl.col("user@domain").cast(self.userenum).alias("user@domain"),
                pl.col("source computer").cast(self.hostenum).alias("source computer"),
                pl.col("destination computer").cast(self.hostenum).alias("destination computer"),
                malicious = pl.lit(True).cast(pl.Boolean),
            )

    def transform_data(self, data: Union[Data, HeteroData]) -> BaseData:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def generate_bins(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Find only the time which is of interest
        df = df.filter(
            (pl.col("time") >= self.from_time) & (pl.col("time") < self.to_time)
        )

        # Apply the enums
        df = df.with_columns(
            pl.col("source user@domain").cast(self.userenum).alias("source user@domain"),
            pl.col("destination user@domain").cast(self.userenum).alias("destination user@domain"),
            pl.col("source computer").cast(self.hostenum).alias("source computer"),
            pl.col("destination computer").cast(self.hostenum).alias("destination computer"),

            pl.col("authentication type").cast(auth_type_enum).alias("authentication type"),
            pl.col("logon type").cast(logon_type_enum).alias("logon type"),
            pl.col("authentication orientation").cast(auth_orient_enum).alias("authentication orientation"),
            pl.col("success/failure").cast(result_enum).alias("success/failure"),
        )

        # Ensure 'time' is 
        df.with_columns(
            time = pl.col("time").cast(pl.Float32).drop_nans()
        )

        # Merge with redteam
        df = self.merge_with_redteam(df)

        # Create bins and batches
        df = df.with_columns(
            bin = pl.col("time") // self.bin_size,
        ).with_columns(
            batch = pl.col("bin") // self.batch_size,
        ).with_columns(
            bin = pl.col("bin") % self.batch_size
        )

        return df

    def generate_batches(self):
        batch_mask: torch.Tensor = self.batch_mask
        df = pl.scan_parquet(self.auth_file["parquet"])

        df = self.generate_bins(df)
        collected_df = df.collect()
        batches = collected_df.select("batch").unique().to_series()

        for batch_num in range(batch_mask.size(0)):
            batch = collected_df.filter(pl.col("batch") == batches[batch_num])
            batch_stage = batch_mask[batch_num].item()
            
            final_batch = [
                batch.filter(pl.col("bin") == bin_num)
                for bin_num in range(self.batch_size)
            ]
            yield final_batch, batch_stage
    
    def merge_with_redteam(self, df: pl.LazyFrame) -> pl.LazyFrame:
        
        # First, do a merge on source_user
        merge_source = df.join(
            self.redteam,
            left_on=['time', 'source user@domain', 'source computer', 'destination computer'],
            right_on=['time', 'user@domain', 'source computer', 'destination computer'],
            how='left'
        ).select("malicious").fill_null(False).with_columns(
            malicious_src = pl.col("malicious")
        ).select("malicious_src")

        # Then, do a merge on destination_user
        merge_dest = df.join(
            self.redteam,
            left_on=['time', 'destination user@domain', 'source computer', 'destination computer'],
            right_on=['time', 'user@domain', 'source computer', 'destination computer'],
            how='left'
        ).select("malicious").fill_null(False).with_columns(
            malicious_dest = pl.col("malicious")
        ).select("malicious_dest")

        # Add flags
        df = pl.concat([df, merge_source, merge_dest], how="horizontal")\
            .with_columns(
                malicious = (pl.col("malicious_src")) | (pl.col("malicious_dest"))
            ).drop(
                ["malicious_src", "malicious_dest"]
            )

        return df

    def df_to_data(self, df: pl.DataFrame):
        data = Data()
        data.time = df.select("time").to_torch(dtype=pl.Float32)
        data.x = torch.zeros((len(self.hostmap), 0))
        data.edge_index = df.select(pl.col("source computer").to_physical(), pl.col("destination computer").to_physical()).to_torch(dtype=pl.Int64).T
        data.edge_attr = df.select([
            pl.col("authentication type").to_physical(),
            pl.col("logon type").to_physical(),
            pl.col("authentication orientation").to_physical(),
            pl.col("success/failure").to_physical(),
        ]).fill_null(0).to_torch(dtype=pl.Float32)
        data.y = df.select(pl.col("malicious").to_physical()).to_torch(dtype=pl.Int8).flatten()
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
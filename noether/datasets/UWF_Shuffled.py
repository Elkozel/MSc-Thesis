import os
from typing import Literal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import polars as pl
import datasets.UWF

class UWF22(datasets.UWF.UWF22):
    def __init__(self, 
                 data_dir: str, 
                 bin_size: int = 20,
                 shuffle_bin_size: float = 0.5,
                 batch_size: int = 350, 
                 from_time: int = 0,
                 to_time: int = 26816821, # (Relative) timestamp of last event is 26816820.542023897
                 transforms: list = [],
                 num_workers: int = 0,
                 batch_split: list = [0.6, 0.25, 0.15], 
                 dataset_name: str = "UWF22",
                 account_for_duration: bool = False,
                 shuffle: bool = True,
                 shuffle_type: Literal["random", "day"] = "random",
                 shuffle_every_time: bool = False):
        
        self.original_dataset_name = dataset_name
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, num_workers, batch_split, self.original_dataset_name, account_for_duration)
        
        self.shuffle = shuffle
        self.shuffle_type = shuffle_type
        self.shuffle_every_time = shuffle_every_time
        self.shuffle_bin_size = shuffle_bin_size
        self.shuffled_dataset_name = f"{self.original_dataset_name}_{self.shuffle_type}_{shuffle_bin_size}"
        self.shuffled_dataset_dir = os.path.join(data_dir, self.shuffled_dataset_name)
        
        if self.shuffle and self.shuffle_type == "day":
            self.ts_first_event = 0


    def dataset_to_df(self):
        dfs = []

        for file in self.download_data:
            filename = os.path.join(self.data_dir, file["raw_file"])
            part = pl.scan_parquet(filename)
            dfs.append(part)

        df = pl.concat(dfs)
        
        return df
    
    def is_already_shuffled(self):
        all_files = [
            os.path.exists(os.path.join(self.shuffled_dataset_dir, file["raw_file"]))
            for file in self.download_data
        ]
        return not all(all_files)

    def shuffle_dataset(self, force = False):
        if not self.is_already_shuffled() and force == False:
            return
        
        df = self.dataset_to_df()
        df = self.shuffle_fn(df)

        # Split the dataframe into parts
        num_parts = len(self.download_data)
        df = df.sort(["ts"]).collect().to_pandas()
        parts = np.array_split(df, num_parts)

        os.makedirs(self.shuffled_dataset_dir, exist_ok=True)
        for i, file in enumerate(self.download_data):
            df_part: pd.DataFrame = parts[i] # type: ignore
            result_file = os.path.join(self.shuffled_dataset_dir, file["raw_file"])
            df_part.to_parquet(result_file, index=False)

    def shuffle_fn(self, df: pl.LazyFrame):
        if self.shuffle_type == "random":
            df = self.random_shuffle(df)
        elif self.shuffle_type == "day":
            df = self.day_shuffle(df)
        else:
            raise NotImplementedError(f"Shuffle {self.shuffle_type} is not implemented")
        
        return df
        
    def day_shuffle(self, df: pl.LazyFrame):
        df = df.with_columns(
            # For each event calculate the day 
            day = pl.col("datetime").dt.day(),
            day_start = pl.col("datetime").dt.truncate("1d")
        ).with_columns(
            # Find the relative time between the start of the day and the event
            rel_ts = pl.col("datetime") - pl.col("day_start")
        ).with_columns(
            ts = pl.col("rel_ts").dt.total_seconds()
        )
        
        # Clean up
        df = df.drop(["rel_ts", "day_start", "day"])

        return df
    
    def random_shuffle(self, df: pl.LazyFrame):
        # Create bins
        df = df.with_columns(
            bin = pl.col("ts") // self.shuffle_bin_size
        ).with_columns(
            ts_diff = pl.col("ts") - (pl.col("bin") * self.shuffle_bin_size)
        )

        # Shuffle groups        
        df = df.with_columns(
            bin = pl.col("bin").shuffle()
        )

        # Repair the ts
        df = df.with_columns(
            ts = pl.col("ts_diff") + (pl.col("bin") * self.shuffle_bin_size)
        )

        # Clean up
        df = df.drop(["bin", "ts_diff"]).sort("ts")

        return df

    def prepare_data(self):
        super().prepare_data()
        if self.shuffle:
            # Check if shuffle is needed
            self.shuffle_dataset(self.shuffle_every_time)
            # Copy the service and host maps
            pl.read_parquet(os.path.join(self.data_dir, "services.parquet"))\
                .write_parquet(os.path.join(self.shuffled_dataset_dir, "services.parquet"))
            pl.read_parquet(os.path.join(self.data_dir, "hosts.parquet"))\
                .write_parquet(os.path.join(self.shuffled_dataset_dir, "hosts.parquet"))
            # Set the new dataset dir
            self.data_dir = self.shuffled_dataset_dir
            self.dataset_name = self.shuffled_dataset_name



class UWF22Fall(datasets.UWF.UWF22Fall):
    def __init__(self, 
                 data_dir: str, 
                 bin_size: int = 20,
                 shuffle_bin_size: float = 0.5,
                 batch_size: int = 350, 
                 from_time: int = 0,
                 to_time: int = 26816821, # (Relative) timestamp of last event is 26816820.542023897
                 transforms: list = [],
                 num_workers: int = 0,
                 batch_split: list = [0.6, 0.25, 0.15], 
                 dataset_name: str = "UWF22Fall",
                 account_for_duration: bool = True,
                 shuffle: bool = True,
                 shuffle_type: Literal["random", "day"] = "random",
                 shuffle_every_time: bool = False):
        
        self.original_dataset_name = dataset_name
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, num_workers, batch_split, self.original_dataset_name, account_for_duration)
        
        self.shuffle = shuffle
        self.shuffle_type = shuffle_type
        self.shuffle_every_time = shuffle_every_time
        self.shuffle_bin_size = shuffle_bin_size
        self.shuffled_dataset_name = f"{self.original_dataset_name}_{self.shuffle_type}_{shuffle_bin_size}"
        self.shuffled_dataset_dir = os.path.join(data_dir, self.shuffled_dataset_name)
        
        if self.shuffle and self.shuffle_type == "day":
            self.ts_first_event = 0


    def dataset_to_df(self):
        dfs = []

        for file in self.download_data:
            filename = os.path.join(self.data_dir, file["raw_file"])
            part = pl.scan_parquet(filename)
            dfs.append(part)

        df = pl.concat(dfs)
        
        return df
    
    def is_already_shuffled(self):
        all_files = [
            os.path.exists(os.path.join(self.shuffled_dataset_dir, file["raw_file"]))
            for file in self.download_data
        ]
        return not all(all_files)

    def shuffle_dataset(self, force = False):
        if not self.is_already_shuffled() and force == False:
            return
        
        df = self.dataset_to_df()
        df = self.shuffle_fn(df)

        # Split the dataframe into parts
        num_parts = len(self.download_data)
        df = df.sort(["ts"]).collect().to_pandas()
        parts = np.array_split(df, num_parts)

        os.makedirs(self.shuffled_dataset_dir, exist_ok=True)
        for i, file in enumerate(self.download_data):
            df_part: pd.DataFrame = parts[i] # type: ignore
            result_file = os.path.join(self.shuffled_dataset_dir, file["raw_file"])
            df_part.to_parquet(result_file, index=False)

    def shuffle_fn(self, df: pl.LazyFrame):
        if self.shuffle_type == "random":
            df = self.random_shuffle(df)
        elif self.shuffle_type == "day":
            df = self.day_shuffle(df)
        else:
            raise NotImplementedError(f"Shuffle {self.shuffle_type} is not implemented")
        
        return df
        
    def day_shuffle(self, df: pl.LazyFrame):
        df = df.with_columns(
            # For each event calculate the day 
            day = pl.col("datetime").dt.day(),
            day_start = pl.col("datetime").dt.truncate("1d")
        ).with_columns(
            # Find the relative time between the start of the day and the event
            rel_ts = pl.col("datetime") - pl.col("day_start")
        ).with_columns(
            ts = pl.col("rel_ts").dt.total_seconds()
        )
        
        # Clean up
        df = df.drop(["rel_ts", "day_start", "day"])

        return df
    
    def random_shuffle(self, df: pl.LazyFrame):
        # Create bins
        df = df.with_columns(
            bin = pl.col("ts") // self.shuffle_bin_size
        ).with_columns(
            ts_diff = pl.col("ts") - (pl.col("bin") * self.shuffle_bin_size)
        )

        # Shuffle groups        
        df = df.with_columns(
            bin = pl.col("bin").shuffle()
        )

        # Repair the ts
        df = df.with_columns(
            ts = pl.col("ts_diff") + (pl.col("bin") * self.shuffle_bin_size)
        )

        # Clean up
        df = df.drop(["bin", "ts_diff"]).sort("ts")

        return df

    def prepare_data(self):
        super().prepare_data()
        if self.shuffle:
            # Check if shuffle is needed
            self.shuffle_dataset(self.shuffle_every_time)
            # Copy the service and host maps
            pl.read_parquet(os.path.join(self.data_dir, "services.parquet"))\
                .write_parquet(os.path.join(self.shuffled_dataset_dir, "services.parquet"))
            pl.read_parquet(os.path.join(self.data_dir, "hosts.parquet"))\
                .write_parquet(os.path.join(self.shuffled_dataset_dir, "hosts.parquet"))
            # Set the new dataset dir
            self.data_dir = self.shuffled_dataset_dir
            self.dataset_name = self.shuffled_dataset_name
        
class UWF24(datasets.UWF.UWF24):
    def __init__(self, 
                 data_dir: str, 
                 bin_size: int = 20,
                 shuffle_bin_size: float = 0.5,
                 batch_size: int = 350, 
                 from_time: int = 0,
                 to_time: int = 26816821, # (Relative) timestamp of last event is 26816820.542023897
                 transforms: list = [],
                 num_workers: int = 0,
                 batch_split: list = [0.6, 0.25, 0.15], 
                 dataset_name: str = "UWF24",
                 account_for_duration: bool = True,
                 shuffle: bool = True,
                 shuffle_type: Literal["random", "day"] = "random",
                 shuffle_every_time: bool = False):
        
        self.original_dataset_name = dataset_name
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, num_workers, batch_split, self.original_dataset_name, account_for_duration)
        
        self.shuffle = shuffle
        self.shuffle_type = shuffle_type
        self.shuffle_every_time = shuffle_every_time
        self.shuffle_bin_size = shuffle_bin_size
        self.shuffled_dataset_name = f"{self.original_dataset_name}_{self.shuffle_type}_{shuffle_bin_size}"
        self.shuffled_dataset_dir = os.path.join(data_dir, self.shuffled_dataset_name)
        
        if self.shuffle and self.shuffle_type == "day":
            self.ts_first_event = 0


    def dataset_to_df(self):
        dfs = []

        for file in self.download_data:
            filename = os.path.join(self.data_dir, file["raw_file"])
            part = pl.scan_parquet(filename)
            dfs.append(part)

        df = pl.concat(dfs)
        
        return df
    
    def is_already_shuffled(self):
        all_files = [
            os.path.exists(os.path.join(self.shuffled_dataset_dir, file["raw_file"]))
            for file in self.download_data
        ]
        return not all(all_files)

    def shuffle_dataset(self, force = False):
        if not self.is_already_shuffled() and force == False:
            return
        
        df = self.dataset_to_df()
        df = self.shuffle_fn(df)

        # Split the dataframe into parts
        num_parts = len(self.download_data)
        df = df.sort(["ts"]).collect().to_pandas()
        parts = np.array_split(df, num_parts)

        os.makedirs(self.shuffled_dataset_dir, exist_ok=True)
        for i, file in enumerate(self.download_data):
            df_part: pd.DataFrame = parts[i] # type: ignore
            result_file = os.path.join(self.shuffled_dataset_dir, file["raw_file"])
            df_part.to_parquet(result_file, index=False)

    def shuffle_fn(self, df: pl.LazyFrame):
        if self.shuffle_type == "random":
            df = self.random_shuffle(df)
        elif self.shuffle_type == "day":
            df = self.day_shuffle(df)
        else:
            raise NotImplementedError(f"Shuffle {self.shuffle_type} is not implemented")
        
        return df
        
    def day_shuffle(self, df: pl.LazyFrame):
        df = df.with_columns(
            # For each event calculate the day 
            day = pl.col("datetime").dt.day(),
            day_start = pl.col("datetime").dt.truncate("1d")
        ).with_columns(
            # Find the relative time between the start of the day and the event
            rel_ts = pl.col("datetime") - pl.col("day_start")
        ).with_columns(
            ts = pl.col("rel_ts").dt.total_seconds()
        )
        
        # Clean up
        df = df.drop(["rel_ts", "day_start", "day"])

        return df
    
    def random_shuffle(self, df: pl.LazyFrame):
        # Create bins
        df = df.with_columns(
            bin = pl.col("ts") // self.shuffle_bin_size
        ).with_columns(
            ts_diff = pl.col("ts") - (pl.col("bin") * self.shuffle_bin_size)
        )

        # Shuffle groups        
        df = df.with_columns(
            bin = pl.col("bin").shuffle()
        )

        # Repair the ts
        df = df.with_columns(
            ts = pl.col("ts_diff") + (pl.col("bin") * self.shuffle_bin_size)
        )

        # Clean up
        df = df.drop(["bin", "ts_diff"]).sort("ts")

        return df

    def prepare_data(self):
        super().prepare_data()
        if self.shuffle:
            # Check if shuffle is needed
            self.shuffle_dataset(self.shuffle_every_time)
            # Copy the service and host maps
            pl.read_parquet(os.path.join(self.data_dir, "services.parquet"))\
                .write_parquet(os.path.join(self.shuffled_dataset_dir, "services.parquet"))
            pl.read_parquet(os.path.join(self.data_dir, "hosts.parquet"))\
                .write_parquet(os.path.join(self.shuffled_dataset_dir, "hosts.parquet"))
            # Set the new dataset dir
            self.data_dir = self.shuffled_dataset_dir
            self.dataset_name = self.shuffled_dataset_name


class UWF24Fall(datasets.UWF.UWF24Fall):
    def __init__(self, 
                 data_dir: str, 
                 bin_size: int = 20,
                 shuffle_bin_size: float = 0.5,
                 batch_size: int = 350, 
                 from_time: int = 0,
                 to_time: int = 26816821, # (Relative) timestamp of last event is 26816820.542023897
                 transforms: list = [],
                 num_workers: int = 0,
                 batch_split: list = [0.6, 0.25, 0.15], 
                 dataset_name: str = "UWF24Fall",
                 account_for_duration: bool = True,
                 shuffle: bool = True,
                 shuffle_type: Literal["random", "day"] = "random",
                 shuffle_every_time: bool = False):
        
        self.original_dataset_name = dataset_name
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, num_workers, batch_split, self.original_dataset_name, account_for_duration)
        
        self.shuffle = shuffle
        self.shuffle_type = shuffle_type
        self.shuffle_every_time = shuffle_every_time
        self.shuffle_bin_size = shuffle_bin_size
        self.shuffled_dataset_name = f"{self.original_dataset_name}_{self.shuffle_type}_{shuffle_bin_size}"
        self.shuffled_dataset_dir = os.path.join(data_dir, self.shuffled_dataset_name)
        
        if self.shuffle and self.shuffle_type == "day":
            self.ts_first_event = 0


    def dataset_to_df(self):
        dfs = []

        for file in self.download_data:
            filename = os.path.join(self.data_dir, file["raw_file"])
            part = pl.scan_parquet(filename)
            dfs.append(part)

        df = pl.concat(dfs)
        
        return df
    
    def is_already_shuffled(self):
        all_files = [
            os.path.exists(os.path.join(self.shuffled_dataset_dir, file["raw_file"]))
            for file in self.download_data
        ]
        return not all(all_files)

    def shuffle_dataset(self, force = False):
        if not self.is_already_shuffled() and force == False:
            return
        
        df = self.dataset_to_df()
        df = self.shuffle_fn(df)

        # Split the dataframe into parts
        num_parts = len(self.download_data)
        df = df.sort(["ts"]).collect().to_pandas()
        parts = np.array_split(df, num_parts)

        os.makedirs(self.shuffled_dataset_dir, exist_ok=True)
        for i, file in enumerate(self.download_data):
            df_part: pd.DataFrame = parts[i] # type: ignore
            result_file = os.path.join(self.shuffled_dataset_dir, file["raw_file"])
            df_part.to_parquet(result_file, index=False)

    def shuffle_fn(self, df: pl.LazyFrame):
        if self.shuffle_type == "random":
            df = self.random_shuffle(df)
        elif self.shuffle_type == "day":
            df = self.day_shuffle(df)
        else:
            raise NotImplementedError(f"Shuffle {self.shuffle_type} is not implemented")
        
        return df
        
    def day_shuffle(self, df: pl.LazyFrame):
        df = df.with_columns(
            # For each event calculate the day 
            day = pl.col("datetime").dt.day(),
            day_start = pl.col("datetime").dt.truncate("1d")
        ).with_columns(
            # Find the relative time between the start of the day and the event
            rel_ts = pl.col("datetime") - pl.col("day_start")
        ).with_columns(
            ts = pl.col("rel_ts").dt.total_seconds()
        )
        
        # Clean up
        df = df.drop(["rel_ts", "day_start", "day"])

        return df
    
    def random_shuffle(self, df: pl.LazyFrame):
        # Create bins
        df = df.with_columns(
            bin = pl.col("ts") // self.shuffle_bin_size
        ).with_columns(
            ts_diff = pl.col("ts") - (pl.col("bin") * self.shuffle_bin_size)
        )

        # Shuffle groups        
        df = df.with_columns(
            bin = pl.col("bin").shuffle()
        )

        # Repair the ts
        df = df.with_columns(
            ts = pl.col("ts_diff") + (pl.col("bin") * self.shuffle_bin_size)
        )

        # Clean up
        df = df.drop(["bin", "ts_diff"]).sort("ts")

        return df

    def prepare_data(self):
        super().prepare_data()
        if self.shuffle:
            # Check if shuffle is needed
            self.shuffle_dataset(self.shuffle_every_time)
            # Copy the service and host maps
            pl.read_parquet(os.path.join(self.data_dir, "services.parquet"))\
                .write_parquet(os.path.join(self.shuffled_dataset_dir, "services.parquet"))
            pl.read_parquet(os.path.join(self.data_dir, "hosts.parquet"))\
                .write_parquet(os.path.join(self.shuffled_dataset_dir, "hosts.parquet"))
            # Set the new dataset dir
            self.data_dir = self.shuffled_dataset_dir
            self.dataset_name = self.shuffled_dataset_name
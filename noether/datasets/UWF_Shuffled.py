import os
from typing import Literal
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets.UWF import UWF22L

class UWF22S(UWF22L):
    def __init__(self, 
                 data_dir: str, 
                 bin_size: int = 20,
                 shuffle_bin_size: float = 0.5,
                 batch_size: int = 350, 
                 from_time: int = 0,
                 to_time: int = 26816821, # (Relative) timestamp of last event is 26816820.542023897
                 transforms: list = [],
                 batch_split: list = [0.6, 0.25, 0.15], 
                 dataset_name: str = "UWF22_Shuffled",
                 account_for_duration: bool = True,
                 shuffle: bool = True,
                 shuffle_type: Literal["random", "day"] = "random",
                 shuffle_every_time: bool = False):
        
        self.original_dataset_name = dataset_name.replace("_Shuffled", "")
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, batch_split, self.original_dataset_name, account_for_duration)
        
        self.shuffle = shuffle
        self.shuffle_type = shuffle_type
        self.shuffle_every_time = shuffle_every_time
        self.shuffle_bin_size = shuffle_bin_size
        self.shuffled_dataset_name = dataset_name
        self.shuffled_dataset_dir = os.path.join(data_dir, dataset_name)

        self.ts_first_event = 0

    def database_to_df(self):
        dfs = []

        for file in self.download_data:
            filename = os.path.join(self.data_dir, file["raw_file"])
            part = pd.read_parquet(filename)
            dfs.append(part)

        df = pd.concat(dfs)
        
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
        
        df = self.database_to_df()
        df = self.shuffle_fn(df)

        # Split the dataframe into parts
        num_parts = len(self.download_data)
        df = df.sort_values("ts")
        parts = np.array_split(df, num_parts)

        os.makedirs(self.shuffled_dataset_dir, exist_ok=True)
        for i, file in enumerate(self.download_data):
            df_part: pd.DataFrame = parts[i] # type: ignore
            result_file = os.path.join(self.shuffled_dataset_dir, file["raw_file"])
            df_part.to_parquet(result_file, index=False)

    def shuffle_fn(self, df: pd.DataFrame):
        if self.shuffle_type == "random":
            df = self.random_shuffle(df)
        elif self.shuffle_type == "day":
            df = self.day_shuffle(df)
        else:
            raise NotImplementedError(f"Shuffle {self.shuffle_type} is not implemented")
        
        return df
        
    def day_shuffle(self, df: pd.DataFrame):
        df["day"] = df["datetime"].dt.day
        df['day_start'] = df['datetime'].dt.normalize()
        df['rel_ts'] = df['datetime'] - df['day_start']
        df["ts"] = df["rel_ts"].dt.total_seconds()
        
        df = df.drop(columns=["rel_ts", "day_start", "day"])

        return df
    
    def random_shuffle(self, df: pd.DataFrame):
        # Create bins
        df['bin'] = df['ts'] // self.shuffle_bin_size
        df["ts_diff"] = df["ts"] - (df["bin"] * self.shuffle_bin_size)

        # Shuffle groups
        unique_groups = df["bin"].unique()
        shuffled_groups = np.random.permutation(unique_groups)

        group_mapping = dict(zip(unique_groups, shuffled_groups))
        df["bin"] = df["bin"].map(group_mapping)

        # Repair the ts
        df["ts"] = df["ts_diff"] + (df["bin"] * self.shuffle_bin_size)

        # Drop the extra columns
        df = df.drop(columns=["bin", "ts_diff"])
        df = df.sort_values(["ts"], ignore_index=True)

        return df

    def prepare_data(self):
        super().prepare_data()
        if self.shuffle:
            # Check if shuffle is needed
            self.shuffle_dataset(self.shuffle_every_time)
            # Set the new dataset dir
            self.data_dir = self.shuffled_dataset_dir
            self.dataset_name = self.shuffled_dataset_name
        
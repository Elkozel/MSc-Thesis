import os
from tqdm import tqdm
import pandas as pd
from elasticsearch.helpers import bulk
from typing import Callable, Optional
from datetime import datetime, timedelta

def check_elk(es):
    try:
        if es.ping():
            print("Elasticsearch is running.")
        else:
            print("Elasticsearch is not reachable.")
    except Exception as e:
        print(f"An error occurred trying to check for Elasticsearch: {e}")

class LANL_Loader():
    def __init__(self, lanl_folder, index_name):
        if not os.path.isdir(lanl_folder):
            raise ValueError(f"The path {lanl_folder} is not a valid directory.")
        
        self.index_name = index_name
        self.auth_file = os.path.join(lanl_folder, "auth.txt")
        self.redteam_file = os.path.join(lanl_folder, "redteam.txt")

        if not os.path.isfile(self.auth_file):
            raise FileNotFoundError(f"Error: {self.auth_file} does not exist.")
        if not os.path.isfile(self.redteam_file):
            raise FileNotFoundError(f"Error: {self.redteam_file} does not exist.")
    
    AUTH_FILE_SIZE = 1051430459
    AUTH_FILE_HEADERS = [
        "time", 
        "source user@domain", 
        "destination user@domain", 
        "source computer", 
        "destination computer", 
        "authentication type", 
        "logon type", 
        "authentication orientation", 
        "success/failure"
    ]
    def _authfile_generator(self, map: Optional[Callable[[dict], dict]] = None, filter: Optional[Callable[[dict], bool]] = None, chunksize: int = 50000):
        for chunk in pd.read_csv(self.auth_file, names=self.AUTH_FILE_HEADERS, chunksize=chunksize):
            for row in chunk.to_dict(orient="records"):
                if filter and filter(row):
                    if map:
                        row = map(row)
                    yield row
    
    REDTEAM_FILE_SIZE = 749
    REDTEAM_FILE_HEADERS = [
        "time", 
        "user@domain", 
        "source computer", 
        "destination computer"
    ]
    def _redteamfile_generator(self, map: Optional[Callable[[dict], dict]] = None, filter: Optional[Callable[[dict], bool]] = None, chunksize=50000):
        for chunk in pd.read_csv(self.redteam_file, names=self.REDTEAM_FILE_HEADERS, chunksize=chunksize):
            for row in chunk.to_dict(orient="records"):
                if filter and filter(row):
                    if map:
                        row = map(row)
                    yield row

    def create_index(self, es):
        if es.indices.exists(index=self.index_name):
            print(f"Index '{self.index_name}' already exists.")
            return

        settings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "time": {"type": "date", "format": "epoch_second"},
                    "source user@domain": {"type": "keyword"},
                    "destination user@domain": {"type": "keyword"},
                    "source computer": {"type": "keyword"},
                    "destination computer": {"type": "keyword"},
                    "authentication type": {"type": "keyword"},
                    "logon type": {"type": "keyword"},
                    "authentication orientation": {"type": "keyword"},
                    "success/failure": {"type": "keyword"},
                    "user@domain": {"type": "keyword"},
                    "file": {"type": "keyword"}
                }
            }
        }

        try:
            es.indices.create(index=self.index_name, body=settings)
            print(f"Index '{self.index_name}' created successfully.")
        except Exception as e:
            print(f"An error occurred while creating the index: {e}")
        
    def push_to_elastic(self, es, start_from_second=0):
        start_of_2015 = datetime(2015, 1, 1)
        def start_filter(row):
            return int(row["time"]) >= start_from_second
        def map_auth(row):
            row["_index"] = self.index_name
            row["timestamp"] = start_of_2015 + timedelta(seconds=row["time"])
            row["file"] = "auth"
            return row
        auth_generator = self._authfile_generator(map_auth, start_filter)
        # for success, info in bulk(es, tqdm(auth_generator, "Loading auth.txt into Elasticsearch", total=self.AUTH_FILE_SIZE)):
        #     if not success:
        #         print(f"Failed to index document: {info}")  

        def map_redteam(row):
            row["_index"] = self.index_name
            row["timestamp"] = start_of_2015 + timedelta(seconds=row["time"])
            row["file"] = "redteam"
            return row
        redteam_generator = self._redteamfile_generator(map_redteam, start_filter)
        for success, info in bulk(es, tqdm(redteam_generator, "Loading redteam.txt into Elasticsearch", total=self.REDTEAM_FILE_SIZE)):
            if not success:
                print(f"Failed to index document: {info}")

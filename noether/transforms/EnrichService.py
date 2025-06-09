import ipaddress
import torch
import pandas as pd
from typing import Union
from torch_geometric.utils import degree
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class EnrichService(BaseTransform):
    def __init__(self, col_id):
        super().__init__()
        self.col_id = col_id

    @staticmethod
    def enrich_service(service_port: int, service_id: int):
        return {
            "port": service_port,
            "service_id": service_id,
            "is_known": service_port < 1024
        }

    @staticmethod
    def enrich_service_df(service_df: pd.DataFrame, hostmap: pd.DataFrame):
        pass
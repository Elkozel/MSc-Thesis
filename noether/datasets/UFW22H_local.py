from typing import Any, List
import pandas as pd
import numpy as np
import torch
from datasets.UFW22_local import UFW22L
from torch_geometric.data import HeteroData

from noether.transforms.EnrichService import EnrichService
from noether.transforms.EnrichHost import EnrichHost


class UFW22HL(UFW22L):
    """
    A heterogeneous graph representation of the UFW22 dataset.
    """
    def __init__(self, data_dir, bin_size=20, batch_size=350, dataset_name="UFW22h", transforms=[], rnn_window=30):
        super().__init__(data_dir, bin_size, batch_size, dataset_name, transforms, rnn_window)
        self.node_features = 6
        self.edge_features = 12
        self.num_node_types = 2
        self.num_edge_types = 3
        self.edge_type_emb_dim = 1
        self.edge_attr_emb_dim = 1
        self.num_classes


    def df_to_data(self, df: pd.DataFrame): # type: ignore
        # Node types are "host" and "service"
        # A connection looks like this:
        #   (host) - uses -> (service) - communicates_with -> (service) - belongs_to -> (host)
        assert self.hostmap is not None
        assert self.servicemap is not None

        hostmap_df = pd.DataFrame([
            EnrichHost.enrich_host(host, id) for id, host in enumerate(self.hostmap)
        ])

        servicemap_df = pd.DataFrame([
            EnrichService.enrich_service(service, id) for id, service in enumerate(self.servicemap)
        ])

        data = HeteroData()
        # Create two node types "host" and "service":
        data['host'].x = torch.from_numpy(hostmap_df[[
                "internal",
                "broadcast",
                "multicast",
                "ipv4",
                "ipv6",
                "valid",
        ]].to_numpy()).to(torch.float32)
        data['service'].x = torch.from_numpy(servicemap_df[[
            "port",
            "isknown"
            # TODO: Add info about the port
        ]].astype(np.float32).to_numpy()).to(torch.float32)

        # Add (host) - uses -> (service)
        data['host', 'uses', 'service'].edge_index = torch.from_numpy(
            df[["src_ip_id", "src_service_id"]].to_numpy().T
            ).to(torch.int64)
        # TODO: Adde edge attr
        data['host', 'uses', 'service'].edge_attr = torch.empty((len(df), 0)).to(torch.float32)
        
        data['host', 'uses', 'service'].time = torch.from_numpy(df["ts"].to_numpy(dtype=float)).to(dtype=torch.float64)
        data['host', 'uses', 'service'].y = torch.from_numpy(df["label_tactic"].to_numpy()).to(torch.int64)

        # Add (service) - communicates_with -> (service)
        data['service', 'communicates_with', 'service'].edge_index = torch.from_numpy(
            df[["src_service_id", "dest_service_id"]].to_numpy().T
            ).to(torch.int64)
        data['service', 'communicates_with', 'service'].edge_attr = torch.from_numpy(df[[
                "conn_status",
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
                # TODO: Add history
        ]].astype(float).to_numpy()).to(torch.float32)
        data['service', 'communicates_with', 'service'].time = torch.from_numpy(df["ts"].to_numpy(dtype=float)).to(dtype=torch.float64)
        data['service', 'communicates_with', 'service'].y = torch.from_numpy(df["label_tactic"].to_numpy()).to(torch.int64)

        # Add (service) - belongs_to -> (host)
        data['service', 'belongs_to', 'host'].edge_index = torch.from_numpy(
            df[["dest_service_id", "dest_ip_id"]].to_numpy().T
            ).to(torch.int64)
        # TODO: Adde edge attr
        data['service', 'belongs_to', 'host'].edge_attr = torch.empty((len(df), 0)).to(torch.float32)
        data['service', 'belongs_to', 'host'].time = torch.from_numpy(df["ts"].to_numpy(dtype=float)).to(dtype=torch.float64)
        data['service', 'belongs_to', 'host'].y = torch.from_numpy(df["label_tactic"].to_numpy()).to(torch.int64)
        
        yield data
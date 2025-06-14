from typing import Any, Hashable, List
import pandas as pd
import numpy as np
import torch
from datasets.UFW22_local import UFW22L
from torch_geometric.data import HeteroData


class UFW22HL(UFW22L):
    """
    A heterogeneous graph representation of the UFW22 dataset.
    """
    def __init__(self, data_dir, bin_size=20, batch_size=350, dataset_name="UFW22", transforms=[], rnn_window=30):
        super().__init__(data_dir, bin_size, batch_size, dataset_name, transforms, rnn_window)

    def df_to_data(self, df: pd.DataFrame, bin_ranges: List[Any]): # type: ignore
        # Node types are "host" and "service"
        # A connection looks like this:
        #   (host) - uses -> (service) - communicates_with -> (service) - belongs_to -> (host)
        assert self.hostmap is not None
        assert self.servicemap is not None

        data = HeteroData()
        # Create two node types "host" and "service":
        data['host'].x = torch.from_numpy(self.hostmap[[
                "internal",
                "broadcast",
                "multicast",
                "ipv4",
                "ipv6",
                "valid",
        ]].to_numpy()).to(torch.float32)
        data['service'].x = torch.from_numpy(self.servicemap[[
            "port",
            "isknown"
            # TODO: Add info about the port
        ]].astype(np.float32).to_numpy()).to(torch.float32)

        # Add (host) - uses -> (service)
        data['host', 'uses', 'service'].edge_index = torch.from_numpy(
            df[["src_ip_id", "src_service_id"]].to_numpy().T
            ).to(torch.int64)
        # TODO: Adde edge attr
        data.edge_attr = torch.empty((len(data), 0)).to(torch.float32)

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

        # Add (service) - belongs_to -> (host)
        data['service', 'belongs_to', 'host'].edge_index = torch.from_numpy(
            df[["dest_service_id", "dest_ip_id"]].to_numpy().T
            ).to(torch.int64)
        # TODO: Adde edge attr
        data['service', 'belongs_to', 'host'].edge_attr = torch.empty((len(data), 0)).to(torch.float32)

        # add labels
        data.y = torch.from_numpy(df["label_tactic"].to_numpy()).to(torch.int64)

        # Generate bins
        for range in bin_ranges:
            if range["empty"]:
                bin_data = HeteroData()

                yield bin_data
            else:
                start_idx = int(range["start"])
                end_idx = int(range["end"])
                bin_data = HeteroData()

                # Bin edges
                bin_data['service', 'belongs_to', 'host'].edge_index = \
                    data['service', 'belongs_to', 'host'].edge_index[start_idx:end_idx]
                bin_data['service', 'communicates_with', 'service'].edge_index = \
                    data['service', 'communicates_with', 'service'].edge_index[start_idx:end_idx]
                
                # Bin edge attributes
                bin_data['service', 'belongs_to', 'host'].edge_attr = \
                    data['service', 'belongs_to', 'host'].edge_attr[start_idx:end_idx]
                bin_data['service', 'communicates_with', 'service'].edge_attr = \
                    data['service', 'communicates_with', 'service'].edge_attr[start_idx:end_idx]

                # Bin labels
                bin_data.y = data.y[start_idx:end_idx]
                
                # Bin node features
                bin_data['service'].x = data['service'].x
                bin_data['host'].x = data['host'].x

                yield bin_data
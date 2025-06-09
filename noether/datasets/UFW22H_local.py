from typing import Any, Hashable
import pandas as pd
import torch
from datasets.UFW22_local import UFW22L
from torch_geometric.data import HeteroData


class UFW22HL(UFW22L):
    """
    A heterogeneous graph representation of the UFW22 dataset.
    """

    def df_to_data(self, df: pd.DataFrame, bin_ranges: dict[Hashable, Any]):
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
        data['service'].x = torch.randn(500, 20)
        # torch.from_numpy(self.servicemap[[
        #     "port",
        #     "is_known"
        # ]].to_numpy()).to(torch.float32)

        # Add service belongs to host edges
        data['service', 'belongs_to', 'host'].torch.from_numpy(df[["src_ip_id", "dest_ip_id"]].to_numpy().T).to(torch.int64)
        data.edge_attr = torch.from_numpy(df[[
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
        ]].astype(float).to_numpy()).to(torch.float32)

        # Add service communicates with service edges
        data['service', 'communicates_with', 'service'].edge_index = torch.from_numpy(
            df[["src_service_id", "dest_service_id"]].to_numpy().T
            ).to(torch.int64)
        data['service', 'communicates_with', 'service'].edge_attr = torch.from_numpy(df[[
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
        ]].astype(float).to_numpy()).to(torch.float32)


        data.time = torch.from_numpy(df["ts"].to_numpy()).to(dtype=torch.int64)
        data.x = torch.from_numpy(self.hostmap[[
                "internal",
                "broadcast",
                "multicast",
                "ipv4",
                "ipv6",
                "valid",
        ]].to_numpy()).to(torch.float32)
        data.edge_index = torch.from_numpy(df[["src_ip_id", "dest_ip_id"]].to_numpy().T).to(torch.int64)
        data.edge_attr = torch.from_numpy(df[[
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
        ]].astype(float).to_numpy()).to(torch.float32)
        data.y = torch.from_numpy(df["label_tactic"].to_numpy()).to(torch.int64)

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
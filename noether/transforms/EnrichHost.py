import ipaddress
import torch
from typing import Union
from torch_geometric.utils import degree
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class EnrichHost(BaseTransform):
    def __init__(self, col_id):
        super().__init__()
        self.col_id = col_id

    @staticmethod
    def enrich_host(host: str, host_id: int):
        try:
            ip_obj = ipaddress.ip_address(host)
            return {
                "ip": host,
                "host_id": host_id,
                "internal": ip_obj.is_private,
                "broadcast": ip_obj == ipaddress.IPv4Address('255.255.255.255'),
                "multicast": ip_obj.is_multicast,
                "ipv4": ip_obj.version == 4,
                "ipv6": ip_obj.version == 6,
                "valid": True
            }
            
        except ValueError:
            return {
                "ip": host,
                "host_id": host_id,
                "internal": False,
                "broadcast": False,
                "multicast": False,
                "ipv4": False,
                "ipv6": False,
                "valid": False
            }

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if self.col_id not in store:
                continue

            host_ids = torch.arange(store[self.col_id].size(0))
            host_list = store[self.col_id]

            # Convert from tensor to list of strings
            if isinstance(host_list, torch.Tensor):
                host_list = [h.decode("utf-8") if isinstance(h, bytes) else str(h) for h in host_list]

            enriched = [self.enrich_host(host, int(host_id)) for host, host_id in zip(host_list, host_ids)]

            # Create new tensors for each enriched attribute
            internal = torch.tensor([e["internal"] for e in enriched], dtype=torch.bool)
            broadcast = torch.tensor([e["broadcast"] for e in enriched], dtype=torch.bool)
            multicast = torch.tensor([e["multicast"] for e in enriched], dtype=torch.bool)
            ipv4 = torch.tensor([e["ipv4"] for e in enriched], dtype=torch.bool)
            ipv6 = torch.tensor([e["ipv6"] for e in enriched], dtype=torch.bool)
            valid = torch.tensor([e["valid"] for e in enriched], dtype=torch.bool)

            # Add to the node store
            store["is_internal"] = internal
            store["is_broadcast"] = broadcast
            store["is_multicast"] = multicast
            store["is_ipv4"] = ipv4
            store["is_ipv6"] = ipv6
            store["is_valid_ip"] = valid

        return data
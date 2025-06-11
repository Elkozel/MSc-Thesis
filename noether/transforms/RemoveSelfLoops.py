from typing import List, Union
import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

def find_self_loops(edge_index):
    src = edge_index[0]
    dest = edge_index[1]
    
    return torch.nonzero(src!=dest, as_tuple=True)[0]

# @functional_transform('remove_duplicated_edges_temporal')
class RemoveSelfLoops(BaseTransform):
    r"""Removes all self-loops in the given homogeneous or heterogeneous
    graph (functional name: :obj:`remove_self_loops`).

    Args:
        attr (str, optional): The name of the attribute of edge weights
            or multi-dimensional edge features to pass to
            :meth:`torch_geometric.utils.remove_self_loops`.
            (default: :obj:`"edge_weight"`)
    """
    def __init__(self, attr: Union[str, List[str]] = ['edge_attr', 'edge_weight', 'time'],) -> None:
        if isinstance(attr, str):
            attr = [attr]
            
        self.attr = attr

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite():
                continue
            # Skip empty stores
            if torch.numel(store.edge_index) == 0:
                continue
            
            keys = [key for key in self.attr if key in store]
            mask = find_self_loops(store.edge_index)
            
            store.edge_index = store.edge_index[:, mask]
            for key in keys:
                store[key] = store[key][mask]

        return data
from typing import List, Union

import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

def find_duplicated_edges(edge_index):
    _, indexes = torch.unique(edge_index, dim=1, return_inverse=True)
    return torch.unique(indexes)

# @functional_transform('remove_duplicated_edges_temporal')
class RemoveDuplicatedEdges(BaseTransform):
    r"""Removes duplicated edges from a given homogeneous or heterogeneous
    graph. Useful to clean-up known repeated edges/self-loops in common
    benchmark datasets, *e.g.*, in :obj:`ogbn-products`.
    (functional name: :obj:`remove_duplicated_edges_temporal`).

    Args:
        key (str or [str], optional): The name of edge attribute(s) to merge in
            case of duplication. (default: :obj:`["edge_weight", "edge_attr"]`)
        reduce (str, optional): The reduce operation to use for merging edge
            attributes (:obj:`"add"`, :obj:`"mean"`, :obj:`"min"`,
            :obj:`"max"`, :obj:`"mul"`). (default: :obj:`"add"`)
    """
    def __init__(
        self,
        key: Union[str, List[str]] = ['edge_attr', 'edge_weight', 'time'],
        reduce: str = "add",
    ) -> None:
        if isinstance(key, str):
            key = [key]

        self.keys = key
        self.reduce = reduce
        
    @staticmethod
    def find_duplicated_edges(edge_index: torch.Tensor):
    
        _, indexes = torch.unique(edge_index, dim=1, return_inverse=True)
        return torch.unique(indexes)

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.edge_stores:
            # Skip empty stores
            if torch.numel(store.edge_index) == 0:
                continue
            
            keys = [key for key in self.keys if key in store]
            mask = RemoveDuplicatedEdges.find_duplicated_edges(store.edge_index)
            
            store.edge_index = store.edge_index[:, mask]
            for key in keys:
                store[key] = store[key][mask]

        return data
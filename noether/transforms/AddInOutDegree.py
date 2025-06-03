import torch
from typing import Union
from torch_geometric.utils import degree
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform

class AddInOutDegree(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            edge_index = data.edge_stores[0].edge_index

            in_degree = degree(edge_index[0], store.num_nodes).unsqueeze(1)
            out_degree = degree(edge_index[1], store.num_nodes).unsqueeze(1)

            store = torch.cat([
                store.x,
                in_degree,
                out_degree
            ], dim=1)

        return data
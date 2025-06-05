import torch
from typing import List, Self
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData

class SimpleBatch(Data):
    @classmethod
    def from_list(cls, graphs: List[BaseData]) -> Self:
        all_edges = [g.edge_index for g in graphs if g.edge_index is not None]
        all_y = [g.y for g in graphs if isinstance(g.y, torch.Tensor)]

        ptr = [0]
        for g in graphs:
            ptr.append(ptr[-1] + g.num_edges)
        ptr.pop()

        batch = cls()
        batch.x = graphs[0].x
        batch.y = torch.cat(all_y)
        batch.edge_index = torch.cat(all_edges, dim=1)
        batch.ptr = ptr

        return batch

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return len(self.ptr) if hasattr(self, 'ptr') else 0

    def get_example(self, idx):
        # Assertions
        assert self.edge_index is not None
        assert self.x is not None
        assert isinstance(self.y, torch.Tensor)

        # Locate the start and end point
        start = self.ptr[idx]
        end = self.ptr[idx + 1] if idx + 1 < len(self.ptr) else self.edge_index.size(1)

        # Mask for selecting edges
        edge_index = self.edge_index[:, start:end]
        y = self.y[start:end]

        return Data(x=self.x, edge_index=edge_index, y=y)
    
    def index_select(self, idxs):
        """
        Select a subset of graphs from the batch using indices or a slice.

        Args:
            idxs (list[int] | torch.Tensor | slice): Indices of graphs to extract.

        Returns:
            SimpleBatch: A new batch with the selected graphs.
        """
        if isinstance(idxs, slice):
            idxs = list(range(self.num_graphs))[idxs]
        elif isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()

        selected_graphs = [self.get_example(i) for i in idxs]
        return selected_graphs
    
    def to_data_list(self):
        return [self.get_example(id) for id in range(self.num_graphs)]
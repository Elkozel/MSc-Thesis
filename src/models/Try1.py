import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x  # node embeddings
    
class RNNEncoder(nn.Module):
    def __init__(self, hidden_channels, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(hidden_channels, hidden_channels, num_layers=num_layers,  batch_first=True)

    def forward(self, x_seq):
        # x_seq shape: [num_nodes, sequence_length, hidden_channels]
        output, final_state = self.rnn(x_seq)
        return output, final_state  # optional depending on decoder
    
class DotProductDecoder(nn.Module):
    def forward(self, z_src, z_dst):
        return (z_src * z_dst).sum(dim=1)
    
class FullModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gnn = GNNEncoder(in_channels, hidden_channels)
        self.rnn = RNNEncoder(hidden_channels)
        self.decoder = DotProductDecoder()

    def forward(self, graph_sequence, edge_pairs):
        # Generate the features at each timestamp
        # shape (timestamp, nodes, features)
        graph_features = [self.gnn(data.x, data.edge_index) for data in graph_sequence]

        # Switch the order, so that each node is treated as a batch
        # shape (nodes, timestamp, features)
        graph_features = torch.stack(graph_features)
        graph_features = graph_features.permute(1, 0, 2)
        rnn_output, _ = self.rnn(graph_features)

        # Get the embeddings at time t+1
        # shape (nodes, timestamp (t+1), features)
        predicted_embeddings = rnn_output[:, -1, :]  # use last time step

        # Generate the scores for 
        src = edge_pairs[0]
        dst = edge_pairs[1]
        scores = self.decoder(predicted_embeddings[src], predicted_embeddings[dst])  # shape [num_edges]

        return scores


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.Try1 import LitFullModel as Try1

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_features):
        out = self.conv1(x, edge_index)
        out = F.relu(out)
        out = self.conv2(out, edge_index)
        out = F.relu(out)
        out = self.conv3(out, edge_index)
        return out  # node embeddings
    
class FakeRNN(nn.Module):
    def forward(self, x_seq):
        return x_seq
    
class MLPDecoder(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        
    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        # src,dst: [num_edges, embed_dim]
        h = torch.cat([src, dst], dim=-1)    # → [num_edges, 2*embed_dim]
        return self.net(h).squeeze(-1)       # → [num_edges]

class LinkTypeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, h_src, h_dst, link_pred = torch.Tensor([])):
        h_concat = torch.cat([h_src, h_dst, link_pred], dim=-1)
        return self.mlp(h_concat)  # No softmax; use CrossEntropyLoss
    
class LitFullModel(Try1):
    """
    PyTorch Lightning module for temporal graph-based edge classification using a GNN-RNN architecture.

    This model processes a sequence of network activity graphs using a Graph Neural Network (GNN),
    encodes temporal evolution with a Recurrent Neural Network (RNN), and predicts whether edges
    in the final graph are malicious or benign using a bilinear decoder.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden units for both the GNN and RNN layers.
        dropout_rate (float): Dropout probability used in the GNN encoder.
        out_classes (int, optional): Number of output classes. Defaults to 1 for binary classification.
        rnn_window (int, optional): Length of the input sequence (number of time steps). Defaults to 30.
        rnn_num_layers (int, optional): Number of layers in the RNN. Defaults to 1.
        binary_threshold (float, optional): Threshold for binary classification metrics. Defaults to 0.5.
        model_name (str, optional): Optional model identifier. Defaults to "Try2".
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout_rate,
        out_classes = 1,
        rnn_window_size = 30,
        rnn_num_layers = 1,
        binary_threshold = 0.5,
        negative_edge_sampling_min = 20,
        pred_alpha = 0.8,
        link_pred_only = False,
        model_name="Try0"
    ):
        super().__init__(in_channels, hidden_channels, dropout_rate, out_classes, rnn_window_size, rnn_num_layers, binary_threshold, negative_edge_sampling_min, pred_alpha, link_pred_only, model_name)


        self.gnn = GNNEncoder(in_channels, hidden_channels)
        self.rnn = FakeRNN()
        self.link_pred = MLPDecoder(hidden_channels, 1)
        self.link_classifier = LinkTypeClassifier(hidden_channels, out_classes)
        self.save_hyperparameters()
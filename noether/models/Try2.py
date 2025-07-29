import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from models.Try1 import LitFullModel as Try1


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_rate, edge_dim):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels*3, edge_dim=edge_dim)
        self.norm1 = nn.LayerNorm(hidden_channels*3)
        self.conv2 = GATv2Conv(hidden_channels*3, hidden_channels*2, edge_dim=edge_dim)
        self.norm2 = nn.LayerNorm(hidden_channels*2)
        self.conv3 = GATv2Conv(hidden_channels*2, hidden_channels, edge_dim=edge_dim)
        self.norm3 = nn.LayerNorm(hidden_channels)
        
        self.dropout = dropout_rate

    def forward(self, x, edge_index, edge_features):
        out = self.conv1(x, edge_index, edge_attr=edge_features)
        out = self.norm1(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out, edge_index, edge_attr=edge_features)
        out = self.norm2(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv3(out, edge_index, edge_attr=edge_features)
        out = self.norm3(out)
        return out  # node embeddings
    
class RNNEncoder(nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(hidden_channels, hidden_channels, num_layers=num_layers,  batch_first=True)

    def forward(self, x_seq):
        # x_seq shape: [num_nodes, sequence_length, hidden_channels]
        output, final_state = self.rnn(x_seq)
        return output, final_state  # optional depending on decoder
    
class LinkPredictor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(in_dim, in_dim, out_dim)

    def forward(self, z_src, z_dst):
        return self.bilinear(z_src, z_dst).squeeze(-1)
    
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
        hidden_channels = None,
        dropout_rate = 0.0,
        edge_dim = None,
        out_classes = 1,
        rnn_window_size = 30,
        rnn_num_layers = 1,
        binary_threshold = 0.5,
        negative_edge_sampling_min = 20,
        pred_alpha = 0.8,
        model_name="Try2"
    ):
        if hidden_channels is None:
            hidden_channels = in_channels * 3
        super().__init__(in_channels, hidden_channels, dropout_rate, out_classes, rnn_window_size, 
                         rnn_num_layers, binary_threshold, negative_edge_sampling_min, pred_alpha, model_name)
        
        self.gnn = GNNEncoder(in_channels, hidden_channels, dropout_rate, edge_dim)
        self.rnn = RNNEncoder(hidden_channels, rnn_num_layers)
        self.link_pred = LinkPredictor(hidden_channels, 1)
        self.link_classifier = LinkTypeClassifier(hidden_channels, out_classes)
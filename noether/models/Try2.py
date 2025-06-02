import torch
import torchmetrics
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data, Batch
import torchmetrics.classification


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout_rate):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels*3)
        self.norm1 = nn.LayerNorm(hidden_channels*3)
        self.conv2 = GATv2Conv(hidden_channels*3, hidden_channels*2)
        self.norm2 = nn.LayerNorm(hidden_channels*2)
        self.conv3 = GATv2Conv(hidden_channels*2, hidden_channels)
        self.norm3 = nn.LayerNorm(hidden_channels)
        
        self.dropout = dropout_rate

    def forward(self, x, edge_index):
        out = self.conv1(x, edge_index)
        out = self.norm1(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out, edge_index)
        out = self.norm2(out)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv3(out, edge_index)
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
    
class BilinearDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(in_dim, in_dim, out_dim)

    def forward(self, z_src, z_dst):
        return self.bilinear(z_src, z_dst).squeeze(-1)

class LitFullModel(L.LightningModule):
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
        out_classes=1,
        rnn_window_size=30,
        rnn_num_layers=1,
        binary_threshold=0.5,
        model_name="Try2"
    ):
        super().__init__()

        self.gnn = GNNEncoder(in_channels, hidden_channels, dropout_rate)
        self.rnn = RNNEncoder(hidden_channels, rnn_num_layers)
        self.decoder = BilinearDecoder(hidden_channels, out_classes)

        self.binary_acc = torchmetrics.classification.BinaryAccuracy(threshold=binary_threshold)

        self.rnn_window = rnn_window_size
        self.model_name = model_name
        self.binary_threshold = binary_threshold

        self.save_hyperparameters()


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

        # Generate the scores
        src = edge_pairs[0]
        dst = edge_pairs[1]
        scores = self.decoder(predicted_embeddings[src], predicted_embeddings[dst])  # shape [num_edges]

        return scores

    def training_step(self, batch: Batch, batch_idx):
        """Trains the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window + 1)
        
        if num_windows < 1:
            print("Training batch did not have enough")
            
        total_loss = 0
        total_acc = 0
            
        for i in range(num_windows):
            to_idx = i + self.window_size
            x_graphs = batch.index_select(slice(i, to_idx))
            y_graph = batch.get_example(to_idx)


            # Positive and negative edge sampling
            positive_mask = y_graph.y.squeeze() == 0
            positive_edges = y_graph.edge_index[:, positive_mask]
            negative_edges = negative_sampling(
                edge_index=y_graph.edge_index,
                num_nodes=y_graph.num_nodes,
                num_neg_samples=max(positive_edges.size(1), 20)
            )

            edge_pairs = torch.cat([positive_edges, negative_edges], dim=1)
            labels = torch.cat([
                torch.ones(positive_edges.size(1)),
                torch.zeros(negative_edges.size(1))
            ])

            # Grab scores and calculate metrics
            scores = self(x_graphs, edge_pairs)
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())
            acc = self.binary_acc(scores, labels.int())
            
            total_loss += loss
            total_acc += acc

        avg_loss = total_loss / len(num_windows)
        avg_acc = total_acc / len(num_windows)
        
        # Logging
        self.log("train_loss", avg_loss, on_epoch=True)
        self.log("train_acc", avg_acc, prog_bar=True, on_epoch=True)

        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        x_graphs, y_graph = batch

        # Sometimes, validation could have redteam activity, so we check on that as well
        redteam_mask = y_graph.y.squeeze() == 1
        red_team_edges = y_graph.edge_index[:, redteam_mask]
        
        if red_team_edges.size(1) > 0:
            red_team_labels = torch.zeros(red_team_edges.size(1), device=self.device)
            
            red_team_pred = self(x_graphs, red_team_edges)
            red_team_acc = self.binary_acc(red_team_pred, red_team_labels)
            
            self.log("val_redteam", red_team_acc)

        # Positive and negative edge sampling
        positive_mask = y_graph.y.squeeze() == 0
        positive_edges = y_graph.edge_index[:, positive_mask]
        negative_edges = negative_sampling(
            edge_index=y_graph.edge_index,
            num_nodes=y_graph.num_nodes,
            num_neg_samples=positive_edges.size(1)
        )

        edge_pairs = torch.cat([positive_edges, negative_edges], dim=1).to(self.device)
        labels = torch.cat([
            torch.ones(positive_edges.size(1)),
            torch.zeros(negative_edges.size(1))
        ])
        randomize_edges = torch.randperm(edge_pairs.size(1))
        edge_pairs = edge_pairs[:, randomize_edges]
        labels = labels[randomize_edges]

        # Grab scores and calculate metrics
        scores = self(x_graphs, edge_pairs)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float().to(self.device))
        acc = self.binary_acc(scores, labels.int().to(self.device))

        # Logging
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

        return loss
    
    def test_step(self, batch: tuple[list[Data], Data], batch_idx):
        """Validates the model on one batch of temporal graphs."""
        x_graphs, y_graph = batch
        
        # For testing, we also need to see if red team activities will be detected
        redteam_mask = y_graph.y.squeeze() == 1
        red_team_edges = y_graph.edge_index[:, redteam_mask]
        
        self.log("test_size", red_team_edges.size(1))
        if red_team_edges.size(1) > 0:
            red_team_labels = torch.zeros(red_team_edges.size(1))
            
            red_team_pred = self(x_graphs, red_team_edges)
            red_team_acc = self.binary_acc(red_team_pred, red_team_labels)
            
            self.log("test_redteam", red_team_acc)

        # Positive and negative edge sampling
        positive_mask = y_graph.y.squeeze() == 0
        positive_edges = y_graph.edge_index[:, positive_mask]
        negative_edges = negative_sampling(
            edge_index=y_graph.edge_index,
            num_nodes=y_graph.num_nodes,
            num_neg_samples=positive_edges.size(1)
        )

        edge_pairs = torch.cat([positive_edges, negative_edges], dim=1)
        labels = torch.cat([
            torch.ones(positive_edges.size(1)),
            torch.zeros(negative_edges.size(1))
        ])
        randomize_edges = torch.randperm(edge_pairs.size(1))
        edge_pairs = edge_pairs[:, randomize_edges]
        labels = labels[randomize_edges]

        # Grab scores and calculate metrics
        scores = self(x_graphs, edge_pairs)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        acc = self.binary_acc(scores, labels.int())

        # Logging
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
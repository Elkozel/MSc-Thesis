import torch
import warnings
import torchmetrics
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from typing import List, Literal, Union
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData


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

    def forward(self, h_src, h_dst):
        h_concat = torch.cat([h_src, h_dst], dim=-1)
        return self.mlp(h_concat)  # No softmax; use CrossEntropyLoss

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
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels * 3

        self.gnn = GNNEncoder(in_channels, hidden_channels, dropout_rate, edge_dim)
        self.rnn = RNNEncoder(hidden_channels, rnn_num_layers)
        self.link_pred = LinkPredictor(hidden_channels, 1)
        self.link_classifier = LinkTypeClassifier(hidden_channels, out_classes)

        self.link_predict_acc = torchmetrics.Accuracy(task="binary", threshold=binary_threshold)
        self.link_predict_auc = torchmetrics.AUROC(task="binary")

        self.link_class_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_classes)
        self.link_class_auc = torchmetrics.AUROC(task="multiclass", num_classes=out_classes)

        self.mal_count = torchmetrics.MeanMetric()
        self.model_loss = torchmetrics.MeanMetric()

        self.rnn_window_size = rnn_window_size
        self.model_name = model_name
        self.binary_threshold = binary_threshold
        self.negative_edge_sampling_min = negative_edge_sampling_min
        self.pred_alpha = pred_alpha

        self.save_hyperparameters()

    def forward(self, graph_sequence: Union[list[Data], torch.Tensor], edge_pairs):
        # Generate the features at each timestamp if not already computed
        # shape (timestamp, nodes, features)
        if isinstance(graph_sequence, torch.Tensor):
            graph_features = graph_sequence
        else:
            graph_features = [self.gnn(data.x, data.edge_index, data.edge_attr) for data in graph_sequence]
            graph_features = torch.stack(graph_features)

        # Switch the order, so that each node is treated as a batch
        # shape (nodes, timestamp, features)
        graph_features = graph_features.permute(1, 0, 2)
        rnn_output, _ = self.rnn(graph_features)

        # Get the embeddings at time t+1
        # shape (nodes, timestamp (t+1), features)
        predicted_embeddings = rnn_output[:, -1, :]  # use last time step

        # Generate the scores
        src = edge_pairs[0]
        dst = edge_pairs[1]
        link_pred_scores = self.link_pred(predicted_embeddings[src], predicted_embeddings[dst])  # shape [num_edges]
        link_class_logits = self.link_classifier(predicted_embeddings[src], predicted_embeddings[dst]) # shape [num_edges]

        return link_pred_scores, link_class_logits

    def precompute_features(self, batch: Batch) -> torch.Tensor:
        # Run GNN in a batched fashion
        x_out = self.gnn(batch.x, batch.edge_index, batch.edge_attr, batch=batch.batch)  # [T*N, F]

        # Recover shape: [T, N, F]
        num_snapshots = batch.num_graphs
        num_nodes_per_snapshot = batch.x.size(0) // num_snapshots
        feature_dim = x_out.size(1)

        return x_out.view(num_snapshots, num_nodes_per_snapshot, feature_dim)
        
    def run_trough_batch(self, batch: Batch, step: Literal['train', 'val', 'test']):
        window = self.rnn_window_size

        # Step 1: GNN (batched)
        gnn_features = self.precompute_features(batch)  # [T, N, F]
        T, N, F = gnn_features.shape
        num_windows = T - window

        # Step 2: Build RNN input
        # Result: [num_windows, window, N, F]
        x_windows = torch.stack([
            gnn_features[i:i + window] for i in range(num_windows)
        ], dim=0)

        # RNN input shape: [batch=N*num_windows, window, F]
        x_windows = x_windows.permute(2, 0, 1, 3).reshape(N * num_windows, window, F)

        # Step 3: RNN forward (batch_first=True)
        rnn_out, _ = self.rnn(x_windows)  # [N * num_windows, window, H]
        rnn_last = rnn_out[:, -1, :]      # [N * num_windows, H]

        # Reshape back to [num_windows, N, H]
        rnn_last = rnn_last.view(N, num_windows, -1).permute(1, 0, 2)  # [num_windows, N, H]

        # Step 4: Prepare decoding
        graph_list = batch.to_data_list()[window:]  # list of graphs to predict
        all_pos_edges = []
        all_neg_edges = []
        all_labels = []
        all_edge_labels = []
        all_node_reprs = []

        for i, y_graph in enumerate(graph_list):  # i ∈ [0, num_windows-1]
            pos_edges = y_graph.edge_index
            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=y_graph.num_nodes,
                num_neg_samples=max(pos_edges.size(1), self.negative_edge_sampling_min)
            )
            edge_pairs = torch.cat([pos_edges, neg_edges], dim=1)
            labels = torch.cat([
                torch.ones(pos_edges.size(1)),
                torch.zeros(neg_edges.size(1))
            ])

            all_pos_edges.append(pos_edges)
            all_neg_edges.append(neg_edges)
            all_labels.append(labels)
            all_edge_labels.append(y_graph.y)
            all_node_reprs.append(rnn_last[i])  # [N, H]

        # Step 5: Batch everything
        all_edge_pairs = []
        all_logits_labels = []
        all_class_targets = []
        all_class_logits = []

        total_loss = 0.0
        for i in range(num_windows):
            node_repr = all_node_reprs[i]                  # [N, H]
            edge_pairs = torch.cat([all_pos_edges[i], all_neg_edges[i]], dim=1)  # [2, E]
            labels = all_labels[i]                         # [E]
            edge_labels = all_edge_labels[i]               # [E_pos]

            link_pred, link_class = self.forward(node_repr, edge_pairs)  # [E], [E, C]

            pred_loss = F.binary_cross_entropy_with_logits(link_pred, labels.float())

            if edge_labels.numel() > 0:
                class_logits = link_class[labels.bool()]
                class_loss = F.cross_entropy(class_logits, edge_labels.long())

                self.link_class_acc(class_logits, edge_labels.int())
                if edge_labels.unique().numel() > 1:
                    self.link_class_auc(class_logits, edge_labels.int())

                self.mal_count.update(edge_labels.count_nonzero() / edge_labels.size(0))
            else:
                class_loss = torch.tensor(0.0)

            loss = pred_loss + self.pred_alpha * class_loss
            total_loss += loss

            # Metrics
            self.model_loss.update(loss)
            self.link_predict_acc(link_pred, labels.int())
            if labels.unique().numel() > 1:
                self.link_predict_auc(link_pred, labels.int())

        avg_loss = total_loss / num_windows

        return {
            "loss": avg_loss,
            "pred_acc": self.link_predict_acc.compute(),
            "pred_auc": self.link_predict_auc.compute() if self.link_predict_auc.update_count > 0 else 0,
            "class_acc": self.link_class_acc.compute(),
            "class_auc": self.link_class_auc.compute() if self.link_class_auc.update_count > 0 else 0,
            "mal_count": self.mal_count.compute()
        }


    def training_step(self, batch: Batch, batch_idx):
        """Trains the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "train"
        
        if num_windows < 1:
            warnings.warn(f"Training batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        results = self.run_trough_batch(batch, num_windows, step)
        
        # Logging
        for metric, value in results.items():
            self.log(f"{step}_{metric}", value, batch_size=batch.num_graphs)

        return results["loss"]
    
    def validation_step(self, batch: Batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "val"
        
        if num_windows < 1:
            warnings.warn(f"Validation batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        results = self.run_trough_batch(batch, num_windows, step)
        
        # Logging
        for metric, value in results.items():
            self.log(f"{step}_{metric}", value, batch_size=batch.num_graphs)

        return results["loss"]
    
    def test_step(self, batch: Batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "test"
        
        if num_windows < 1:
            warnings.warn(f"Testing batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        results = self.run_trough_batch(batch, num_windows, step)
        
        # Logging
        for metric, value in results.items():
            self.log(f"{step}_{metric}", value, batch_size=batch.num_graphs)

        return results["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
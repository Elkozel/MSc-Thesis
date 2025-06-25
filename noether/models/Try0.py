import warnings
import torch
import torchmetrics
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from typing import Literal, Union
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data, Batch
import torchmetrics.classification

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
        hidden_channels,
        dropout_rate,
        out_classes = 1,
        rnn_window_size = 30,
        rnn_num_layers = 1,
        binary_threshold = 0.5,
        negative_edge_sampling_min = 20,
        pred_alpha = 0.8,
        model_name="Try0"
    ):
        super().__init__()


        self.gnn = GNNEncoder(in_channels, hidden_channels)
        self.link_pred = MLPDecoder(hidden_channels, 1)
        self.link_classifier = LinkTypeClassifier(hidden_channels, out_classes)

        self.link_predict_acc = torchmetrics.classification.BinaryAccuracy(threshold=binary_threshold)
        self.link_class_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_classes)

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
            graph_features = [self.gnn(data.x, data.edge_index) for data in graph_sequence]
            graph_features = torch.stack(graph_features)

        # Get the embeddings at time t+1
        # shape (nodes, timestamp (t+1), features)
        predicted_embeddings = graph_features[:, -1, :]  # use last time step

        # Generate the scores
        src = edge_pairs[0]
        dst = edge_pairs[1]
        link_pred_scores = self.link_pred(predicted_embeddings[src], predicted_embeddings[dst])  # shape [num_edges]
        link_class_logits = self.link_classifier(predicted_embeddings[src], predicted_embeddings[dst]) # shape [num_edges]

        return link_pred_scores, link_class_logits

    def run_trough_batch(self, batch: Batch, num_windows: int, step: Literal['train', 'validation', 'test']):
        total_loss = torch.tensor(0.0).to(self.device)
        total_pred_acc = torch.tensor(0.0).to(self.device)
        total_class_acc = torch.tensor(0.0).to(self.device)
        features = [self.gnn(data.x, data.edge_index, data.edge_attr) for data in batch.to_data_list()]
        features = torch.stack(features)
            
        for i in range(num_windows):
            to_idx = i + self.rnn_window_size
            x_features = features[i:to_idx]
            y_graph = batch.get_example(to_idx)
            
            assert y_graph.edge_index is not None
            assert isinstance(y_graph.y, torch.Tensor)

            # Positive and negative edge sampling
            positive_edges = y_graph.edge_index
            negative_edges = negative_sampling(
                edge_index=y_graph.edge_index,
                num_nodes=y_graph.num_nodes,
                num_neg_samples=max(positive_edges.size(1), self.negative_edge_sampling_min)
            )
            edge_labels = y_graph.y

            edge_pairs = torch.cat([positive_edges, negative_edges], dim=1)
            labels = torch.cat([
                torch.ones(positive_edges.size(1)),
                torch.zeros(negative_edges.size(1))
            ]).to(self.device)

            # Grab scores
            link_pred, link_class = self(x_features, edge_pairs)

            # Calculate the loss for link prediction
            pred_loss = F.binary_cross_entropy_with_logits(link_pred, labels.float())

            # Mask the score
            link_class = link_class[labels.bool()]
            if positive_edges.any():
                class_loss = F.cross_entropy(link_class, edge_labels.long())
                class_acc = self.link_class_acc(link_class, edge_labels.int())
            else:
                class_loss = torch.tensor(0.0, device=self.device)
                class_acc = torch.tensor(0.0, device=self.device)

            loss = pred_loss + self.pred_alpha * class_loss
            pred_acc = self.link_predict_acc(link_pred, labels.int())
            
            
            total_loss += loss
            total_pred_acc += pred_acc
            total_class_acc += class_acc

        avg_loss = total_loss / num_windows
        avg_pred_acc = total_pred_acc / num_windows
        avg_class_acc = total_class_acc / num_windows

        return {
            "avg_loss": avg_loss,
            "avg_pred_acc": avg_pred_acc,
            "avg_class_acc": avg_class_acc
        }

    def training_step(self, batch: Batch, batch_idx):
        """Trains the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        
        if num_windows < 1:
            warnings.warn(f"Training batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        results = self.run_trough_batch(batch, num_windows, "train")
        
        # Logging
        self.log("train_loss", results["avg_loss"], batch_size=batch.num_graphs, on_epoch=True)
        self.log("train_pred_acc", results["avg_pred_acc"], batch_size=batch.num_graphs, prog_bar=True, on_epoch=True)
        self.log("train_class_acc", results["avg_class_acc"], batch_size=batch.num_graphs, prog_bar=True, on_epoch=True)

        return results["avg_loss"]
    
    def validation_step(self, batch: Batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        
        if num_windows < 1:
            warnings.warn(f"Validation batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        results = self.run_trough_batch(batch, num_windows, "validation")
        
        # Logging
        self.log("val_loss", results["avg_loss"], batch_size=batch.num_graphs, on_epoch=True)
        self.log("val_pred_acc", results["avg_pred_acc"], batch_size=batch.num_graphs, prog_bar=True, on_epoch=True)
        self.log("val_class_acc", results["avg_class_acc"], batch_size=batch.num_graphs, prog_bar=True, on_epoch=True)

        return results["avg_loss"]
    
    def test_step(self, batch: Batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        
        if num_windows < 1:
            warnings.warn(f"Testing batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        results = self.run_trough_batch(batch, num_windows, "test")
        
        # Logging
        self.log("test_loss", results["avg_loss"], batch_size=batch.num_graphs, on_epoch=True)
        self.log("test_pred_acc", results["avg_pred_acc"], batch_size=batch.num_graphs, prog_bar=True, on_epoch=True)
        self.log("test_class_acc", results["avg_class_acc"], batch_size=batch.num_graphs, prog_bar=True, on_epoch=True)

        return results["avg_loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
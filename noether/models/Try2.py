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

    def forward2(
        self,
        graph_sequence_batch: torch.Tensor,             # shape: [B, T, N, F]
        edge_pairs_batch: list[torch.Tensor]            # each: shape [2, E_i], one per batch element
    ):
        B, T, N, F = graph_sequence_batch.shape

        # Step 1: Reshape for RNN: [B * N, T, F]
        graph_features = graph_sequence_batch.permute(0, 2, 1, 3)  # [B, N, T, F]
        rnn_input = graph_features.reshape(B * N, T, F)

        # Step 2: Pass through RNN
        rnn_output, _ = self.rnn(rnn_input)  # [B * N, T, hidden_dim]
        rnn_output = rnn_output[:, -1, :]    # take output at last time step: [B * N, hidden_dim]

        # Step 3: Reshape back to [B, N, hidden_dim]
        hidden_dim = rnn_output.size(-1)
        node_embeddings = rnn_output.view(B, N, hidden_dim)  # [B, N, hidden]

        # Step 4: Compute edge predictions
        all_link_scores = []
        all_class_logits = []

        for b in range(B):
            edge_index = edge_pairs_batch[b]  # shape: [2, E]
            src = edge_index[0]  # shape: [E]
            dst = edge_index[1]

            h_src = node_embeddings[b, src]  # shape: [E, hidden]
            h_dst = node_embeddings[b, dst]

            link_score = self.link_pred(h_src, h_dst)         # shape: [E]
            link_logits = self.link_classifier(h_src, h_dst)  # shape: [E, C]

            all_link_scores.append(link_score)
            all_class_logits.append(link_logits)

        # Step 5: Concatenate results across batch
        link_pred_scores = torch.cat(all_link_scores, dim=0)     # shape: [sum_E]
        link_class_logits = torch.cat(all_class_logits, dim=0)   # shape: [sum_E, C]

        return link_pred_scores, link_class_logits


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

    def precompute_features(self, batch: Batch) -> Batch:
        features = self.gnn(batch.x, batch.edge_index, batch.edge_attr) # type: ignore
        batch.x = features # type: ignore
        return batch
    
    def run_trough_batch(self, batch: Batch, num_windows: int, step: Literal['train', 'val', 'test']):
        # 1. Precompute GNN features (modifies .x)
        batch = self.precompute_features(batch)
        graph_list = batch.to_data_list()

        # 2. Organize input windows for the RNN
        rnn_input_sequences = []   # List[List[Data]]
        edge_pairs_batch = []      # List[edge_index tensors]
        label_batch = []           # List[label tensors]

        for window_num in range(num_windows):
            input_start = window_num
            input_end = window_num + self.rnn_window_size
            target_idx = input_end  # predict on this graph

            x_sequence = graph_list[input_start:input_end]
            y_graph = graph_list[target_idx]

            # Get edges and labels for this target graph
            pos_edges = y_graph.edge_index
            num_neg = max(pos_edges.size(1), self.negative_edge_sampling_min)

            neg_edges = negative_sampling(
                edge_index=pos_edges,
                num_nodes=y_graph.num_nodes,
                num_neg_samples=num_neg
            )

            edge_pairs = torch.cat([pos_edges, neg_edges], dim=1)
            edge_labels = torch.cat([
                torch.ones(pos_edges.size(1)),
                torch.zeros(neg_edges.size(1))
            ]).to(self.device)

            rnn_input_sequences.append(x_sequence)
            edge_pairs_batch.append(edge_pairs.to(self.device))
            label_batch.append(edge_labels)

        # 3. Convert list of sequences → [B, T, N, F] tensor
        B = len(rnn_input_sequences)
        T = self.rnn_window_size
        N = graph_list[0].num_nodes
        Fs = graph_list[0].x.size(1)

        graph_tensor = torch.zeros(B, T, N, Fs, device=self.device)

        for b, seq in enumerate(rnn_input_sequences):
            for t, graph in enumerate(seq):
                graph_tensor[b, t] = graph.x  # assumes consistent node ordering

        # 4. Run model forward
        link_pred, link_class = self.forward2(graph_tensor, edge_pairs_batch)

        # 5. Compute losses
        all_labels = torch.cat(label_batch)  # shape: [total_edges]
        pred_loss = F.binary_cross_entropy_with_logits(link_pred, all_labels.float())

        # Classification: Only compute for positive edges
        positive_mask = all_labels.bool()
        link_class_pos = link_class[positive_mask]

        edge_type_labels = torch.cat([
            graph_list[i + self.rnn_window_size].y.to(self.device)
            for i in range(num_windows)
        ])

        if positive_mask.any():
            class_loss = F.cross_entropy(link_class_pos, edge_type_labels.long())
            class_acc = self.link_class_acc(link_class_pos, edge_type_labels.int())

            if edge_type_labels.unique().numel() > 1:
                class_auc = self.link_class_auc(link_class_pos, edge_type_labels.int())

            self.mal_count.update(edge_type_labels.count_nonzero() / edge_type_labels.size(0))
        else:
            class_loss = torch.tensor(0.0, device=self.device)

        # Combine losses
        loss = pred_loss + self.pred_alpha * class_loss
        self.model_loss.update(loss)

        # Metrics
        pred_acc = self.link_predict_acc(link_pred, all_labels.int())
        if all_labels.unique().numel() > 1:
            pred_auc = self.link_predict_auc(link_pred, all_labels.int())

        # 6. Return metrics
        return {
            "loss": self.model_loss.compute(),
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
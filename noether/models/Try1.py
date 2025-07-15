import warnings
import torch
import torchmetrics
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from typing import List, Literal, Union
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, batched_negative_sampling
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData

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
    
class RNNEncoder(nn.Module):
    def __init__(self, hidden_channels, num_layers=1):
        super().__init__()
        self.rnn = nn.GRU(hidden_channels, hidden_channels, num_layers=num_layers,  batch_first=True)

    def forward(self, x_seq):
        # x_seq shape: [num_nodes, sequence_length, hidden_channels]
        output, final_state = self.rnn(x_seq)
        return output, final_state  # optional depending on decoder
    
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
        model_name="Try1"
    ):
        super().__init__()


        self.gnn = GNNEncoder(in_channels, hidden_channels)
        self.rnn = RNNEncoder(hidden_channels)
        self.link_pred = MLPDecoder(hidden_channels, 1)
        self.link_classifier = LinkTypeClassifier(hidden_channels, out_classes)

        self.link_predict_acc = torchmetrics.Accuracy(task="binary", threshold=binary_threshold)
        self.link_predict_auc = torchmetrics.AUROC(task="binary")

        self.link_class_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_classes)
        self.link_class_auc = torchmetrics.AUROC(task="multiclass", num_classes=out_classes)

        self.mal_acc = torchmetrics.Accuracy(task="binary", threshold=0.5)
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
            graph_features = [self.gnn(data.x, data.edge_index) for data in graph_sequence]
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
    
    def run_trough_batch(self, data: Batch, stage: Literal['train', 'val', 'test']):

        ### GNN Encoding ###
        gnn_features = self.gnn(data.x, data.edge_index, data.edge_attr)
        data.x = gnn_features # [Host* , hidden_dim]

        # RNN
        # Important assumption, the X axis is the same on all graphs

        # Create a batch for each sliding window to be run trough the RNN (in one run)
        graph_list = data.to_data_list()
        num_windows = data.num_graphs - self.rnn_window_size
        num_nodes = graph_list[0].num_nodes
        assert num_nodes is not None

        rnn_batch = torch.stack([
            torch.stack([graph_list[t + i].x for i in range(self.rnn_window_size)])  # shape [window_size, Node, Feature]
            for t in range(num_windows)
        ])  # Final shape: [num_windows, window_size, Nnode, Feature]
        
        # Permute to [num_windows, num_nodes, window_size, features]
        rnn_batch = rnn_batch.permute(0, 2, 1, 3)  # [Batch, Node, Window, Feature]
        rnn_batch = rnn_batch.reshape(-1, self.rnn_window_size, rnn_batch.shape[-1])  # [Batch * Node, Window, Feature]
        rnn_output, _ = self.rnn(rnn_batch)  # [Batch * Node, Window, Feature]
        hidden_dim = rnn_output[:, -1, :].shape[-1]
        rnn_predicted_embeddings = rnn_output[:, -1, :].view(num_windows * num_nodes, hidden_dim) # [Batch * Host, Features]

        # Adjust for the offset, as each prediction is for the next timestep in the batch. This is done by 
        # increasing the ID of the node by the size of the nodes per graph (as all graphs have the same node size)
        start = self.rnn_window_size * num_nodes
        data.x[start:, :] = rnn_predicted_embeddings

        # Decoder
        # Create positive and negative edge indecies
        edge_sources = data.edge_index[0]
        edge_batch = data.batch[edge_sources]
        positive_edges = data.edge_index[:, edge_batch >= self.rnn_window_size] # Only grab edges from 

        if positive_edges.size(1) == 0:
            raise Exception("Positive edges are 0")
        
        negative_edges = batched_negative_sampling(
            positive_edges,
            data.batch
        )
        test_edges = torch.cat([
            positive_edges,
            negative_edges.int()
        ], dim=1)
        edge_label = torch.cat([
            torch.ones(positive_edges.size(1)),
            torch.zeros(negative_edges.size(1))
        ])
        edge_class = torch.cat([
            data.y[edge_batch >= self.rnn_window_size],
            torch.zeros(negative_edges.size(1))
        ])

        # Run the full decoding with the batch
        src = test_edges[0]
        dst = test_edges[1]
        predicted_embeddings = data.x
        link_pred = self.link_pred(predicted_embeddings[src], predicted_embeddings[dst])  # shape [num_edges]
        link_class = self.link_classifier(predicted_embeddings[src], predicted_embeddings[dst]) # shape [num_edges]

        # Calculate loss
        pred_loss = F.binary_cross_entropy_with_logits(link_pred, edge_label.float())
        class_loss = F.cross_entropy(link_class, edge_class.long())
        loss = pred_loss + self.pred_alpha * class_loss

        pred_acc = self.link_predict_acc(link_pred, edge_label.int())
        class_acc = self.link_class_acc(link_class, edge_class.int())


        pred_auc = self.link_predict_auc(link_pred, edge_label.int())
        class_auc = self.link_class_auc(link_class, edge_class.int()) if edge_class.sum() > 0 else 0

        mal_acc = self.mal_acc((torch.argmax(link_class, dim=1) > 0.5).int(), (edge_class > 0.5).int())
        mal_count = self.mal_count(edge_class.count_nonzero())

        # Calculate all metrics (also for the full batch)
        return {
            "loss": loss,
            "pred_acc": pred_acc,
            "pred_auc": pred_auc,
            "class_acc": class_acc,
            "class_auc": class_auc,
            "mal_acc": mal_acc,
            "mal_count": mal_count
        }
    
    def training_step(self, batch: Batch, batch_idx):
        """Trains the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "train"
        
        if num_windows < 1:
            warnings.warn(f"Training batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        
        try:
            results = self.run_trough_batch(batch, step)
        except:
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        
        # Logging
        for metric, value in results.items():
            self.log(f"{step}_{metric}", value, batch_size=batch.num_graphs, on_epoch=True, sync_dist=True)

        return results["loss"]
    
    def validation_step(self, batch: Batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "val"
        
        if num_windows < 1:
            warnings.warn(f"Validation batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        try:
            results = self.run_trough_batch(batch, step)
        except:
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        
        # Logging
        for metric, value in results.items():
            self.log(f"{step}_{metric}", value, batch_size=batch.num_graphs, on_epoch=True, sync_dist=True)

        return results["loss"]
    
    def test_step(self, batch: Batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "test"
        
        if num_windows < 1:
            warnings.warn(f"Testing batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
            
        try:
            results = self.run_trough_batch(batch, step)
        except:
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        
        # Logging
        for metric, value in results.items():
            self.log(f"{step}_{metric}", value, batch_size=batch.num_graphs, on_epoch=True, sync_dist=True)

        return results["loss"]


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
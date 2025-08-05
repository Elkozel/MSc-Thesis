import warnings
import torch
import torchmetrics
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from typing import Literal, Union
from torch_geometric.nn import GCNConv
from torch_geometric.utils import batched_negative_sampling
from torch_geometric.data import Data, Batch

class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_features):
        out = self.conv1(x, edge_index)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out, edge_index)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv3(out, edge_index)
        return out  # node embeddings
    
class RNNEncoder(nn.Module):
    def __init__(self, hidden_channels, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.GRU(hidden_channels, hidden_channels, num_layers=num_layers,  batch_first=True, dropout=dropout)

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
            nn.Linear(input_dim * 2 + 1, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes)
        )

    def forward(self, h_src, h_dst, link_pred = torch.Tensor([])):
        h_concat = torch.cat([h_src, h_dst, link_pred], dim=-1)
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
        pred_alpha = 1.2,
        link_pred_only = False,
        model_name="Try1"
    ):
        super().__init__()


        self.gnn = GNNEncoder(in_channels, hidden_channels, dropout=dropout_rate)
        self.rnn = RNNEncoder(hidden_channels, rnn_num_layers, dropout=dropout_rate)
        self.link_pred = MLPDecoder(hidden_channels, 1)
        self.link_classifier = LinkTypeClassifier(hidden_channels, out_classes)

        self.link_pred_metrics = nn.ModuleDict({
            "pred_acc": torchmetrics.Accuracy(task="binary", threshold=binary_threshold),
            "pred_auc": torchmetrics.AUROC(task="binary"),
            "pred_f1": torchmetrics.F1Score(task="binary"),
            "pred_ap": torchmetrics.AveragePrecision(task="binary")
        })
        self.link_pred_stat_scores = torchmetrics.StatScores(task="binary")

        self.link_class_metrics = nn.ModuleDict({
            "class_acc": torchmetrics.Accuracy(task="multiclass", num_classes=out_classes),
            "class_auc": torchmetrics.AUROC(task="multiclass", num_classes=out_classes),
            "class_f1": torchmetrics.F1Score(task="multiclass", num_classes=out_classes),
            "class_ap": torchmetrics.AveragePrecision(task="multiclass", num_classes=out_classes)
        })

        self.mal_metrics = nn.ModuleDict({
            "mal_acc": torchmetrics.Accuracy(task="binary", threshold=binary_threshold),
            "mal_auc": torchmetrics.AUROC(task="binary"),
            "mal_f1": torchmetrics.F1Score(task="binary"),
            "mal_ap": torchmetrics.AveragePrecision(task="binary"),
        })
        self.mal_stat_scores = torchmetrics.StatScores(task="binary")

        self.mal_only_metrics = nn.ModuleDict({
            "mal_only_acc": torchmetrics.Accuracy(task="multiclass", num_classes=out_classes),
            "mal_only_auc": torchmetrics.AUROC(task="multiclass", num_classes=out_classes),
            "mal_only_f1": torchmetrics.F1Score(task="multiclass", num_classes=out_classes),
            "mal_only_ap": torchmetrics.AveragePrecision(task="multiclass", num_classes=out_classes)
        })

        self.rnn_window_size = rnn_window_size
        self.out_classes = out_classes
        self.model_name = model_name
        self.binary_threshold = binary_threshold
        self.negative_edge_sampling_min = negative_edge_sampling_min
        self.pred_alpha = pred_alpha
        self.link_pred_only = link_pred_only

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
    
    def run_trough_batch(self, data: Batch):

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
        ])  # Final shape: [num_windows, window_size, Node, Feature]
        
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
        positive_edges = data.edge_index[:, edge_batch >= self.rnn_window_size] # Only grab edges after the starting window
                                                                                # Other edges are not relevant
        positive_edge_labels = data.y[edge_batch >= self.rnn_window_size]

        if positive_edges.size(1) == 0:
            raise NoPositiveEdgesException(f"Positive edges are {positive_edges.size(1)}")
        
        negative_edges = batched_negative_sampling(
            positive_edges,
            data.batch
        )
        test_edges = torch.cat([
            positive_edges,
            negative_edges.int()
        ], dim=1)
        edge_pred_labels = torch.cat([
            (positive_edge_labels == 0).int(),
            torch.zeros(negative_edges.size(1)).to(self.device)
        ]).to(self.device)
        edge_class_labels = torch.cat([
            positive_edge_labels,
            torch.zeros(negative_edges.size(1)).to(self.device)
        ]).to(self.device)

        # Run the full decoding with the batch
        src = test_edges[0]
        dst = test_edges[1]
        predicted_embeddings = data.x
        link_pred_logits = self.link_pred(predicted_embeddings[src], predicted_embeddings[dst])  # shape [num_edges]
        link_pred_probs = torch.sigmoid(link_pred_logits).unsqueeze(1)  # Shape: [num_edges, 1]

        link_class = self.link_classifier(predicted_embeddings[src], predicted_embeddings[dst], link_pred_probs)

        # Calculate loss
        pred_loss = F.binary_cross_entropy_with_logits(link_pred_logits, edge_pred_labels.float())
        weights = torch.Tensor([1] + [
            5 for _ in range(self.out_classes - 1)
        ]).to(self.device)
        class_loss = F.cross_entropy(link_class, edge_class_labels.long(), weight=weights)

        # Confidence loss
        # conf_weighted_class_loss = F.cross_entropy(link_class, edge_class_labels.long(), reduction='none')
        # conf_weighted_class_loss = (conf_weighted_class_loss * link_pred_probs.squeeze()).mean()
        
        if self.link_pred_only:
            loss = pred_loss
        else:
            loss = pred_loss * self.pred_alpha +  class_loss

        # Metrics from link prediction
        for name, metric in self.link_pred_metrics.items():
            metric(link_pred_logits, edge_pred_labels.int())

        # Metrics from link classification
        for name, metric in self.link_class_metrics.items():
            metric(link_class, edge_class_labels.int())

        # Metrics from overall malicious event prediction
        malicious_event_mask = (edge_class_labels > 0.5).int()

        for name, metric in self.mal_metrics.items():            
            metric((torch.argmax(link_class, dim=1) > 0.5).float(), malicious_event_mask)

        # Metrics from only malicious events        
        for name, metric in self.mal_only_metrics.items():
            if malicious_event_mask.sum() > 0:
                expanded_malicious_event_mask = malicious_event_mask.reshape(malicious_event_mask.size(0),1).expand(link_class.shape).bool()
                malicious_labels = torch.masked_select(edge_class_labels, malicious_event_mask.bool())
                malicious_predictions = torch.masked_select(link_class, expanded_malicious_event_mask).reshape(malicious_labels.size(0), -1)
                metric(malicious_predictions, malicious_labels.int())
        

        # Calculate all metrics (also for the full batch)
        metrics: dict[str, torchmetrics.Metric] = dict()
        metrics.update(self.link_pred_metrics) # type: ignore
        metrics.update(self.link_class_metrics) # type: ignore
        metrics.update(self.mal_metrics) # type: ignore
        metrics.update(self.mal_only_metrics) # type: ignore
        matric_results = {
            key: metric.compute() if metric.update_count > 0 else 0
            for key, metric in metrics.items()
        }
        for metric in metrics.values():
            metric.reset()

        tp_mal, fp_mal, tn_mal, fn_mal, sup_mal = self.mal_stat_scores((torch.argmax(link_class, dim=1) > 0.5).float(), malicious_event_mask)
        mal_stat_results = {
            "mal_tp": tp_mal.float(),
            "mal_fp": fp_mal.float(),
            "mal_tn": tn_mal.float(),
            "mal_fn": fn_mal.float(),
            "mal_sup": sup_mal.float(),
        }
        tp_pred, fp_pred, tn_pred, fn_pred, sup_pred = self.link_pred_stat_scores(link_pred_logits, edge_pred_labels.int())
        pred_stat_results = {
            "pred_tp": tp_pred.float(),
            "pred_fp": fp_pred.float(),
            "pred_tn": tn_pred.float(),
            "pred_fn": fn_pred.float(),
            "pred_sup": sup_pred.float(),
        }

        self.mal_stat_scores.reset()
        self.link_pred_stat_scores.reset()

        return {
            "loss": loss,
            "mal_count": data.y.count_nonzero().float(),
            "benign_count": data.y.size(0) - int(data.y.count_nonzero()),
            "edge_count": positive_edges.size(1),
            **mal_stat_results,
            **pred_stat_results,
            **matric_results
        }
    
    def training_step(self, batch: Batch, batch_idx):
        """Trains the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "train"
        
        if num_windows < 1:
            warnings.warn(f"Training batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True)
        
        try:
            results = self.run_trough_batch(batch)
        except NoPositiveEdgesException as e:
            return torch.tensor(0.0, requires_grad=True)
        
        # Logging
        self.log_dict({
            f"{step}_{metric}": value 
            for metric, value in results.items()
        }, logger=True, batch_size=batch.num_graphs, sync_dist=True)
        
        return results["loss"]
    
    def validation_step(self, batch: Batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "val"
        
        if num_windows < 1:
            warnings.warn(f"Validation batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True)
            
        try:
            results = self.run_trough_batch(batch)
        except NoPositiveEdgesException as e:
            return torch.tensor(0.0, requires_grad=True)
        
        # Logging
        self.log_dict({
            f"{step}_{metric}": value 
            for metric, value in results.items()
        }, logger=True, batch_size=batch.num_graphs, sync_dist=True)
        
        return results["loss"]
    
    def test_step(self, batch: Batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        num_windows = batch.num_graphs - (self.rnn_window_size + 1)
        step = "test"
        
        if num_windows < 1:
            warnings.warn(f"Testing batch ID: {batch_idx} (size {batch.num_graphs}) is not enough \
                          for a full window of {self.rnn_window_size}")
            return torch.tensor(0.0, requires_grad=True)
            
        try:
            results = self.run_trough_batch(batch)
        except NoPositiveEdgesException as e:
            return torch.tensor(0.0, requires_grad=True)
        
        # Logging
        self.log_dict({
            f"{step}_{metric}": value 
            for metric, value in results.items()
        }, logger=True, batch_size=batch.num_graphs, sync_dist=True)
        
        return results["loss"]


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    

class NoPositiveEdgesException(Exception):
    pass
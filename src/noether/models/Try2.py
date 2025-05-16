import torch
import torchmetrics
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
import torchmetrics.classification


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
    
class MLPDecoder(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        # src,dst: [num_edges, embed_dim]
        h = torch.cat([src, dst], dim=-1)    # → [num_edges, 2*embed_dim]
        return self.net(h).squeeze(-1)       # → [num_edges]

class LitFullModel(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, ):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = GNNEncoder(in_channels, hidden_channels)
        self.rnn = RNNEncoder(hidden_channels)
        self.decoder = MLPDecoder(hidden_channels)

        # define metrics
        self.train_acc = torchmetrics.classification.BinaryAccuracy(threshold=0.5)  
        self.val_acc   = torchmetrics.classification.BinaryAccuracy(threshold=0.5)


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

    def training_step(self, batch, batch_idx):
        """Trains the model on one batch of temporal graphs."""
        x_graphs, y_graph = batch


        # Positive and negative edge sampling
        positive_edges = y_graph.edge_index
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

        # Grab scores and calculate metrics
        scores = self(x_graphs, edge_pairs)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        acc = self.train_acc(scores, labels.int())

        # Logging
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validates the model on one batch of temporal graphs."""
        x_graphs, y_graph = batch

        # Positive and negative edge sampling
        positive_edges = y_graph.edge_index
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

        # Grab scores and calculate metrics
        scores = self(x_graphs, edge_pairs)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        acc = self.train_acc(scores, labels.int())

        # Logging
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        return loss
    
    def test_step(self, batch: tuple[list[Data], Data], batch_idx):
        """Validates the model on one batch of temporal graphs."""
        x_graphs, y_graph = batch

        # Positive and negative edge sampling
        positive_edges = y_graph.edge_index
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
        
        # For testing, we also need to see if red team activities will be detected
        # red_team_edges = y_graph.edge_index[y_graph.y == 1]
        
        # if red_team_edges.size > 0:            
        #     red_team_labels = torch.zeros(red_team_edges.size(1))
            
        #     red_team_pred = self(x_graphs, red_team_edges)
        #     red_team_acc = self.red_team_acc(red_team_pred, red_team_labels)
            
        #     self.log("redteam_acc", red_team_acc)

        # Grab scores and calculate metrics
        scores = self(x_graphs, edge_pairs)
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        acc = self.test_acc(scores, labels.int())

        # Logging
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
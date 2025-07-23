import torch
import warnings
import torchmetrics
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from typing import Literal, Union
from torch_geometric.nn import HEATConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import HeteroData, Batch
import torchmetrics.classification


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim, dropout_rate):
        super().__init__()
        self.conv1 = HEATConv(in_channels, hidden_channels*2, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_channels*2)
        self.conv2 = HEATConv(hidden_channels*2, hidden_channels*4, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim, dropout=dropout_rate)
        self.norm2 = nn.LayerNorm(hidden_channels*4)
        self.conv3 = HEATConv(hidden_channels*4, hidden_channels*3, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim, dropout=dropout_rate)
        self.norm3 = nn.LayerNorm(hidden_channels*3)
        self.conv4 = HEATConv(hidden_channels*3, hidden_channels*2, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim, dropout=dropout_rate)
        self.norm4 = nn.LayerNorm(hidden_channels*2)
        self.conv5 = HEATConv(hidden_channels*2, hidden_channels, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim, dropout=dropout_rate)
        self.norm5 = nn.LayerNorm(hidden_channels)
        
        self.dropout = dropout_rate

    
    def transform_hetero_data(self, data: HeteroData):
        # ========================
        #         Nodes          
        # ========================
        
        # Extract node features from the input data dictionary
        x_dict = data.x_dict
        
        # Determine the maximum number of features (columns) across all node types
        max_node_cols = max([t.size(1) for t in x_dict.values()])
        
        # Initialize lists to store node features and node types
        x = []
        node_type = []

        # Loop through each node type and pad the features to match the max_node_cols
        for type, t in enumerate(x_dict.values()):
            # Calculate padding width required for each node type to match max_node_cols
            pad_width = max_node_cols - t.size(1)
            
            # Pad node features with zeros (constant padding) to ensure all have the same column size
            t = F.pad(t, (0, pad_width), mode='constant', value=0.0)
            
            # Append padded node features to the x list
            x.append(t)
            
            # Create a tensor to denote the node type for each node in the current batch
            node_type.append(torch.full((t.size(0),), type))

        # Concatenate all node features into a single tensor
        node_type = torch.cat(node_type)
        x = torch.cat(x)

        # ========================
        #     Edge Attributes     
        # ========================
        
        # Extract edge attributes from the input data dictionary
        edge_attr_dict = data.edge_attr_dict
        
        # Determine the maximum number of edge features (columns) across all edge types
        max_edge_cols = max([t.size(1) for t in edge_attr_dict.values()])

        # Initialize lists to store edge attributes and edge types
        edge_attr = []
        edge_type = []

        # Loop through each edge type and pad the features to match the max_edge_cols
        for type, t in enumerate(edge_attr_dict.values()):
            # Calculate padding width required for each edge type to match max_edge_cols
            pad_width = max_edge_cols - t.size(1)
            
            # Pad edge features with zeros (constant padding) to ensure all have the same column size
            t = F.pad(t, (0, pad_width), mode='constant', value=0.0)
            
            # Append padded edge features to the edge_attr list
            edge_attr.append(t)
            
            # Create a tensor to denote the edge type for each edge in the current batch
            edge_type.append(torch.full((t.size(0),), type))

        # Concatenate all edge attributes into a single tensor
        edge_attr = torch.cat(edge_attr)
        edge_type = torch.cat(edge_type)

        # ========================
        #         Edges          
        # ========================
        
        # Extract edge indices from the input data dictionary
        edge_dict = data.edge_index_dict
        
        # Concatenate all edge indices (from different edge types) into a single tensor
        edge_index = torch.cat(list(edge_dict.values()), dim=1)
        
        # Return the final processed data: node features, edge indices, node types, edge types, and edge attributes
        return x, edge_index, node_type, edge_type, edge_attr

    def forward(self, data):
        x, edge_index, node_type, edge_type, edge_features = self.transform_hetero_data(data)

        # x: Tensor, edge_index: Union[Tensor, SparseTensor], node_type: Tensor, edge_type: Tensor, edge_attr: Optional[Tensor] = None
        out = self.conv1(x, edge_index, node_type, edge_type, edge_features)
        out = self.norm1(out)
        out = F.relu(out)
        # out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out, edge_index, node_type, edge_type, edge_features)
        out = self.norm2(out)
        out = F.relu(out)
        # out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv3(out, edge_index, node_type, edge_type, edge_features)
        out = self.norm3(out)
        out = F.relu(out)
        # out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv4(out, edge_index, node_type, edge_type, edge_features)
        out = self.norm4(out)
        out = F.relu(out)
        # out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv5(out, edge_index, node_type, edge_type, edge_features)
        out = self.norm5(out)
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
        num_node_types,
        num_edge_types,
        edge_type_emb_dim,
        edge_attr_emb_dim,
        hidden_channels = None,
        dropout_rate = 0.0,
        edge_dim = None,
        out_classes = 1,
        rnn_window_size = 30,
        rnn_num_layers = 1,
        binary_threshold = 0.5,
        negative_edge_sampling_min = 20,
        pred_alpha = 0.8,
        model_name="Try2H"
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels * 3

        self.gnn = GNNEncoder(in_channels, hidden_channels, num_node_types, num_edge_types, edge_type_emb_dim, edge_dim, edge_attr_emb_dim, dropout_rate)
        self.rnn = RNNEncoder(hidden_channels, rnn_num_layers)
        self.link_pred = LinkPredictor(hidden_channels, 1)
        self.link_classifier = LinkTypeClassifier(hidden_channels, out_classes)

        self.link_predict_acc = torchmetrics.classification.BinaryAccuracy(threshold=binary_threshold)
        self.link_class_acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_classes)

        self.rnn_window_size = rnn_window_size
        self.model_name = model_name
        self.binary_threshold = binary_threshold
        self.negative_edge_sampling_min = negative_edge_sampling_min
        self.pred_alpha = pred_alpha

        self.save_hyperparameters()


    def forward(self, graph_sequence: Union[list[HeteroData], torch.Tensor], edge_pairs):
        # Generate the features at each timestamp if not already computed
        # shape (timestamp, nodes, features)
        if isinstance(graph_sequence, torch.Tensor):
            graph_features = graph_sequence
        else:
            graph_features = [self.gnn(data) for data in graph_sequence]
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
    
    def run_trough_batch(self, batch: Batch, num_windows: int, step: Literal['train', 'validation', 'test']):
        total_loss = torch.tensor(0.0).to(self.device)
        total_pred_acc = torch.tensor(0.0).to(self.device)
        total_class_acc = torch.tensor(0.0).to(self.device)
        features = [self.gnn(data) for data in batch.to_data_list()] # type: ignore
        features = torch.stack(features)
            
        for i in range(num_windows):
            to_idx = i + self.rnn_window_size
            x_features = features[i:to_idx]
            y_nodes, y_edge_index, _, _, _ = self.gnn.transform_hetero_data(batch.get_example(to_idx)) # type: ignore
            y_labels = torch.cat([store.y for store in batch.get_example(to_idx).edge_stores])

            # Positive and negative edge sampling
            positive_edges = y_edge_index
            negative_edges = negative_sampling(
                edge_index=y_edge_index,
                num_nodes=y_nodes.size(dim=0),
                num_neg_samples=max(positive_edges.size(1), self.negative_edge_sampling_min)
            )
            edge_labels = y_labels

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
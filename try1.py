from comet_ml import start, login
from comet_ml.integration.pytorch import log_model, watch

import torch
import logging
import warnings
import torch.nn as nn
from tqdm import tqdm
from collections import deque
from models.Try1 import FullModel
from elasticsearch import Elasticsearch
from torch_geometric.utils import negative_sampling
from utils.elastic_datafetcher import LANLGraphFetcher

# -----------------------
# Experiment Configuration
# -----------------------

# Initialize a Comet experiment for logging metrics
experiment = start(
  api_key="CcXihX0g0UcdsprkmlKiD19Z8",
  project_name="Thesis",
  workspace="elkozel"
)

# Ensure user is logged in to Comet
login()

# Silence Elasticsearch transport warnings and PyTorch Geometric user warnings
logging.getLogger("elastic_transport.transport").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

# -----------------------
# Hyperparameters and Config
# -----------------------

train_from = 0
train_to = 86400
window_size = 20

# Default hyperparameter dictionary
default_hyper_params = {
    "in_channels": 3,
    "hidden_channels": 15,
    "learning_rate": 0.001,
    "num_epochs": 5,

    "seconds_bin": 60,
    "train_from": train_from,
    "train_to": train_to,
    "valid_from": 86400,
    "valid_to": 99360,
    "test_from": 99360,
    "test_to": 112320,
    "window_size": window_size,
    "steps": train_to - train_from - window_size,
    "batch_size": window_size,
}

def main(es_url="http://localhost:9200", hyper_params=default_hyper_params):
    """
    Main training loop for the GNN model using temporal LANL graph data.
    Includes training, validation, and testing stages with Comet logging.
    """
    experiment.log_parameters(hyper_params)  # Log all hyperparameters

    # -----------------------
    # Model Initialization
    # -----------------------

    model = FullModel(hyper_params["in_channels"], hyper_params["hidden_channels"])
    watch(model)  # Track model gradients and parameters with Comet

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    # -----------------------
    # Training Function
    # -----------------------
    def train(train_batch):
        """Trains the model on one batch of temporal graphs."""
        optimizer.zero_grad()

        x_graphs = list(train_batch)[:-1]  # Context graphs
        y_graph = list(train_batch)[-1]    # Target graph

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

        scores = model(x_graphs, edge_pairs)
        loss = criterion(scores, labels.float())
        loss.backward()
        optimizer.step()

        return loss, labels.size(0)

    # -----------------------
    # Evaluation Function
    # -----------------------
    def test(test_data):
        """Evaluates the model on one batch of temporal graphs."""
        x_graphs = list(test_data)[:-1]
        y_graph = list(test_data)[-1]

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

        scores = model(x_graphs, edge_pairs)
        loss = criterion(scores, labels.float())

        return loss, labels.size(0)

    # -----------------------
    # Data Fetching
    # -----------------------

    es = Elasticsearch(es_url)

    train_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                                   hyper_params["train_from"], hyper_params["train_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"])

    valid_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                         hyper_params["valid_from"], hyper_params["valid_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"])

    test_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                         hyper_params["test_from"], hyper_params["test_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"])

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(hyper_params["num_epochs"]):
        experiment.log_current_epoch(epoch)

        total_trainig_loss = 0
        trainig_samples = 1
        buffer = deque(maxlen=hyper_params["window_size"])

        with experiment.train():
            for graph_sec in tqdm(train_data, f"Training (Epoch {epoch + 1})"):
                buffer.append(graph_sec)
                if len(buffer) != hyper_params["window_size"]:
                    continue
                loss, num_edges = train(buffer)
                total_trainig_loss += loss.item() * num_edges
                trainig_samples += num_edges

        total_validation_loss = 0
        validation_samples = 1
        buffer = deque(maxlen=hyper_params["window_size"])

        with experiment.validate():
            model.eval()
            with torch.no_grad():
                for graph_sec in tqdm(valid_data, f"Validating (Epoch {epoch + 1})"):
                    buffer.append(graph_sec)
                    if len(buffer) != hyper_params["window_size"]:
                        continue
                    loss, num_edges = test(buffer)
                    total_validation_loss += loss.item() * num_edges
                    validation_samples += num_edges

        # Log metrics
        avg_train_loss = total_trainig_loss / trainig_samples
        experiment.log_metric("avg_train_loss", avg_train_loss)
        avg_validation_loss = total_validation_loss / validation_samples
        experiment.log_metric("avg_valid_loss", avg_validation_loss)

        print(f"Epoch {epoch + 1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Valid Loss: {avg_validation_loss:.4f} ")

    # -----------------------
    # Final Testing
    # -----------------------
    with experiment.test():
        with torch.no_grad():
            total_testing_loss = 0
            testing_samples = 1
            buffer = deque(maxlen=hyper_params["window_size"])

            for graph_sec in tqdm(test_data, f"Testing"):
                buffer.append(graph_sec)
                if len(buffer) != hyper_params["window_size"]:
                    continue
                loss, num_edges = test(buffer)
                total_testing_loss += loss.item() * num_edges
                testing_samples += num_edges

            avg_test_loss = total_testing_loss / testing_samples
            experiment.log_metric("test_loss", avg_test_loss)
            print(f"Test results: {avg_test_loss:.4f}")

    # Save model to Comet
    log_model(experiment, model=model, model_name="TheModel")

# Run main if script is executed
main()

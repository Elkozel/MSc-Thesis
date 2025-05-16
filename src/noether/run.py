import torch
from torch_geometric.utils import degree
from pytorch_lightning.loggers import CometLogger
from torch_geometric.data import Data
from transforms import RemoveSelfLoops, RemoveDuplicatedEdges
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import lightning as L
from models.Try1 import LitFullModel
from utils.window_dataloader import MovingWindowDataloader
from elasticsearch import Elasticsearch
from utils.elastic_datafetcher import LANLGraphFetcher

# -----------------------
# Experiment Configuration
# -----------------------

# Initialize a Comet experiment for logging metrics
comet_logger = CometLogger(
    api_key="CcXihX0g0UcdsprkmlKiD19Z8",
    project="Thesis",
    workspace="elkozel"
)

# -----------------------
# Constants
# -----------------------
TIME_FIRST_MALICIOUS = 1_758_226

# -----------------------
# Hyperparameters and Config
# -----------------------

default_hyper_params = {
    "in_channels": 3,
    "hidden_channels": 15,
    "learning_rate": 0.001,
    "num_epochs": 5,

    "seconds_bin": 60, # How many seconds of data to be put together into a single graph
    "train_from": 0,
    "train_to": 967_680,
    "valid_from": 967_681,
    "valid_to": 1_206_600,
    "test_from": 1_758_000,
    "test_to": 2_000_000,
    "window_size": 20, # How many past data's should be given to the RNN
}
default_hyper_params = {
    "in_channels": 3,
    "hidden_channels": 15,
    "learning_rate": 0.001,
    "num_epochs": 1,

    "seconds_bin": 60, # How many seconds of data to be put together into a single graph
    "train_from": 0,
    "train_to": 5_000,
    "valid_from": 2_301_630,
    "valid_to": 2_302_630,
    "test_from": 1_847_858,
    "test_to": 1_852_419,
    "window_size": 20, # How many past data's should be given to the RNN
}
comet_logger.log_hyperparams(default_hyper_params)

es_url="http://localhost:9200"
hyper_params=default_hyper_params

es = Elasticsearch(es_url)
def transform(data: Data) -> Data:
    rem_dub = RemoveDuplicatedEdges(key=["edge_attr", "edge_weight", "time", "y"])
    data = rem_dub(data)
    rem_self_loop = RemoveSelfLoops(attr=["edge_attr", "edge_weight", "time", "y"])
    data = rem_self_loop(data)
    
    in_degree = degree(data.edge_index[0], data.num_nodes)
    out_degree = degree(data.edge_index[1], data.num_nodes)
    data.x = torch.cat([
        data.x,
        in_degree.unsqueeze(1),
        out_degree.unsqueeze(1)
    ], dim=1)
    return data
    
train_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                                hyper_params["train_from"], 
                                hyper_params["train_to"], 
                                prefetch=True, 
                                seconds_bin=hyper_params["seconds_bin"],
                                transform=transform)
valid_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                      hyper_params["valid_from"], hyper_params["valid_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"], transform=transform)
test_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                      hyper_params["test_from"], hyper_params["test_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"], transform=transform)

model = LitFullModel(hyper_params["in_channels"], hyper_params["hidden_channels"])

early_stop_callback = EarlyStopping(monitor="val_loss", mode="max")
    
trainer = L.Trainer(max_epochs=hyper_params["num_epochs"], logger=comet_logger, callbacks=[early_stop_callback])
trainer.fit(model=model, train_dataloaders=MovingWindowDataloader(train_data, hyper_params["window_size"]),
              val_dataloaders=MovingWindowDataloader(valid_data, hyper_params["window_size"]))
trainer.test(model=model, dataloaders=MovingWindowDataloader(test_data, hyper_params["window_size"]))
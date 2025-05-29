from sqlalchemy import create_engine
import torch
from torch_geometric.utils import degree
from pytorch_lightning.loggers import CometLogger
from torch_geometric.data import Data
from dataloaders.UWF22 import UWFGraphMaker
from transforms import AddInOutDegree, RemoveSelfLoops, RemoveDuplicatedEdges
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import sys

import lightning as L
from dataloaders.window_dataloader import MovingWindowDataloader
from elasticsearch import Elasticsearch
from dataloaders.LANL import LANLGraphLoader

from models.Try1_gat import LitFullModel as GAT
from models.Try2 import LitFullModel as Try2
from datasets.UFW22 import UFW22L

datasets = {
    "UFW22": UFW22L
}
dataset_name = sys.argv[2] if len(sys.argv) > 2 else "UFW22"
transforms = [
    RemoveDuplicatedEdges(key=["edge_attr", "edge_weight", "time", "y"]),
    RemoveSelfLoops(attr=["edge_attr", "edge_weight", "time", "y"]),
    AddInOutDegree()
]
dataset = datasets[dataset_name]()

models = {
    "gat": GAT,
    "try2": Try2,
}
model_name = sys.argv[1] if len(sys.argv) > 1 else "try2"
model = models[model_name](
    in_channels = dataset.node_features, 
    hiddne_channels = dataset.node_features * 3,
    dropout_rate = 0.0)

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
lanl_training = {
    "model_name": model_name,
    "dataset": dataset_name,
    "in_channels": 3,
    "hidden_channels": 15,
    "learning_rate": 0.001,
    "dropout_rate": 0.02,
    "num_epochs": 8,

    "seconds_bin": 20, # How many seconds of data to be put together into a single graph
    "train_from": 0,
    "train_to": 150_880,
    "valid_from": 725_488,
    "valid_to": 870_000,
    "test_from": 1_066_394,
    "test_to": 1_363_492,
    "window_size": 20, # How many past data's should be given to the RNN
}
default_hyper_params = {
    "model_name": model_name,
    "dataset": dataset_name,
    "in_channels": 3,
    "hidden_channels": 15,
    "learning_rate": 0.001,
    "dropout_rate": 0.02,
    "num_epochs": 8,

    "seconds_bin": 20, # How many seconds of data to be put together into a single graph
    "train_from": 0,
    "train_to": 150_880,
    "valid_from": 725_488,
    "valid_to": 870_000,
    "test_from": 1_066_394,
    "test_to": 1_363_492,
    "window_size": 20, # How many past data's should be given to the RNN
}
# default_hyper_params = {
#     "model_name": model_name,
#     "in_channels": 3,
#     "hidden_channels": 15,
#     "learning_rate": 0.001,
#     "dropout_rate": 0.02,
#     "num_epochs": 2,

#     "seconds_bin": 60, # How many seconds of data to be put together into a single graph
#     "train_from": 0,
#     "train_to": 150_885,
#     "valid_from": 460_000,
#     "valid_to": 500_000,
#     "test_from": 734_000,
#     "test_to": 774_000,
#     "window_size": 20, # How many past data's should be given to the RNN
# }
comet_logger.log_hyperparams(default_hyper_params)

es_url="http://localhost:9200"
hyper_params=default_hyper_params

es = Elasticsearch(es_url)
def transform(data: Data) -> Data:
    rem_dub = RemoveDuplicatedEdges(key=["edge_attr", "edge_weight", "time", "y"])
    data = rem_dub(data)
    rem_self_loop = RemoveSelfLoops(attr=["edge_attr", "edge_weight", "time", "y"])
    data = rem_self_loop(data)
    
    in_degree = degree(data.edge_index[0], data.num_nodes).unsqueeze(1)
    out_degree = degree(data.edge_index[1], data.num_nodes).unsqueeze(1)
    data.x = torch.cat([
        data.x,
        in_degree,
        out_degree
    ], dim=1)
    return data
    
engine = create_engine('postgresql+psycopg://lanl:lanl@localhost/lanl', echo=False)
train_data = UWFGraphMaker(engine, hyper_params["train_from"], hyper_params["train_to"], hyper_params["seconds_bin"])

train_data = LANLGraphLoader(es, ("lanl-auth", "lanl-redteam"),
                                hyper_params["train_from"], 
                                hyper_params["train_to"], 
                                prefetch=True, 
                                seconds_bin=hyper_params["seconds_bin"],
                                transform=transform)
valid_data = LANLGraphLoader(es, ("lanl-auth", "lanl-redteam"),
                      hyper_params["valid_from"], hyper_params["valid_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"], transform=transform)
test_data = LANLGraphLoader(es, ("lanl-auth", "lanl-redteam"),
                      hyper_params["test_from"], hyper_params["test_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"], transform=transform)


print(f"Chosen model \n {model}")

early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", verbose=True, check_on_train_epoch_end=False)


trainer = L.Trainer(max_epochs=hyper_params["num_epochs"], logger=comet_logger, callbacks=[early_stop_callback], default_root_dir=f"models/{hyper_params['model_name']}/")
trainer.fit(model=model, train_dataloaders=MovingWindowDataloader(train_data, hyper_params["window_size"]),
              val_dataloaders=MovingWindowDataloader(valid_data, hyper_params["window_size"]))
trainer.test(model=model, dataloaders=MovingWindowDataloader(test_data, hyper_params["window_size"]))
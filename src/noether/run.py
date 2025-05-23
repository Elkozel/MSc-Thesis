import torch
from torch_geometric.utils import degree
from pytorch_lightning.loggers import CometLogger
from torch_geometric.data import Data
from transforms import RemoveSelfLoops, RemoveDuplicatedEdges
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import sys

import lightning as L
from utils.window_dataloader import MovingWindowDataloader
from elasticsearch import Elasticsearch
from dataloaders.LANL import LANLGraphLoader

from models.Try1_gat import LitFullModel as GAT
from models.Try2 import LitFullModel as Try2

models = {
    "gat": GAT,
    "try2": Try2,
}
model_name = sys.argv[1] if len(sys.argv) > 1 else "try2"
model_constr = models[model_name] 

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
    "model_name": model_name,
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

model = model_constr(hyper_params["in_channels"], hyper_params["hidden_channels"], hyper_params["dropout_rate"])

print(f"Chosen model \n {model}")

early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", verbose=True, check_on_train_epoch_end=False)


trainer = L.Trainer(max_epochs=hyper_params["num_epochs"], logger=comet_logger, callbacks=[early_stop_callback], default_root_dir=f"models/{hyper_params['model_name']}/")
trainer.fit(model=model, train_dataloaders=MovingWindowDataloader(train_data, hyper_params["window_size"]),
              val_dataloaders=MovingWindowDataloader(valid_data, hyper_params["window_size"]))
trainer.test(model=model, dataloaders=MovingWindowDataloader(test_data, hyper_params["window_size"]))
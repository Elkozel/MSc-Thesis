from pytorch_lightning.loggers import CometLogger

import lightning as L
from models.lit_Try1 import FullModel
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

    "seconds_bin": 60, # How many seconds of data to be put together into a single graph
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
comet_logger.log_hyperparams(default_hyper_params)

es_url="http://localhost:9200"
hyper_params=default_hyper_params

es = Elasticsearch(es_url)
train_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                                hyper_params["train_from"], hyper_params["train_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"])
valid_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                      hyper_params["valid_from"], hyper_params["valid_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"])
test_data = LANLGraphFetcher(es, ("lanl-auth", "lanl-redteam"),
                      hyper_params["test_from"], hyper_params["test_to"], prefetch=True, seconds_bin=hyper_params["seconds_bin"])

model = FullModel(hyper_params["in_channels"], hyper_params["hidden_channels"])

old_list = 
for s in MovingWindowDataloader(train_data, hyper_params["window_size"]):
    x,y = s
    for xs in x:
        print(xs)

    print("")


# trainer = L.Trainer(max_epochs=10, logger=comet_logger)
# trainer.fit(model=model, train_dataloaders=MovingWindowDataloader(train_data, hyper_params["window_size"]),
#               val_dataloaders=MovingWindowDataloader(valid_data, hyper_params["window_size"]))
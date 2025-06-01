import comet_ml
from pytorch_lightning.loggers import CometLogger
from transforms.AddInOutDegree import AddInOutDegree
from transforms.RemoveSelfLoops import RemoveSelfLoops
from transforms.RemoveDuplicatedEdges import RemoveDuplicatedEdges
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import sys

import lightning as L

from models.Try1 import LitFullModel as Try1
from models.Try2 import LitFullModel as Try2
from datasets.UFW22_local import UFW22L

if __name__ == '__main__':
    # Initialize a Comet experiment for logging metrics
    comet_logger = CometLogger(
        api_key="CcXihX0g0UcdsprkmlKiD19Z8",
        project="Thesis",
        workspace="elkozel"
    )

    datasets = {
        "UFW22": UFW22L
    }
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else "UFW22"
    transformations = [
        RemoveDuplicatedEdges(key=["edge_attr", "edge_weight", "time", "y"]),
        RemoveSelfLoops(attr=["edge_attr", "edge_weight", "time", "y"]),
        AddInOutDegree()
    ]
    DATASET_DIR = "D:\\Datasets\\UWF22"
    datamodule = datasets[dataset_name](DATASET_DIR)

    models = {
        "try1": Try1,
        "try2": Try2,
    }
    model_name = sys.argv[1] if len(sys.argv) > 1 else "try2"
    model = models[model_name](
        datamodule.node_features,
        datamodule.node_features * 3,
        dropout_rate = 0.0)

    early_stop_callback = EarlyStopping(monitor="val_loss", mode="min", verbose=True, check_on_train_epoch_end=False)
    trainer = L.Trainer(max_epochs = 50, profiler="simple", logger=comet_logger)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
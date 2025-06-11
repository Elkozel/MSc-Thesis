import comet_ml
import torch
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
from datasets.UFW22H_local import UFW22HL

if __name__ == '__main__':
    # Initialize a Comet experiment for logging metrics
    comet_logger = CometLogger(
        api_key="CcXihX0g0UcdsprkmlKiD19Z8",
        project="Thesis",
        workspace="elkozel"
    )

    transformations = [
        RemoveDuplicatedEdges(key=["edge_attr", "edge_weight", "time", "y"]),
        RemoveSelfLoops(attr=["edge_attr", "edge_weight", "time", "y"]),
        AddInOutDegree()
    ]
    DATASET_DIR = "/data/datasets/UWF22"
    datasets = {
        "UFW22": UFW22L(DATASET_DIR, transforms=transformations),
        "UFW22H": UFW22HL(DATASET_DIR, transforms=transformations)
    }
    dataset = datasets[sys.argv[2]] if len(sys.argv) > 2 else datasets["UFW22"]

    models = {
        "try1": Try1(
        dataset.node_features,
        dataset.node_features * 3,
        out_classes = dataset.num_classes,
        dropout_rate = 0.0
        ),
        "try2": Try2(
        dataset.node_features,
        out_classes = dataset.num_classes,
        pred_alpha = 1.1,
        edge_dim = dataset.edge_features
        ),
    }
    model = models[sys.argv[1]] if len(sys.argv) > 1 else models["try1"]
    
    # If we train on tensor cores as well
    torch.set_float32_matmul_precision('high')

    trainer = L.Trainer(max_epochs = 50,
                        logger=comet_logger,  # type: ignore
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", check_on_train_epoch_end=False)]) # type: ignore
    trainer.fit(model, dataset)
    trainer.test(model, dataset)
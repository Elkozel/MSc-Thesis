import comet_ml
from comet_ml.integration.pytorch import log_model
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
from models.Try2H import LitFullModel as Try2H
from datasets.UFW22_local import UFW22L
from datasets.UFW22H_local import UFW22HL

if __name__ == '__main__':
    # Initialize a Comet experiment for logging metrics
    comet_logger = CometLogger(
        api_key="CcXihX0g0UcdsprkmlKiD19Z8",
        project="Thesis",
        workspace="elkozel"
    )
    model = "try2"
    dataset = "UFW22"

    transformations = [
        RemoveDuplicatedEdges(key=["edge_attr", "edge_weight", "time", "y"]),
        RemoveSelfLoops(attr=["edge_attr", "edge_weight", "time", "y"]),
        # AddInOutDegree()
    ]
    DATASET_DIR = "/data/datasets/UWF22"

    if dataset == "UFW22":
        dataset = UFW22L(DATASET_DIR, transforms=transformations)
    elif dataset == "UFW22H":
        dataset = UFW22HL(DATASET_DIR, transforms=transformations)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented")

    if model == "try1":
        model = Try1(
        dataset.node_features,
        dataset.node_features * 3,
        out_classes = dataset.num_classes,
        dropout_rate = 0.0
        )
    elif model == "try2":
        model = Try2(
        dataset.node_features,
        out_classes = dataset.num_classes,
        pred_alpha = 1.1,
        edge_dim = dataset.edge_features
        )
    elif model == "try2h":
        model = Try2H(
        dataset.node_features,
        dataset.num_node_types, # type: ignore
        dataset.num_edge_types, # type: ignore
        dataset.edge_type_emb_dim, # type: ignore
        dataset.edge_attr_emb_dim, # type: ignore
        out_classes = dataset.num_classes,
        pred_alpha = 1.1,
        edge_dim = dataset.edge_features,
        )
    else:
        raise NotImplementedError(f"Model {model} is not implemented")

    log_model(
        experiment=comet_logger.experiment,
        model=model,
        model_name=model.model_name,
    )
    
    # If we train on tensor cores as well
    torch.set_float32_matmul_precision('high')

    trainer = L.Trainer(max_epochs = 50,
                        logger=comet_logger,  # type: ignore
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", check_on_train_epoch_end=False)]) # type: ignore
    trainer.fit(model, dataset)
    trainer.test(model, dataset)
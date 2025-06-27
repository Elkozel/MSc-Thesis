import argparse
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
from models.Try3 import LitFullModel as Try3
from datasets.UFW22_local import UFW22L
from datasets.UFW22H_local import UFW22HL
from datasets.LANL_local import LANLL


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a GNN model with Comet logging")
    
    parser.add_argument('--model', type=str, choices=['try1', 'try2', 'try2h', 'try3'], required=True,
                        help="Model type to use")
    parser.add_argument('--dataset', type=str, choices=['UFW22', 'UFW22h', 'LANL'], required=True,
                        help="Dataset to use")
    parser.add_argument('--max-epochs', type=int, default=50,
                        help="Maximum number of training epochs")
    
    return parser.parse_args()

def check_hetero(model_name: str, dataset_name: str):
    return model_name.endswith("h") ^ dataset_name.endswith("h")

def main():
    args = parse_args()
    # Initialize a Comet experiment for logging metrics
    comet_logger = CometLogger(
        api_key="CcXihX0g0UcdsprkmlKiD19Z8",
        project="Thesis",
        workspace="elkozel"
    )
    model = "try2"
    dataset = "LANL"

    transformations = [
        RemoveDuplicatedEdges(key=["edge_attr", "edge_weight", "time", "y"]),
        RemoveSelfLoops(attr=["edge_attr", "edge_weight", "time", "y"]),
        # AddInOutDegree()
    ]

    if args.dataset == "UFW22":
        dataset = UFW22L("/data/datasets/UWF22", transforms=transformations)
    elif args.dataset == "UFW22h":
        dataset = UFW22HL("/data/datasets/UWF22", transforms=transformations)
    elif args.dataset == "LANL":
        dataset = LANLL(
            "/data/datasets/LANL",
            download=True, 
            lanl_URL="https://csr.lanl.gov/data-fence/1750885650/Eao2ITLSwQl4pLAxzgE-vjOVk9Q=/cyber1/", 
            transforms=transformations)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented")

    if args.model == "try1":
        model = Try1(
        dataset.node_features,
        dataset.node_features * 3,
        out_classes = dataset.num_classes,
        dropout_rate = 0.0
        )
    elif args.model == "try2":
        model = Try2(
        dataset.node_features,
        out_classes = dataset.num_classes,
        pred_alpha = 1.1,
        edge_dim = dataset.edge_features
        )
    elif args.model == "try2h":
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
    elif args.model == "try3":
        model = Try3(
        dataset.node_features,
        out_classes = dataset.num_classes,
        pred_alpha = 1.1,
        edge_dim = dataset.edge_features
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

    trainer = L.Trainer(max_epochs = args.max_epochs,
                        logger=comet_logger,  # type: ignore
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", check_on_train_epoch_end=False)]) # type: ignore
    trainer.fit(model, dataset)
    trainer.test(model, dataset)
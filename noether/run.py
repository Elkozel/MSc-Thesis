import comet_ml

import os
import argparse
from comet_ml.integration.pytorch import log_model
import torch
from pytorch_lightning.loggers import CometLogger
from transforms.AddInOutDegree import AddInOutDegree
from transforms.RemoveSelfLoops import RemoveSelfLoops
from transforms.RemoveDuplicatedEdges import RemoveDuplicatedEdges
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import lightning as L

from models.Try0 import LitFullModel as Try0
from models.Try1 import LitFullModel as Try1
from models.Try2 import LitFullModel as Try2
from models.Try2H import LitFullModel as Try2H
from models.Try3 import LitFullModel as Try3
from datasets.UWF22H_local import UWF22HL
from datasets.UWF import UWF22L, UWF22FallL, UWF24L, UWF24FallL
from datasets.UWF_Shuffled import UWF22S
from datasets.LANL_local import LANLL

def parse_args():
    parser = argparse.ArgumentParser(description="Train and test a GNN model with Comet logging")
    
    parser.add_argument('--model', type=str, choices=["try0", 'try1', 'try2', 'try2h', 'try3'],
                        help="Model type to use", default="try2")
    parser.add_argument('--dataset', type=str, choices=["UWF22", "UWF22h", "UWF22Fall", "UWF24", "UWF24Fall", 'LANL'],
                        help="Dataset to use", default="LANL")
    parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction,
                        help="Whether the dataset should be shuffled", default=True)
    

    parser.add_argument('--max-epochs', type=int, default=50,
                        help="Maximum number of training epochs")
    parser.add_argument('--num-devices', default="auto",
                        help="The amount of devices used by the Trainer")
    parser.add_argument('--dataset-folder', type=str, default="/data/datasets",
                        help="Folder to store all datasets")
    
    return parser.parse_args()

def check_hetero(model_name: str, dataset_name: str):
    if model_name.endswith("h") ^ dataset_name.endswith("h"):
        raise Exception(f"Model {model_name} cannot be run with dataset {dataset_name} \
                    (one of them is for heterogeneous graphs)")

def main():
    args = parse_args()
    # Initialize a Comet experiment for logging metrics
    comet_logger = CometLogger(
        api_key="CcXihX0g0UcdsprkmlKiD19Z8",
        project="Thesis",
        workspace="elkozel"
    )
    check_hetero(args.model, args.dataset)

    transformations = [
        AddInOutDegree()
    ]

    if args.dataset == "UWF22":
        dataset = UWF22S(args.dataset_folder,
                         bin_size=5,
                         batch_size=60,
                         account_for_duration=False,
                         shuffle=True,
                         shuffle_every_time=True,
                         shuffle_type="day",
                         shuffle_bin_size=0.1,
                         transforms=transformations)
    elif args.dataset == "UWF22h":
        transformations = []
        dataset = UWF22HL(args.dataset_folder, 
                         bin_size=20,
                         batch_size=350,
                         transforms=transformations)
    elif args.dataset == "UWF22Fall":
        dataset = UWF22FallL(args.dataset_folder, 
                         bin_size=20,
                         batch_size=350,
                         transforms=transformations)
    elif args.dataset == "UWF24":
        dataset = UWF24L(args.dataset_folder, 
                         bin_size=20,
                         batch_size=350,
                         transforms=transformations)
    elif args.dataset == "UWF24Fall":
        dataset = UWF24FallL(args.dataset_folder, 
                         bin_size=20,
                         batch_size=350,
                         transforms=transformations)
    elif args.dataset == "LANL":
        transformations = [
            RemoveDuplicatedEdges(key=["edge_attr", "edge_weight", "time", "y"]),
            RemoveSelfLoops(attr=["edge_attr", "edge_weight", "time", "y"]),
            AddInOutDegree()
        ]
        dataset = LANLL(
            args.dataset_folder,
            bin_size=20,
            batch_size=100,
            transforms=transformations)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not implemented")

    if args.model == "try0":
        model = Try0(
        dataset.node_features + 2,
        dataset.node_features * 3,
        out_classes = dataset.num_classes,
        dropout_rate = 0.0
        )
    elif args.model == "try1":
        model = Try1(
        dataset.node_features + 2,
        dataset.node_features + 2 * 3,
        out_classes = dataset.num_classes,
        dropout_rate = 0.0
        )
    elif args.model == "try2":
        model = Try2(
        dataset.node_features + 2,
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
        raise NotImplementedError(f"Model {args.model} is not implemented")

    log_model(
        experiment=comet_logger.experiment,
        model=model,
        model_name=model.model_name,
    )
    
    # If we train on tensor cores as well
    torch.set_float32_matmul_precision('high')

    trainer = L.Trainer(max_epochs = args.max_epochs,
                        devices=args.num_devices,
                        logger=comet_logger,  # type: ignore
                        log_every_n_steps=1,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", check_on_train_epoch_end=False)]) # type: ignore
    trainer.fit(model, dataset)
    trainer.test(model, dataset)

if __name__ == "__main__":
    main()
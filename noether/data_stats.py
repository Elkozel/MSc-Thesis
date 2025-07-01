from datasets.LANL_local import LANLL
from ordered_set import OrderedSet
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import clear_output

dataset = LANLL("/data/datasets/LANL")

dataset.prepare_data()
dataset.setup(stage="fit")

def LANL_unique_v1(dataset):
    # Get dataset columns
    columns = list(next(iter(dataset.generate_bins())).columns)

    # Generate keymaps
    exclude_columns = [
        "time",
        "source computer",
        "destination computer",
        "source user@domain",
        "destination user@domain"
    ]
    keyword_map = {col: OrderedSet([]) for col in columns if col not in exclude_columns}
    hostmap = OrderedSet([])
    usermap = OrderedSet([])

    # Keep track of the added values per column
    added_items_per_bin = {col: [] for col in columns if col not in exclude_columns}
    added_items_per_bin["hosts"] = []
    added_items_per_bin["services"] = []

    def count_updates(ordered_set: OrderedSet, update):
        set_size = len(ordered_set)
        ordered_set.update(update)
        new_size = len(ordered_set)
        return new_size

    for batch_id, (batch, stage) in tqdm(enumerate(dataset.generate_batches()), 
                                        "Generating dataset stats in LANL",
                                        total=173):
        # Go trough every bin
        for bin in batch:
            # Update keyword map
            for col in keyword_map.keys():
                diff = count_updates(keyword_map[col], bin[col].dropna().unique())
                added_items_per_bin[col].append(diff)
            
            diff1 = count_updates(hostmap, bin["source computer"].dropna().unique())
            diff2 = count_updates(hostmap, bin["destination computer"].dropna().unique())
            added_items_per_bin["hosts"].append(diff1 + diff2)

            diff1 = count_updates(usermap, bin["source user@domain"].dropna().unique())
            diff2 = count_updates(usermap, bin["destination user@domain"].dropna().unique())
            added_items_per_bin["services"].append(diff1 + diff2)


    df = pd.DataFrame(added_items_per_bin)
    df.to_parquet("keywordmap_percent.parquet")

def LANL_unique_v2(dataset):

    # Get dataset columns
    columns = list(next(iter(dataset.generate_bins())).columns)

    # Generate keymaps
    exclude_columns = [
        "time",
        "source computer",
        "destination computer",
        "source user@domain",
        "destination user@domain"
    ]
    keyword_map = {col: OrderedSet([]) for col in columns if col not in exclude_columns}
    hostmap = OrderedSet([])
    usermap = OrderedSet([])

    # Keep track of the added values per column
    added_items_per_bin = {col: [] for col in columns if col not in exclude_columns}
    added_items_per_bin["hosts"] = []
    added_items_per_bin["services"] = []
    added_items_per_bin["stage"] = []

    def count_updates(ordered_set: OrderedSet, update):
        set_size = len(ordered_set)
        ordered_set.update(update)
        new_size = len(ordered_set)
        return new_size

    # Generate keymaps
    exclude_columns = [
        "time",
        "source computer",
        "destination computer",
        "source user@domain",
        "destination user@domain"
    ]
    keyword_map = {col: OrderedSet([]) for col in columns if col not in exclude_columns}
    hostmap = OrderedSet([])
    usermap = OrderedSet([])

    # Keep track of the added values per column
    added_items_per_bin = {col: [] for col in columns if col not in exclude_columns}
    added_items_per_bin["hosts"] = []
    added_items_per_bin["services"] = []
    added_items_per_bin["stage"] = []

    for batch_id, (batch, stage) in tqdm(enumerate(dataset.generate_batches(0)), 
                                        "Generating dataset stats in LANL (Training)"):
        # Go trough every bin
        for bin in batch:
            # Update keyword map
            for col in keyword_map.keys():
                diff = count_updates(keyword_map[col], bin[col].dropna().unique())
                added_items_per_bin[col].append(diff)
            
            diff1 = count_updates(hostmap, bin["source computer"].dropna().unique())
            diff2 = count_updates(hostmap, bin["destination computer"].dropna().unique())
            added_items_per_bin["hosts"].append(diff1 + diff2)

            diff1 = count_updates(usermap, bin["source user@domain"].dropna().unique())
            diff2 = count_updates(usermap, bin["destination user@domain"].dropna().unique())
            added_items_per_bin["services"].append(diff1 + diff2)

            added_items_per_bin["stage"].append(stage)

    for batch_id, (batch, stage) in tqdm(enumerate(dataset.generate_batches(1)), 
                                        "Generating dataset stats in LANL (Validation)"):
        # Go trough every bin
        for bin in batch:
            # Update keyword map
            for col in keyword_map.keys():
                diff = count_updates(keyword_map[col], bin[col].dropna().unique())
                added_items_per_bin[col].append(diff)
            
            diff1 = count_updates(hostmap, bin["source computer"].dropna().unique())
            diff2 = count_updates(hostmap, bin["destination computer"].dropna().unique())
            added_items_per_bin["hosts"].append(diff1 + diff2)

            diff1 = count_updates(usermap, bin["source user@domain"].dropna().unique())
            diff2 = count_updates(usermap, bin["destination user@domain"].dropna().unique())
            added_items_per_bin["services"].append(diff1 + diff2)

            added_items_per_bin["stage"].append(stage)

    for batch_id, (batch, stage) in tqdm(enumerate(dataset.generate_batches(2)), 
                                        "Generating dataset stats in LANL (Testing)"):
        # Go trough every bin
        for bin in batch:
            # Update keyword map
            for col in keyword_map.keys():
                diff = count_updates(keyword_map[col], bin[col].dropna().unique())
                added_items_per_bin[col].append(diff)
            
            diff1 = count_updates(hostmap, bin["source computer"].dropna().unique())
            diff2 = count_updates(hostmap, bin["destination computer"].dropna().unique())
            added_items_per_bin["hosts"].append(diff1 + diff2)

            diff1 = count_updates(usermap, bin["source user@domain"].dropna().unique())
            diff2 = count_updates(usermap, bin["destination user@domain"].dropna().unique())
            added_items_per_bin["services"].append(diff1 + diff2)

            added_items_per_bin["stage"].append(stage)


    df = pd.DataFrame(added_items_per_bin)
    df.to_parquet("keywordmap_percentv2.parquet")

LANL_unique_v1(dataset)
LANL_unique_v2(dataset)
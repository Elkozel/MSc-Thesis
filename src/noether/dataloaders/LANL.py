import os
from typing import Callable, Optional
import math
import bson
from elasticsearch import Elasticsearch
import torch
import logging
import pandas as pd

from torch_geometric.data import Data
from utils.elastic_datafetcher import ElasticRecordFetcher

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class LANLRecordFetcher(ElasticRecordFetcher):
    def __init__(self, es, indexes, from_sec, to_sec, pagination=10000, sort_on="timestamp", prefetch=False):
        self.from_sec = from_sec
        self.to_sec = to_sec
        self.auth_index, self.redteam_index = indexes

        query = {
            "range": {
                "time": {
                    "gte": from_sec,
                    "lt": to_sec
                }
            }
        }
        super().__init__(es, self.auth_index, query, pagination, sort_on)
        self.maps = {}
        self.authentication_type_map = None
        self.redteam_data = None
        if prefetch:
            self.prefetch()

    def prefetch(self):
        self.get_redteam_logs()
        
        self.get_map("logon type")
        self.get_map("authentication type")
        self.get_map("authentication orientation")
        self.get_map(("source computer", "destination computer"))

    # -------------------------------------------------------------------
    # Redteam Functions
    # -------------------------------------------------------------------
    def get_redteam_logs(self, force: bool = False):
        if self.redteam_data is not None and force == False:
            return self.redteam_data
        
        query = {
            "range": {
                "time": {
                    "gte": self.from_sec,
                    "lt": self.to_sec
                }
            }
        }
        rt_data = ElasticRecordFetcher(self.es, self.redteam_index, query, pagination=self.pagination, sort_on=self.sort_on)
        self.redteam_data = pd.DataFrame(rt_data.fetch_all())
        return self.redteam_data
    
    # -------------------------------------------------------------------
    # Maps
    # -------------------------------------------------------------------
    def _make_map_key(self, fields: tuple[str]) -> str:
        sep = "||"
        # sanity check: no field may contain the separator
        for f in fields:
            if sep in f:
                raise ValueError(
                    f"Field name {f!r} may not contain the map-key separator {sep!r}"
                )
        return sep.join(fields)
    
    def get_map(self, fields: str | tuple[str], force: bool = False):
        # Normalize to a tuple of strings
        if isinstance(fields, str):
            fields = (fields,)
        else:
            # If they passed e.g. a list or set, cast to tuple
            fields = tuple(fields)

        mapkey = self._make_map_key(fields)
        
        # If map is already loaded in memory, 
        if not force and mapkey in self.maps:
            return self.maps.get(mapkey)
        
        # Otherwise, fetch the data
        logger.info(
            f"Fetching unique map for {fields}. This is a one-time thing, "
            "which should take around 2-3 minutes, please be patient."
        )

        self.maps[mapkey] = self._generate_map(fields)
        self.save_maps()
        return self.maps[mapkey]

    def load_maps(self, filename: str = "maps.bson"):
        if os.path.exists(filename):
            logger.debug(f"Maps file does not exist: {filename}")
            return
        logger.debug(f"Loading maps from file: {filename}")
        try:
            with open(filename, "rb") as f:
                # Decode BSON content from the file and assign it to self.nodemap
                self.authentication_type_map = bson.decode(f.read())
                logger.debug(f"Loaded nodemap from file {filename} with {len(self.authentication_type_map)} items")
                return self.authentication_type_map
        except Exception as e:
            # Raise a descriptive exception if loading fails
            raise Exception(
                f"Failed to load nodemap from {filename}. Ensure the file has the correct privileges or is not corrupted. "
                "To reload the nodemap, call this function with force=True or manually delete the file."
            )

    def save_maps(self, filename: str = "maps.bson"):
        with open(filename, "wb") as f:
            # Encode the nodemap as BSON and write it to the file
            res = f.write(bson.BSON.encode(self.maps))
            logger.debug(f"Wrote {res} bytes into {filename}")

    def _generate_map(self, fields: tuple[str], size=5000):
        # Fetch unique values per field
        unique_per_field = [self._get_unique_values(field, size=size) for field in fields]
        all_unique = set().union(*unique_per_field)

        # Sor the unique values
        all_unique = sorted(all_unique)

        # Create the nodemap as a dictionary mapping each computer name to a unique index
        unique_mapped = {name: idx for idx, name in enumerate(all_unique)}
        logger.debug(f"Map created for fields {fields} with {len(unique_mapped)} entries.")

        return unique_mapped

    def _get_unique_values(self, field, size=5000):
        all_buckets = []
        after_key = None
        alias = "value"  # an alias for whatever field you're aggregating

        while True:
            body = {
                "size": 0,
                "aggs": {
                    "unique_values": {
                        "composite": {
                            "sources": [
                                {alias: {"terms": {"field": field}}},
                            ],
                            "size": size,
                            **({"after": after_key} if after_key else {})
                        }
                    }
                }
            }

            response = self.es.search(index=self.auth_index, body=body, request_timeout=600)
            agg = response["aggregations"]["unique_values"]
            buckets = agg["buckets"]

            all_buckets.extend(buckets)

            if "after_key" in agg:
                after_key = agg["after_key"]
            else:
                break  # No more pages

        unique_computers = set(bucket["key"][alias] for bucket in all_buckets)
        logger.debug(f"Found {len(unique_computers)} unique values for field: {field}")
        return unique_computers

class LANLGraphLoader(LANLRecordFetcher):
    def __init__(
        self,
        es: Elasticsearch,
        indexes: list[str] | str,
        from_sec: int,
        to_sec: int,
        pagination: int = 10_000,
        sort_on: str = "timestamp",
        prefetch: bool = False,
        seconds_bin: int = 1,
        transform: Optional[Callable[[Data], Data]] = None,
    ) -> None:
        """
        Initializes the ElasticDataFetcher instance.

        Args:
            es (Elasticsearch): An instance of the Elasticsearch client.
            indexes (list or str): A list of index names or a single index name to query.
            from_sec (int): The starting timestamp (in seconds) for the query range.
            to_sec (int): The ending timestamp (in seconds) for the query range.
            pagination (int, optional): The number of results to fetch per page. Defaults to 10000.
            sort_on (str, optional): The field to sort the query results on. Defaults to "datetime".
            prefetch (bool, optional): Whether to prefetch data during initialization. Defaults to False.
            seconds_bin (int, optional): The bin size in seconds for timestamp bucketing. Defaults to 1.
            transform (Callable[[Data], Data], optional): A function that takes a PyG Data
                object and returns a (possibly modified) Data object. Defaults to None.
        """
        super().__init__(es, indexes, from_sec, to_sec, pagination, sort_on, prefetch)
        self.bin_size = seconds_bin
        self.transform = transform

    def create_graph(self, records, curr_time):
        """
        Create a PyTorch Geometric graph object from LANL records.

        This function is optimized for speed using direct PyTorch tensor allocation,
        assuming that:
            - `records` is a list of dictionaries with keys: 
                'source computer', 'destination computer', 'time', and optionally 'is_malicious'.
            - The nodemap (computer name to node index) is already available.

        Records with missing nodes in the nodemap are skipped.

        Args:
            records (List[Dict]): List of parsed LANL events.

        Returns:
            torch_geometric.data.Data: Graph object containing:
                - x: Node feature matrix (placeholder zeros).
                - edge_index: Edge list (2 x num_edges).
                - time: Timestamps for each edge.
                - y: Labels (0 = benign, 1 = malicious).
        """
        logger.debug(f"Starting graph creation from {len(records)} records...")

        redteam = self.get_redteam_logs()

        nodemap = self.get_map(("source computer", "destination computer"))
        logon_map = self.get_map("logon type")
        auth_map = self.get_map("authentication type")
        authOrient_map = self.get_map("authentication orientation")
        num_records = len(records)

        edge_index = torch.empty((2, num_records), dtype=torch.long)
        edge_attr_len = 4
        edge_attr = torch.empty((num_records, edge_attr_len), dtype=torch.long)

        time_tensor = torch.empty(num_records, dtype=torch.float32)
        y_dim = 1
        y_tensor = torch.zeros((num_records, y_dim), dtype=torch.float32)  # Default to 0
        x_dim = 1
        x_tensor = torch.zeros((len(nodemap), x_dim), dtype=torch.float32)

        i = 0
        skipped = 0
        for r in records:
            try:
                malicious = ((redteam['source computer'] == r["source computer"]) & 
                             (redteam['destination computer'] == r["destination computer"]) &
                             (redteam['time'] == r["time"])).any() if not redteam.empty else False
                edge_index[0, i] = nodemap[r["source computer"]]
                edge_index[1, i] = nodemap[r["destination computer"]]
                time_tensor[i] = float(r["time"])
                edge_attr[i, 0] = 0 if r["success/failure"] == "Fail" else 1
                edge_attr[i, 1] = logon_map[r["logon type"]]
                edge_attr[i, 2] = auth_map[r["authentication type"]]
                edge_attr[i, 3] = authOrient_map[r["authentication orientation"]]
                # y_tensor[i] = int(r.get("is_malicious", 0))
                y_tensor[i] = 1 if malicious else 0
                i += 1
            except KeyError as e:
                skipped += 1
                logger.debug("Skipping record due to: %s", e)
                continue

        # Slice tensors if some records were skipped
        if skipped > 0:
            logger.warning(f"Skipped {skipped} records due to various errors. (time: {curr_time})")
            edge_index = edge_index[:, :i]
            time_tensor = time_tensor[:i]
            y_tensor = y_tensor[:i]

        logger.debug(f"Graph creation complete: {edge_index.size(1)} edges, {skipped} skipped, {x_tensor.size(0)} nodes (time: {curr_time})")

        return Data(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_tensor,
            time=time_tensor
        )
    
    def graph_postprocessing(self, graph: Data):
        if self.transform:
            graph = self.transform(graph)
        return graph

    def __iter__(self):
        """
        Generator that yields (timestamp, [records]) per second in streaming fashion.
        Assumes records are sorted by 'time'.

        Yields:
            tuple: (int timestamp, list of records for that second)
        """
        buffer = []
        current_second_bin = self.from_sec

        for record in super().__iter__():
            record_time = int(record["time"])  # Round down if float

            # Send the buffered events and update the current second if events 
            # from the next second start to appear
            if record_time >= current_second_bin+self.bin_size:
                graph = self.create_graph(buffer, current_second_bin)
                graph = self.graph_postprocessing(graph)
                yield graph
                buffer = []
                current_second_bin = record_time

            # Otherwise, append the event to the buffer
            buffer.append(record)

        # After all the records are 
        if buffer:
            yield self.create_graph(buffer, current_second_bin)

    def __len__(self):
        """
        Returns the length of data in seconds to be loaded
        """
        return math.ceil((self.to_sec - self.from_sec)/self.bin_size)
    
    def fetch_all(self):
        return list(self)
import os
import bson
import math
import torch
import numpy as np
import pandas as pd
import logging
import tempfile
from torch import Tensor
from torch_geometric.data import Data, TemporalData

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ElasticRecordFetcher:
    """
    A generator-like class that retrieves all records from an Elasticsearch index
    and supports the len() function to get the total number of matching records.

    Args:
        es (Elasticsearch): The Elasticsearch client instance.
        index_name (str): The name of the Elasticsearch index.
        query (dict): The query to filter records.
        pagination (int, optional): The number of records to fetch per request. Defaults to 10000.
        sort_on (str, optional): The value on which the events will be sorted 
    """
    def __init__(self, es, index_name, query, pagination=10000, sort_on="datetime"):
        self.es = es
        self.index_name = index_name
        self.query = query
        self.pagination = pagination
        self.sort_on = sort_on
        self.matchcount = None  # The count of matching records
        logger.debug(f"ElasticDataFetcher initialized with index: {index_name}, query: {query}, pagination: {pagination}, sort_on: {sort_on}")

    def process_record(self, record):
        """Subclasses can override this to change how records are handled."""
        return record["_source"]
    
    def __iter__(self):
        search_after = None

        while True:
            try:
                logger.debug(f"Fetching records with search_after: {search_after}")
                resp = self.es.search(
                    index=self.index_name,
                    query=self.query,
                    size=self.pagination,
                    search_after=search_after,
                    sort=[{self.sort_on: "asc"}],
                )
            except Exception as e:
                logger.error(f"Elasticsearch search failed: {e}")
                raise

            hits = resp.body.get("hits", {}).get("hits", [])
            if not hits:
                logger.debug("No more records to fetch.")
                return

            search_after = hits[-1].get("sort")
            for record in hits:
                yield self.process_record(record)

    def __len__(self):
        if self.matchcount is None:
            try:
                logger.debug("Fetching total count of matching records.")
                resp = self.es.count(index=self.index_name, query=self.query)
                self.matchcount = resp.body.get("count", 0)
            except Exception as e:
                logger.error(f"Elasticsearch count failed: {e}")
                raise
        return self.matchcount
    
    def fetch_all(self):
        return list(self)
    
    def __repr__(self):
        return f"<ElasticDataFetcher index='{self.index_name}' pagination={self.pagination}>"
    
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
        self.nodemap = None
        self.redteam_data = None
        if prefetch:
            self.prefetch()

    def prefetch(self):
        self.fetch_redteam_logs()
        self.get_nodemap()

    # -------------------------------------------------------------------
    # Redteam Functions
    # -------------------------------------------------------------------
    def fetch_redteam_logs(self):
        query = {
            "range": {
                "time": {
                    "gte": self.from_sec,
                    "lt": self.to_sec
                }
            }
        }
        rt_data = ElasticRecordFetcher(self.es, self.redteam_index, query, pagination=self.pagination, sort_on=self.sort_on)
        self.redteam_data = rt_data.fetch_all()
        return self.redteam_data

    # -------------------------------------------------------------------
    # Nodemap Functions
    # -------------------------------------------------------------------
    def get_nodemap(self, force: bool = False, filename: str = "nodemap.bson"):
        """
        Retrieve or create a nodemap that maps computer names to unique integer IDs.

        Parameters:
            force (bool): If True, forces regeneration of the nodemap even if it's already loaded or cached on disk.
            filename (str): The path to the file used to cache/load the nodemap. Defaults to 'nodemap.bson'.

        Returns:
            dict: The nodemap containing mappings from computer names to unique indices.
        """
        # If not forcing regeneration, attempt to use cached data
        if not force:
            # If nodemap is already in memory, return it
            if self.nodemap is not None:
                logger.debug("Nodemap already loaded in memory.")
                return self.nodemap
            # If nodemap is not in memory but exists on disk, load it
            if os.path.exists(filename):
                return self.load_nodemap_from_file(filename)
        
        # If forced, or nodemap not available in memory or on disk, regenerate it
        logger.info(
            "Fetching unique computers to create nodemap. This is a one-time thing, "
            "which should take around 2-3 minutes, please be patient."
        )

        # Fetch unique source and destination computers
        src_computers = self._get_unique_computers("source computer")
        dst_computers = self._get_unique_computers("destination computer")

        # Merge and sort all unique computer names
        all_computers = sorted(src_computers | dst_computers)

        # Create the nodemap as a dictionary mapping each computer name to a unique index
        self.nodemap = {name: idx for idx, name in enumerate(all_computers)}
        logger.debug(f"Nodemap created with {len(self.nodemap)} entries.")

        # Save the new nodemap to disk for future reuse
        self.save_nodemap(filename)

        return self.nodemap

    
    def load_nodemap_from_file(self, filename):
        """
        Load a nodemap (dictionary-like data structure) from a BSON file.

        Parameters:
            filename (str): The path to the file from which to load the nodemap.

        Returns:
            dict: The loaded nodemap.

        Raises:
            Exception: If the file can't be read or decoded, an informative exception is raised.
        """
        logger.debug(f"Loading nodemap from file: {filename}")
        try:
            with open(filename, "rb") as f:
                # Decode BSON content from the file and assign it to self.nodemap
                self.nodemap = bson.decode(f.read())
                logger.debug(f"Loaded nodemap from file {filename} with {len(self.nodemap)} items")
                return self.nodemap
        except Exception as e:
            # Raise a descriptive exception if loading fails
            raise Exception(
                f"Failed to load nodemap from {filename}. Ensure the file has the correct privileges or is not corrupted. "
                "To reload the nodemap, call this function with force=True or manually delete the file."
            )

    def save_nodemap(self, filename="nodemap.bson"):
        """
        Save the current nodemap to a BSON file.

        Parameters:
            filename (str): The path to the file where the nodemap should be saved.
                            Defaults to 'nodemap.bson'.

        Raises:
            Exception: If the nodemap has not been set (i.e., is None).
        """
        # Ensure nodemap is not None before attempting to save
        if self.nodemap is None:    
            raise Exception("Nodemap has not been fetched yet. Please call .get_nodemap() first.")
        
        logger.info(f"Saving nodemap to file: {filename}")
        with open(filename, "wb") as f:
            # Encode the nodemap as BSON and write it to the file
            res = f.write(bson.BSON.encode(self.nodemap))
            logger.debug(f"Wrote {res} bytes into {filename}")
    
    def _get_unique_computers(self, field, size=5000):
        all_buckets = []
        after_key = None

        while True:
            body = {
                "size": 0,
                "aggs": {
                    "unique_computers": {
                        "composite": {
                            "sources": [
                                {"computer": {"terms": {"field": field}}},
                            ],
                            "size": size,
                            **({"after": after_key} if after_key else {})
                        }
                    }
                }
            }

            response = self.es.search(index=self.auth_index, body=body, request_timeout=600)
            agg = response["aggregations"]["unique_computers"]
            buckets = agg["buckets"]

            all_buckets.extend(buckets)

            if "after_key" in agg:
                after_key = agg["after_key"]
            else:
                break  # No more pages

        unique_computers = set(bucket["key"]["computer"] for bucket in all_buckets)
        logger.debug(f"Found {len(unique_computers)} unique values for field: {field}")
        return unique_computers

class LANLGraphFetcher(LANLRecordFetcher):
    def __init__(self, es, indexes, from_sec, to_sec, pagination=10000, sort_on="timestamp", prefetch=False, seconds_bin=1):
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
        """
        super().__init__(es, indexes, from_sec, to_sec, pagination, sort_on, prefetch)
        self.bin_size = seconds_bin

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

        nodemap = self.get_nodemap()
        num_records = len(records)

        edge_index = torch.empty((2, num_records), dtype=torch.long)
        time_tensor = torch.empty(num_records, dtype=torch.float32)
        y_tensor = torch.zeros(num_records, dtype=torch.long)  # Default to 0
        x_tensor = torch.zeros((len(nodemap), 3), dtype=torch.float32)

        i = 0
        skipped = 0
        for r in records:
            try:
                edge_index[0, i] = nodemap[r["source computer"]]
                edge_index[1, i] = nodemap[r["destination computer"]]
                time_tensor[i] = float(r["time"])
                y_tensor[i] = int(r.get("is_malicious", 0))
                i += 1
            except KeyError as e:
                skipped += 1
                logger.debug("Skipping record due to missing node: %s", e)
                continue

        # Slice tensors if some records were skipped
        if skipped > 0:
            logger.warning(f"Skipped {skipped} records due to missing nodemap entries. (time: {curr_time})")
            edge_index = edge_index[:, :i]
            time_tensor = time_tensor[:i]
            y_tensor = y_tensor[:i]

        logger.debug(f"Graph creation complete: {edge_index.size(1)} edges, {skipped} skipped, {x_tensor.size(0)} nodes (time: {curr_time})")

        return Data(
            x=x_tensor,
            edge_index=edge_index,
            time=time_tensor,
            y=y_tensor
        )

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
                yield self.create_graph(buffer, current_second_bin)
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
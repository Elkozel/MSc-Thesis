from elasticsearch import Elasticsearch
from noether.utils.Elasticsearch import ElasticRecordFetcher
import torch
from torch_geometric.data import Data
import math
import logging
import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class UWF22RecordFetcher(ElasticRecordFetcher):
    def __init__(self, es, index, from_sec, to_sec, pagination=10000, sort_on="ts"):
        self.from_sec = from_sec
        self.to_sec = to_sec
        self.index = index

        query = {
            "range": {
                "ts": {
                    "gte": from_sec,
                    "lt": to_sec
                }
            }
        }
        super().__init__(es, self.index, query, pagination, sort_on)
    
class UWF22GraphFetcher(UWF22RecordFetcher):
    def __init__(
        self,
        es: Elasticsearch,
        index: str,
        from_sec: int,
        to_sec: int,
        pagination: int = 10_000,
        sort_on: str = "ts",
        seconds_bin: int = 1,
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
        super().__init__(es, index, from_sec, to_sec, pagination, sort_on)
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

class UWFGraphMaker():
    def __init__(self, engine, from_ts: int, to_ts: int, bin_size: int = 20, ufw_ts_offset = ):
        self.engine = engine
        self._nodemap_table = "ufw22_nodemap"
        self._clientmap_table = "ufw22_clientmap"
        self.from_ts = from_ts
        self.to_ts = to_ts
        self.bin_size = bin_size
        self.start_bin = math.floor(from_ts / bin_size) * bin_size
        self.end_bin = math.floor(to_ts / bin_size) * bin_size
        self.clientmap = None
        self.nodemap = None
    
    def get_nodemap(self):
        if self.nodemap is not None:
            return self.nodemap
        
        query = f"""
        SELECT * FROM {self._nodemap_table}
        ORDER BY node_id;
        """
        logger.info("Fetching nodemap")
        self.nodemap = pd.read_sql_query(query, self.engine)
        return self.nodemap
        
    def get_clientmap(self):
        if self.clientmap is not None:
            return self.clientmap
        
        query = f"""
        SELECT * FROM {self._clientmap_table}
        ORDER BY node_id;
        """
        logger.info("Fetching cleintmap")
        self.clientmap = pd.read_sql_query(query, self.engine)
        return self.clientmap
        
    def fetch_data(self):
        query = f"""
        WITH enriched AS (
            SELECT
                FLOOR(ts / {self.bin_size}) * {self.bin_size} AS time_bin,
                n_src.node_id AS src_ip_id,
                n_dst.node_id AS dst_ip_id,
                hp_src.node_id AS src_hostport_id,
                hp_dst.node_id AS dst_hostport_id,
                z.*
            FROM "ufw22" z
            JOIN ufw22_nodemap n_src
            ON z.src_ip_zeek = n_src.ip
            JOIN ufw22_nodemap n_dst
            ON z.dest_ip_zeek = n_dst.ip
            JOIN ufw22_clientmap hp_src
            ON z.src_ip_zeek = hp_src.ip AND z.src_port_zeek = hp_src.port
            JOIN ufw22_clientmap hp_dst
            ON z.dest_ip_zeek = hp_dst.ip AND z.dest_port_zeek = hp_dst.port
            WHERE z.datetime IS NOT NULL 
            AND z.ts >= {self.start_bin}
            AND z.ts < {self.end_bin}
        ),
        filtered AS (
            SELECT
                time_bin,
                src_ip_id,
                dst_ip_id,
                ts,
                label_tactic,
                COUNT(DISTINCT uid) AS flow_count
            FROM enriched
            WHERE src_ip_id <> dst_ip_id
            GROUP BY time_bin, src_ip_id, dst_ip_id, uid, ts, label_tactic
        )
        SELECT *
        FROM filtered
        ORDER BY ts;
        """

        # Execute and read into DataFrame
        df = pd.read_sql_query(query, self.engine)
        label_tactic_map = {
            "Credential Access": 0,
            "Defense Evasion": 1,
            "Discovery": 2,
            "Exfiltration": 3,
            "Initial Access": 4,
            "Lateral Movement": 5,
            "none": 6,
            "Persistence": 7,
            "Privilege Escalation": 8,
            "Reconnaissance": 9,
            "Resource Development": 10
        }
        
        data = []

        for ts in range(self.start_bin, self.end_bin, self.bin_size):
            df_ts = df[df["time_bin"] == ts]
            print(f"{df_ts.size} records between {ts} and {ts+self.bin_size}")
            if df_ts.empty:
                data.append(Data())
                print("Skipping")
                continue

            d = Data()
            d.time = torch.from_numpy(df_ts["ts"].to_numpy())
            d.x = torch.from_numpy(self.get_nodemap()[[
                "ip_is_internal",
                "is_is_external",
                "ip_is_broadcast",
                "ip_is_multicast"
            ]].to_numpy())
            d.edge_index = torch.from_numpy(df_ts[["src_ip_id", "dst_ip_id"]].to_numpy().T)
            d.y = torch.from_numpy(df_ts["label_tactic"].map(label_tactic_map).to_numpy())

            data.append(d)

        return data
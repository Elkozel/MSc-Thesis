import os
import bson
import logging
import pandas as pd

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
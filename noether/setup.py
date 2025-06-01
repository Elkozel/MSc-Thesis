from elasticsearch import Elasticsearch
from utils.elastic_helpers import check_elk, LANL_Loader

# The file is meant to be as a starter file to check that everything is fine



def main():
    print("This script will check the status of your Elasticsearch instance.")
    es_url = input("Please enter the Elasticsearch URL (e.g., http://localhost:9200): ")
    check_elk(es_url)
    
    load_data = input("Would you like to load data into Elasticsearch? (yes/no): ").strip().lower()
    if load_data == 'yes':
        print("You can implement the data loading functionality here.")
    else:
        print("No data will be loaded. Exiting.")

if __name__ == "__main__":
    es = Elasticsearch("https://localhost:9200", api_key="MGZGTDhaVUJHWEpfZm5CYVB1bXo6dXBxVk5ucF9Rc3F6dWh5RjVRVDQzUQ==", verify_certs=False, ssl_show_warn=False)
    es.ping()
    lanl = LANL_Loader("/data/LANL/", "lanl")
    lanl.push_to_elastic(es, 1719870)


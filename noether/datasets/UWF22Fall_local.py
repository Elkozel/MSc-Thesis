import logging
from datasets.UWF22_local import UWF22L


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UWF22FallL(UWF22L):

    download_data = [
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2021-12-12%20-%202021-12-19/part-00000-d512890f-d1e9-49d5-a136-f87f0183cb4d-c000.snappy.parquet",
            "raw_file": "0.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2021-12-19%20-%202021-12-26/part-00000-d28b031b-bff1-4e16-853a-9b7d896627e7-c000.snappy.parquet",
            "raw_file": "1.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2021-12-26%20-%202022-01-02/part-00000-94d13437-ae00-4a8c-9f38-edd0196cfdee-c000.snappy.parquet",
            "raw_file": "2.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-01-02%20-%202022-01-09/part-00000-745e350a-da9e-4619-bd52-8cc23bb41ad5-c000.snappy.parquet",
            "raw_file": "3.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-08-28%20-%202022-09-04/part-00000-9a46dd05-4b06-4a39-a45b-5c8460b6c37b-c000.snappy.parquet",
            "raw_file": "4.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-09-04%20-%202022-09-11/part-00000-ea53b0e8-d346-44e3-9a87-1f60ac35c610-c000.snappy.parquet",
            "raw_file": "5.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-09-11%20-%202022-09-18/part-00000-f9afaec0-242e-41e7-906d-a42681515d75-c000.snappy.parquet",
            "raw_file": "6.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-09-18%20-%202022-09-25/part-00000-9ac876be-c07d-4a18-878d-959efa26f484-c000.snappy.parquet",
            "raw_file": "7.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-09-25%20-%202022-10-02/part-00000-be6d0798-554d-4c7a-9fef-d4c07aa0ce19-c000.snappy.parquet",
            "raw_file": "8.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-10-02%20-%202022-10-09/part-00000-2b76f9cc-0710-45e4-9e33-98ad5808ee79-c000.snappy.parquet",
            "raw_file": "9.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-10-09%20-%202022-10-16/part-00000-b2b625bc-5816-4586-b977-35f9ed4487fd-c000.snappy.parquet",
            "raw_file": "10.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-10-16%20-%202022-10-23/part-00000-9aeb279c-81c6-4481-9b30-d35d4d194fea-c000.snappy.parquet",
            "raw_file": "11.parquet"
        },
        {
            "url": "https://datasets.uwf.edu/data/UWF-ZeekDataFall22/parquet/2022-10-23%20-%202022-10-30/part-00000-23fdcfa3-9dd3-4c72-886c-e945bfcf92e1-c000.snappy.parquet",
            "raw_file": "12.parquet"
        }
    ]

    def __init__(self, 
                 data_dir: str, 
                 bin_size: int = 20, 
                 batch_size: int = 350, 
                 from_time: int = 0,
                 to_time: int = 26816821, # (Relative) timestamp of last event is 26816820.542023897
                 transforms: list = [],
                 batch_split: list = [0.6, 0.25, 0.15], 
                 dataset_name: str = "UWF22Fall"):
        super().__init__(data_dir, bin_size, batch_size, from_time, to_time, transforms, batch_split, dataset_name)

        self.ts_first_event = 1639746045.251239 # This allows us to easily make time relative

        # Lastly, save those hyperparams
        self.save_hyperparameters()
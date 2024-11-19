import os
from data.ossfuzz_data import Ossfuzz

# typing
from data.core import Data
from rich.console import Console


def get_dataset(data_name: str, data_path: str, logger: Console) -> Data:

    if data_name == "ossfuzz":
        logger.log("Using OSSFuzz dataset")
        return Ossfuzz(logger=logger, data_path=data_path)
    else:
        logger.log("Dataset not found")
        return None

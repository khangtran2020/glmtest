from data.ossfuzz_data import OSSFuzzData

# typing
from data.core import Data
from rich.console import Console


def get_dataset(data_name: str, logger: Console) -> Data:

    if data_name == "ossfuzz":
        logger.log("Using OSSFuzz dataset")
        return OSSFuzzData(name=data_name)
    else:
        logger.log("Dataset not found")
        return None

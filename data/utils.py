import os
import requests
from data.ossfuzz_data import OSSFuzz

# typing
from data.core import Data
from rich.console import Console


def get_dataset(
    data_name: str,
    data_path: str,
    logger: Console,
    max_pynguin_run_time: int,
    docker_image: str = None,
    num_cpu: int = -1,
) -> Data:

    if data_name == "ossfuzz":
        logger.log("Using OSSFuzz dataset")
        return OSSFuzz(
            logger=logger,
            path=data_path,
            run_time=max_pynguin_run_time,
            docker_image=docker_image,
            num_cpu=num_cpu,
        )
    else:
        logger.log("Dataset not found")
        return None

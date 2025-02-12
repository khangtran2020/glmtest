import os
import requests
from data.ossfuzz_data import OSSFuzz
from data.codamosa_data import Codamosa
from data.testgeneval_data import TestGenEval
from graph.joerngraph import JoernGraph
from transformers import AutoTokenizer, AutoModel

# typing
from data.core import Data
from rich.console import Console


def get_dataset(
    data_name: str,
    data_path: str,
    logger: Console,
    feat_model: str,
    max_pynguin_run_time: int,
    docker_image: str = None,
    num_cpu: int = -1,
    graph: JoernGraph = None,
    debug: bool = False,
) -> Data:

    model = AutoModel.from_pretrained(feat_model)
    tokenizer = AutoTokenizer.from_pretrained(feat_model)

    if data_name == "ossfuzz":
        logger.log("Using OSSFuzz dataset")
        return OSSFuzz(
            logger=logger,
            path=data_path,
            run_time=max_pynguin_run_time,
            docker_image=docker_image,
            num_cpu=num_cpu,
            graph=graph,
            debug=debug,
        )
    elif data_name == "testgeneval":
        logger.log("Using TestGeneval dataset")
        return TestGenEval(logger=logger, path=data_path, graph=graph, debug=debug)
    else:
        logger.log("Dataset not found")
        return None

import torch
from data.ossfuzz_data import OSSFuzz
from data.codamosa_data import Codamosa
from data.testgeneval_data import TestGenEval
from graph.joerngraph import JoernGraph
from transformers import AutoTokenizer, AutoModel
from utils.constant import (
    GRAPH_START_TOKEN,
    GRAPH_PAD_TOKEN,
    GRAPH_END_TOKEN,
    FUZZ_START_TOKEN,
    FUZZ_END_TOKEN,
)

# typing
from data.core import Data
from rich.console import Console


def get_dataset(
    data_name: str,
    data_path: str,
    logger: Console,
    feat_model: str,
    llm_model: str,
    max_pynguin_run_time: int,
    docker_image: str = None,
    num_cpu: int = -1,
    graph: JoernGraph = None,
    baseline_prompt: str = "code",
    data_max_length: int = 16384,
    debug: bool = False,
    mode: str = "train",
) -> Data:

    if mode == "data":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = AutoModel.from_pretrained(feat_model, trust_remote_code=True).to(device)
    else:
        model = None

    tokenizer = AutoTokenizer.from_pretrained(feat_model, trust_remote_code=True)

    special_tokens_dict = {
        "additional_special_tokens": [
            GRAPH_START_TOKEN,
            GRAPH_PAD_TOKEN,
            GRAPH_END_TOKEN,
            FUZZ_START_TOKEN,
            FUZZ_END_TOKEN,
        ]
    }

    llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_model,
        trust_remote_code=True,
        model_max_length=data_max_length,
        padding_side="right",
        use_fast=True,
    )
    llm_tokenizer.add_special_tokens(special_tokens_dict)

    if data_name == "ossfuzz":
        logger.log("Using OSSFuzz dataset")
        return OSSFuzz(
            logger=logger,
            path=data_path,
            run_time=max_pynguin_run_time,
            docker_image=docker_image,
            num_cpu=num_cpu,
            model=model,
            tokenizer=tokenizer,
            llm_tokenizer=llm_tokenizer,
            graph=graph,
            baseline_prompt=baseline_prompt,
            debug=debug,
        )
    elif data_name == "testgeneval":
        logger.log("Using TestGeneval dataset")
        return TestGenEval(
            logger=logger,
            path=data_path,
            graph=graph,
            model=model,
            tokenizer=tokenizer,
            llm_tokenizer=llm_tokenizer,
            baseline_prompt=baseline_prompt,
            debug=debug,
        )
    else:
        logger.log("Dataset not found")
        return None

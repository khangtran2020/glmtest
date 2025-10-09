import ast
import dgl
import torch
from data.ossfuzz_data import OSSFuzz
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
    model_name: str,
    max_pynguin_run_time: int,
    docker_image: str = None,
    num_cpu: int = -1,
    graph: JoernGraph = None,
    baseline_prompt: str = "code",
    data_max_length: int = 16384,
    debug: bool = False,
    mode: str = "train",
    graph_sampling: bool = False,
    max_tokens: int = 512,
    raw_overwrite: bool = False,
    gnn_mode: str = "node",
    repo_top: int = -1,
    **kwargs,
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
        use_fast=False,
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
            model_name=model_name,
            baseline_prompt=baseline_prompt,
            debug=debug,
            graph_sampling=graph_sampling,
            n_hops=kwargs.get("n_hops", 1),
            max_tokens=max_tokens,
            raw_overwrite=raw_overwrite,
        )
    elif data_name == "testgeneval":
        logger.log("Using TestGeneval dataset")
        return TestGenEval(
            logger=logger,
            path=data_path,
            graph=graph,
            model=model,
            model_name=model_name,
            tokenizer=tokenizer,
            llm_tokenizer=llm_tokenizer,
            baseline_prompt=baseline_prompt,
            debug=debug,
            graph_sampling=graph_sampling,
            max_tokens=max_tokens,
            n_hops=kwargs.get("n_hops", 1),
            raw_overwrite=raw_overwrite,
            repo_top=repo_top,
            gnn_mode=gnn_mode,
        )
    else:
        logger.log("Dataset not found")
        return None


def sampling_neighbor(
    graph: dgl.DGLGraph, mask: torch.Tensor, n_hops: int = 2
) -> dgl.DGLGraph:
    """
    Sample the neighbors of the graph starting from a mask over multiple hops.
    """

    seeds = mask
    blocks = []

    for _ in range(n_hops):
        block = dgl.sampling.sample_neighbors(graph, seeds.long(), fanout=1)
        blocks.append(block)
        seeds = block.nodes()

    final_subgraph = dgl.node_subgraph(
        graph, torch.unique(torch.cat([b.nodes() for b in blocks]))
    )
    return final_subgraph

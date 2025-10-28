import os
import dgl
import json
import torch
import tempfile
import pandas as pd
import numpy as np
from copy import deepcopy
from utils.utils import get_index_by_value
from graph.core import Graph
from graph.utils import get_graph
from data.core import Data
from data.core import (
    PROMPT_CODE,
    PROMPT_GRAPH,
    PROMPT_CODE_GRAPH,
    PROMPT_CODE_TR,
    PROMPT_COT,
)
from accelerate import Accelerator
from inference.test import generate_and_save_on_one_dataset
from inference.verifier import verify_test_case
from data.loader import GLMFDataset, collate_fn
from sklearn.preprocessing import LabelEncoder
from transformers import PreTrainedTokenizer, PreTrainedModel
from branch.utils import get_all_branch, merge_testcases
from rich.console import Console
from functools import partial
from train.utils import extract_code_block

# typing
from typing import Dict, List
from argparse import Namespace


def testcase_generate(
    args: Namespace,
    device: torch.device,
    dataset: Data = None,
    model: PreTrainedModel = None,
    file_path: str = None,
    save_path: str = None,
    console: Console = None,
    mixed_precision: str = "bf16",
    do_generate: bool = True,
):

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with="wandb",
        project_dir=args.log_dir,
    )
    if dataset is not None and file_path is not None:
        raise ValueError("Either dataset or file_path must be provided, but not both.")

    collate_fn_ = partial(
        collate_fn, tokenizer=dataset.llm_tokenizer, max_seq_length=args.max_seq_length
    )

    if dataset is None and file_path is not None:
        graph_extractor = get_graph(
            args=args,
            graph_type=args.graph_type,
            logger=console,
        )
        feat_model = AutoModel.from_pretrained(
            args.feat_model, trust_remote_code=True
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            args.feat_model, trust_remote_code=True
        )
        te_dataset = prepare_module(
            module_path=file_path,
            save_path=save_path,
            graph_extractor=graph_extractor,
            tokenizer=tokenizer,
            feat_model=feat_model,
            device=device,
            console=console,
        )
        generated_dict = generate_and_save_on_one_dataset(
            dataset=te_dataset,
            model=model,
            args=args,
            console=console,
            device=device,
            tokenizer=dataset.llm_tokenizer,
            collate_fn_=collate_fn_,
            accelerator=accelerator,
            suffix="independent_module",
            do_save=True,
        )

        project_dict = {}
        for k, v in generated_dict.items():
            if k.split("_testcase_")[0] not in project_dict.keys():
                project_dict[k.split("_testcase_")[0]] = []
            if args.verifier_model is None:
                project_dict[k.split("_testcase_")[0]].append(
                    extract_code_block(markdown=v)
                )
            else:
                # verify the test case
                with console.status(f"Verifying test case {k}..."):
                    verification_result = verify_test_case(
                        test_case=extract_code_block(markdown=v),
                        model=args.verifier_model,
                        temperature=0.2,
                        max_tokens=2048,
                        api_key=args.verifier_api_key,
                    )
                refactored_code = verification_result["refactored_code"]
                project_dict[k.split("_testcase_")[0]].append(refactored_code)

        generated_testsrc_dict = {}
        for k, v in project_dict.items():
            test_src = merge_testcases(codes=v)
            generated_testsrc_dict[k] = test_src

        # save the generated test source code
        save_dir = os.path.join(args.gen_dir, f"{args.name}_independent_module.json")
        with console.status("Saving results..."):
            # save generated text to jsonl file
            with open(save_dir, "w", encoding="utf-8") as f:
                # save as json file
                json.dump(generated_text, f, ensure_ascii=False, indent=4)

    if dataset is not None:

        console.log("Preparing to generate test case for predefined dataset...")

        # dataset.prepare_data_for_test_gen()
        te_mod_dataset = GLMFDataset(
            data=dataset.test_data["module"],
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
            testing=True,
            dtype=args.dtype,
            num_gpus=args.num_gpu,
        )
        te_proj_dataset = GLMFDataset(
            data=dataset.test_data["project"],
            tokenizer=dataset.llm_tokenizer,
            max_seq_length=args.max_seq_length,
            debug=args.debug,
            n_hops=dataset.n_hops,
            testing=True,
            dtype=args.dtype,
            num_gpus=args.num_gpu,
        )

        if do_generate:
            generated_dict = generate_and_save_on_one_dataset(
                dataset=te_mod_dataset,
                model=model,
                args=args,
                console=console,
                device=device,
                tokenizer=dataset.llm_tokenizer,
                collate_fn_=collate_fn_,
                accelerator=accelerator,
                suffix="module_testcase",
                do_save=True,
            )
        else:
            generated_dict = None
            if os.path.exists(os.path.join(args.gen_dir, f"{args.name}_module.jsonl")):
                with open(
                    os.path.join(args.gen_dir, f"{args.name}_module.jsonl"), "r"
                ) as f:
                    for line in f.readlines():
                        instance = json.loads(line)
                        if generated_dict is None:
                            generated_dict = instance
                        else:
                            generated_dict.update(instance)
            else:
                generated_dict = None

        if generated_dict is not None:
            project_dict = {}
            for k, v in generated_dict.items():
                if k.split("_testcase_")[0] not in project_dict.keys():
                    project_dict[k.split("_testcase_")[0]] = []

                if args.verifier_model is None:
                    project_dict[k.split("_testcase_")[0]].append(
                        extract_code_block(markdown=v)
                    )
                else:
                    # verify the test case
                    with console.status(f"Verifying test case {k}..."):
                        verification_result = verify_test_case(
                            test_case=extract_code_block(markdown=v),
                            model=args.verifier_model,
                            temperature=0.2,
                            api_key=args.verifier_api_key,
                            max_tokens=2048,
                        )
                    refactored_code = verification_result["refactored_code"]
                    project_dict[k.split("_testcase_")[0]].append(refactored_code)

            # generated_testsrc_dict = {}
            # for k, v in project_dict.items():
            #     test_src = merge_testcases(codes=v)
            #     generated_testsrc_dict[k] = test_src

            # save the generated test source code
            save_dir = os.path.join(
                args.gen_dir, f"{args.name}_generated_testcase_module.json"
            )
            with open(save_dir, "w", encoding="utf-8") as f:
                # save as json file
                json.dump(project_dict, f, ensure_ascii=False, indent=4)
        else:
            console.log(
                f"[red]No generated test cases found for modules. Please check the path {os.path.join(args.gen_dir, f'{args.name}_module.json')}[/red]"
            )

        if do_generate:
            generated_dict = generate_and_save_on_one_dataset(
                dataset=te_proj_dataset,
                model=model,
                args=args,
                console=console,
                device=device,
                tokenizer=dataset.llm_tokenizer,
                collate_fn_=collate_fn_,
                accelerator=accelerator,
                suffix="project_testcase",
                do_save=True,
            )
        else:
            generated_dict = None
            if os.path.exists(os.path.join(args.gen_dir, f"{args.name}_project.jsonl")):
                with open(
                    os.path.join(args.gen_dir, f"{args.name}_project.jsonl"), "r"
                ) as f:
                    for line in f.readlines():
                        instance = json.loads(line)
                        if generated_dict is None:
                            generated_dict = instance
                        else:
                            generated_dict.update(instance)
            else:
                generated_dict = None

        if generated_dict is None:
            console.log(
                f"[red]No generated test cases found for projects. Please check the path {os.path.join(args.gen_dir, f'{args.name}_project.json')}[/red]"
            )
            return

        project_dict = {}
        for k, v in generated_dict.items():
            if k.split("_testcase_")[0] not in project_dict.keys():
                project_dict[k.split("_testcase_")[0]] = []
            project_dict[k.split("_testcase_")[0]].append(
                extract_code_block(markdown=v)
            )

        save_dir = os.path.join(
            args.gen_dir, f"{args.name}_generated_testcase_project.json"
        )
        with open(save_dir, "w", encoding="utf-8") as f:
            # save as json file
            json.dump(project_dict, f, ensure_ascii=False, indent=4)


def prepare_module(
    args: Namespace,
    code: str = None,
    module_path: str = None,
    save_path: str = None,
    graph_extractor: Graph = None,
    llm_tokenizer: PreTrainedTokenizer = None,
    tokenizer: PreTrainedTokenizer = None,
    feat_model: PreTrainedModel = None,
    device: torch.device = torch.device("cpu"),
    console: Console = None,
):

    uuid = (
        module_path.split("/")[-1].split(".")[0] if module_path is not None else "temp"
    )
    if save_path is None:
        save_path = tempfile.mkdtemp()
        os.makedirs(save_path, exist_ok=True)

    if (code is None) == (module_path is None):
        raise ValueError("Either code or module_path must be provided, but not both.")

    if code is None:
        assert module_path is not None, "module_path must be provided if code is None"
        with open(module_path, "r") as f:
            code = f.read()
    else:
        code_path = os.path.join(save_path, "module.py")
        with open(code_path, "w") as f:
            f.write(code)

    graph_path = os.path.join(save_path, "graph.pt")
    os.makedirs(graph_path, exist_ok=True)

    # extract graph from code
    graph = graph_extractor.extract_graph(
        code_path=code_path,
        save_path=graph_path,
        overwrite=True,
    )

    node_feat = get_node_features(
        graph=graph, tokenizer=tokenizer, device=device, feat_model=feat_model
    )

    graph_dict = {}
    num_nodes = len(graph["nodes"])
    assert num_nodes == node_feat.shape[0]
    edge_dict = read_edge(graph)

    for etype in edge_dict.keys():
        u = torch.Tensor(edge_dict[etype][0]).long()
        v = torch.Tensor(edge_dict[etype][1]).long()
        graph = dgl.graph((u, v), num_nodes=num_nodes)
        graph.ndata["feat"] = node_feat
        graph_dict[etype] = graph
    graph_dict["num_nodes"] = num_nodes
    graph_dict["feat_size"] = node_feat.size()
    torch.save(graph_dict, graph_path)

    branches = get_all_branch(code=code)
    all_masks = []
    for branch in enumerate(branches):
        mask = get_mask_tensor(graph=graph, branch=branch)
        all_masks.append(mask)

    if console is not None:
        log_info = f"Module processed. Number of branches: {len(branches)}, Feature shape: {node_feat.shape}, number of masks: {len(all_masks)}"
        console.log(log_info)

    # build the prompts
    processed_data = {}
    for i, branch in enumerate(branches):
        prompt = get_prompt(
            src_code=code,
            mask=all_masks[i],
            branch=branch,
            tokenizer=tokenizer,
            gnn_mode=args.gnn_mode,
            baseline_prompt=args.baseline_prompt,
            max_tokens=args.max_seq_length,
        )

        assert len(all_masks[i].shape) == 2, "Mask shape must be (1, num_nodes)"
        active_node = get_index_by_value(a=all_masks[i][0], val=1)

        if "graph" in args.baseline_prompt:
            data = {
                "uuid": f"{uuid}_{i}",
                "prompt": prompt,
                "active_node": active_node.tolist(),
                "mask": all_masks[i].tolist(),
                "graph_path": graph_path,
            }
        else:
            data = {
                "uuid": f"{uuid}_{i}",
                "prompt": prompt,
                "active_node": None,
                "mask": None,
                "graph_path": None,
            }

        data_name = f"{uuid}_testcase_{i}.json"
        data_path = os.path.join(save_path, data_name)
        with open(data_path, "w") as file:
            json.dump(data, file, indent=4)

        processed_data[f"{uuid}_testcase_{i}"] = data_path

    # build a GLMFDataset from the graph, node_feat, and all_masks
    dataset = GLMFDataset(
        data=processed_data,
        tokenizer=llm_tokenizer,
        max_seq_length=args.max_seq_length,
        debug=args.debug,
        n_hops=args.n_layers,
        testing=True,
        dtype=args.dtype,
        num_gpus=args.num_gpu,
    )
    return dataset


def get_node_features(
    graph: Dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    feat_model: PreTrainedModel,
) -> torch.Tensor:
    df = preprocess_graph(graph)
    embeddings = []
    labels = df["LABELS"]

    # Encode LABELS to integers
    label_encoder = LabelEncoder()
    df["LABELS_ENCODED"] = label_encoder.fit_transform(labels)

    # Get Code Embedding
    for code in df["CODE"].tolist():
        inputs = tokenizer(
            code,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        ).to(device)
        with torch.no_grad():
            embedding = feat_model.encoder(**inputs).last_hidden_state.mean(dim=1)[0]
        embeddings.append(embedding.to("cpu").numpy())

    df["CODE_FEATURE"] = embeddings
    # df = df.drop(["LABELS","CODE"],axis=1)
    c_features = deepcopy(df["CODE_FEATURE"])
    df = df[["LABELS_ENCODED", "COLUMN_NUMBER", "ORDER", "LINE_NUMBER"]]
    feat_df = torch.from_numpy(df.values).float()
    c_features = np.concatenate([np.expand_dims(e, 0) for e in c_features], axis=0)
    c_features = torch.from_numpy(c_features).float()
    feat = torch.cat([feat_df, c_features], dim=1)
    return feat


def preprocess_graph(graph):
    labels = []
    cnum = []
    order = []
    code = []
    lnum = []

    for node in graph["nodes"]:
        properties = node["properties"]
        labels.append(node["label"])
        try:
            cnum.append(properties["COLUMN_NUMBER"])
        except:
            cnum.append(-1)
        try:
            order.append(properties["ORDER"])
        except:
            order.append(-1)

        try:
            lnum.append(properties["LINE_NUMBER"])
        except:
            lnum.append(-1)

        try:
            if properties["CODE"] != "":
                code.append(properties["CODE"])
            else:
                code.append("EMPTY")
        except:
            code.append("EMPTY")

    nodes = pd.DataFrame(
        {
            "LABELS": np.array(labels),
            "COLUMN_NUMBER": np.array(cnum),
            "ORDER": np.array(order),
            "CODE": np.array(code),
            "LINE_NUMBER": np.array(lnum),
        }
    )
    return nodes


def get_mask_tensor(graph: Dict, branch: List) -> torch.Tensor:

    mask = np.zeros(len(graph["nodes"]))
    line_list = list(set(np.concatenate(np.array(branch, dtype=object)).tolist()))
    for i in range(len(graph["nodes"])):
        node = graph["nodes"][i]
        if node["location"]["filename"] == "N/A":
            try:
                if node["properties"]["LINE_NUMBER"] in line_list:
                    mask[i] = 1
            except:
                mask[i] = 0
    mask = torch.Tensor([mask])
    return mask


def get_prompt(
    src_code: str,
    mask: torch.Tensor,
    branch: List,
    tokenizer: PreTrainedTokenizer,
    gnn_mode: str = "graph",
    baseline_prompt: str = "graph_tr",
    max_tokens: int = 2048,
):

    if gnn_mode == "graph":
        graph_pad = "<|graph_pad|>"
    else:
        graph_pad = "<|graph_pad|>" * mask.size(0)
    if baseline_prompt == "code":
        code_line = generate_code_line(branch)
        text = PROMPT_CODE.format(src_code, code_line)
    elif baseline_prompt == "graph":
        text = PROMPT_GRAPH.format(graph_pad)
    elif baseline_prompt == "code_graph":
        text = PROMPT_CODE_GRAPH.format(src_code, graph_pad)
    elif baseline_prompt == "code_tr":
        # logger.log("Truncating code...")
        trucated_code = truncate_code(src_code=src_code, branch=branch)
        if trucated_code is None:
            return None
        text = PROMPT_CODE_TR.format(trucated_code)
    elif baseline_prompt == "graph_tr":
        trucated_code = truncate_code(src_code=src_code, branch=branch)
        text = PROMPT_CODE_GRAPH.format(trucated_code, graph_pad)
    elif baseline_prompt == "code_baseline":
        code_line = generate_code_line(branch)
        text = PROMPT_COT.format(module=src_code, execution_branch=code_line)

    task_prompt_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
    )

    if len(tokenizer.tokenize(task_prompt_input)) > max_tokens:
        return None

    return task_prompt_input


def truncate_code(src_code: str, branch: list) -> str:
    set_of_line = []
    for item in branch:
        set_of_line.extend(item)
    set_of_line = sorted(list(set(set_of_line)))

    code_line = src_code.split("\n")
    truncated_code = ""

    for i, line in enumerate(set_of_line):

        if code_line[line - 1].strip().startswith('"""'):
            continue
        if i == 0:
            truncated_code += code_line[line - 1]
            truncated_code += "\n"
        else:
            if line - 2 not in set_of_line:
                indent = len(code_line[line - 1]) - len(code_line[line - 1].lstrip())
                truncated_code += " " * indent + "...\n"
            truncated_code += code_line[line - 1]
            truncated_code += "\n"
    return truncated_code


def generate_code_line(branch):

    code_line = ""
    for item in branch:
        line = "->".join([str(i) for i in item])
        code_line += line + "\n"
    return code_line


def read_edge(graph: dict) -> dict:
    node_dict = get_node_id_dict(graph)
    edge_dict = {}
    for edge in graph["edges"]:
        if edge["label"] not in edge_dict:
            edge_dict[edge["label"]] = [
                [node_dict[edge["src"]]],
                [node_dict[edge["dst"]]],
            ]
        else:
            edge_dict[edge["label"]][0].append(node_dict[edge["src"]])
            edge_dict[edge["label"]][1].append(node_dict[edge["dst"]])
    return edge_dict


def get_node_id_dict(graph: dict) -> dict:
    node_dict = {}
    for i in range(len(graph["nodes"])):
        node = graph["nodes"][i]
        node_dict[node["id"]] = i
    return node_dict

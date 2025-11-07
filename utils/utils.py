import os
import re
import ast
import torch
import psutil
import shlex
import pickle
import random
import requests
import subprocess
import numpy as np
from typing import Dict, List
from utils.console import log_table
import ast
import re
from collections import Counter
from typing import List, Tuple, Dict, Any
import math


def tokenize_code(code: str) -> List[str]:
    """Tokenize code into tokens, handling both keywords and identifiers."""
    # Simple tokenization - splits on whitespace and common delimiters
    tokens = re.findall(r"\w+|[^\w\s]", code)
    return [token.lower() for token in tokens if token.strip()]


def extract_ast_nodes(code: str, language: str = "python") -> List[str]:
    """Extract AST node types from code."""
    if language.lower() != "python":
        # For non-Python languages, return empty list
        # In a full implementation, you'd add parsers for other languages
        return []

    try:
        tree = ast.parse(code)
        nodes = []
        for node in ast.walk(tree):
            nodes.append(type(node).__name__)
        return nodes
    except:
        return []


def extract_dataflow(code: str) -> List[Tuple[str, str]]:
    """Extract simple dataflow dependencies (variable definitions and uses)."""
    try:
        tree = ast.parse(code)
        dataflow = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Variable assignment
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Name):
                            dataflow.append((node.value.id, target.id))
            elif isinstance(node, ast.FunctionDef):
                # Function definition
                for arg in node.args.args:
                    dataflow.append(("param", arg.arg))

        return dataflow
    except:
        return []


def calculate_ngram_match(
    reference_tokens: List[str], hypothesis_tokens: List[str], n: int
) -> Tuple[int, int]:
    """Calculate n-gram matches between reference and hypothesis."""
    if len(hypothesis_tokens) < n:
        return 0, 0

    ref_ngrams = Counter()
    hyp_ngrams = Counter()

    # Generate n-grams for reference
    for i in range(len(reference_tokens) - n + 1):
        ngram = tuple(reference_tokens[i : i + n])
        ref_ngrams[ngram] += 1

    # Generate n-grams for hypothesis
    for i in range(len(hypothesis_tokens) - n + 1):
        ngram = tuple(hypothesis_tokens[i : i + n])
        hyp_ngrams[ngram] += 1

    # Count matches
    matches = 0
    for ngram, count in hyp_ngrams.items():
        matches += min(count, ref_ngrams.get(ngram, 0))

    total_hyp_ngrams = sum(hyp_ngrams.values())
    return matches, total_hyp_ngrams


def calculate_bleu_score(
    reference_tokens: List[str], hypothesis_tokens: List[str], max_n: int = 4
) -> float:
    """Calculate BLEU score."""
    if len(hypothesis_tokens) == 0:
        return 0.0

    # Calculate precision for each n-gram
    precisions = []
    for n in range(1, max_n + 1):
        matches, total = calculate_ngram_match(reference_tokens, hypothesis_tokens, n)
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0

    bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))

    # Brevity penalty
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)

    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

    return bp * bleu


def calculate_ast_similarity(ref_nodes: List[str], hyp_nodes: List[str]) -> float:
    """Calculate AST node similarity."""
    if not ref_nodes and not hyp_nodes:
        return 1.0
    if not ref_nodes or not hyp_nodes:
        return 0.0

    ref_counter = Counter(ref_nodes)
    hyp_counter = Counter(hyp_nodes)

    # Calculate Jaccard similarity
    intersection = sum(
        min(ref_counter[node], hyp_counter[node]) for node in ref_counter
    )
    union = sum(
        max(ref_counter[node], hyp_counter[node]) for node in set(ref_nodes + hyp_nodes)
    )

    return intersection / union if union > 0 else 0.0


def calculate_dataflow_similarity(
    ref_dataflow: List[Tuple[str, str]], hyp_dataflow: List[Tuple[str, str]]
) -> float:
    """Calculate dataflow similarity."""
    if not ref_dataflow and not hyp_dataflow:
        return 1.0
    if not ref_dataflow or not hyp_dataflow:
        return 0.0

    ref_set = set(ref_dataflow)
    hyp_set = set(hyp_dataflow)

    if len(ref_set | hyp_set) == 0:
        return 1.0

    return len(ref_set & hyp_set) / len(ref_set | hyp_set)


def calculate_codebleu(
    reference: str,
    hypothesis: str,
    language: str = "python",
    alpha: float = 0.25,
    beta: float = 0.25,
    gamma: float = 0.25,
    theta: float = 0.25,
) -> Dict[str, float]:
    """
    Calculate CodeBLEU score between reference and hypothesis code.

    Args:
        reference: Reference code string
        hypothesis: Hypothesis/generated code string
        language: Programming language (default: 'python')
        alpha: Weight for BLEU score (default: 0.25)
        beta: Weight for weighted n-gram match (default: 0.25)
        gamma: Weight for AST match (default: 0.25)
        theta: Weight for dataflow match (default: 0.25)

    Returns:
        Dictionary containing individual scores and final CodeBLEU score
    """

    # Tokenize both codes
    ref_tokens = tokenize_code(reference)
    hyp_tokens = tokenize_code(hypothesis)

    # Calculate BLEU score
    bleu_score = calculate_bleu_score(ref_tokens, hyp_tokens)

    # Calculate weighted n-gram match (similar to BLEU but with different weighting)
    weighted_ngram_score = calculate_bleu_score(ref_tokens, hyp_tokens, max_n=2)

    # Extract and compare AST
    ref_ast = extract_ast_nodes(reference, language)
    hyp_ast = extract_ast_nodes(hypothesis, language)
    ast_score = calculate_ast_similarity(ref_ast, hyp_ast)

    # Extract and compare dataflow
    ref_dataflow = extract_dataflow(reference)
    hyp_dataflow = extract_dataflow(hypothesis)
    dataflow_score = calculate_dataflow_similarity(ref_dataflow, hyp_dataflow)

    # Calculate final CodeBLEU score
    codebleu_score = (
        alpha * bleu_score
        + beta * weighted_ngram_score
        + gamma * ast_score
        + theta * dataflow_score
    )

    return {
        "bleu_score": bleu_score,
        "weighted_ngram_score": weighted_ngram_score,
        "ast_score": ast_score,
        "dataflow_score": dataflow_score,
        "codebleu_score": codebleu_score,
    }


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_dict(path: str, dct: Dict):
    with open(path, "wb") as f:
        pickle.dump(dct, f)


def init_history():

    history = {
        "tr_loss": [],
        "tr_perf": [],
        "va_loss": [],
        "va_perf": [],
        "te_loss": [],
        "te_perf": [],
        "dp": [],
        "eqopp": [],
        "eqodd": [],
    }

    return history


def print_args(args):
    arg_dict = {}
    for key in vars(args).keys():
        arg_dict[f"{key}"] = f"{getattr(args, key)}"
    log_table(dct=arg_dict, name="Arguments")
    return arg_dict


# def get_index_by_value(a, val):
#     return (a == val).nonzero(as_tuple=True)[0]


def check_package_exists_in_pypi(package_name: str) -> bool:

    url = f"https://pypi.org/pypi/{package_name}/json"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"The package '{package_name}' exists on PyPI.")
            return True
        else:
            print(f"The package '{package_name}' does NOT exist on PyPI.")
            return False
    except requests.RequestException as e:
        print(f"An error occurred while checking the package: {e}")
        return False


def run_command(command: str, capture_output: bool = False):
    """Run a shell command and optionally capture its output."""
    try:
        if capture_output:
            result = subprocess.run(
                shlex.split(command),
                shell=True,
                text=True,
                check=True,
                stdout=subprocess.PIPE,
            )
            return result.stdout.strip()
        else:
            subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")


def check_docker_image_exists(image_name: str) -> bool:
    try:
        # Run the 'docker images' command and capture the output
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error checking Docker images: {result.stderr}")
            return False

        # Check if the image exists in the output
        images = result.stdout.splitlines()
        for image in images:
            if image_name in image:
                print(f"Image '{image_name}' exists.")
                return True

        print(f"Image '{image_name}' does not exist.")
        return False

    except FileNotFoundError:
        print("Docker is not installed or not in PATH.")
        return False


def extract_list_content(input_string) -> List[str]:
    # Regular expression to find content between 'List(' and ')', including multiline content
    pattern = r"List\((.*?)\)"
    matches = re.findall(pattern, input_string, re.DOTALL)

    return matches


def get_index_by_value(a, val) -> torch.Tensor:
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
    assert isinstance(
        a, torch.Tensor
    ), f"Expected torch.Tensor or np.ndarray, got {type(a)}"
    return (a == val).nonzero(as_tuple=True)[0]


def handle_location_out(out_str: str) -> List[dict]:
    out_str = out_str.strip()
    lines = out_str.split("\n")[1:-1]  # Remove first and last line
    data = []
    stack = []
    for line in lines:
        stack.append(line)
        if "NewLocation(" in line:
            new_data = {}
            for key in ["filename", "lineNumber"]:
                new_data[key] = None  # line.split(key + " = ")[1].split(",")[0].strip()
        if (")" in line) & (len(line) - len(line.lstrip()) == 2):
            for l in stack:
                if "filename = " in line:
                    # print(line)
                    value = line.split("filename = ")[1].split(",")[0].strip()
                    new_data["filename"] = value if value != "<empty>" else None
                elif "lineNumber = " in line:
                    value = line.split("lineNumber = ")[1].split(",")[0].strip()
                    value = value if value != "None" else None
                    if value is not None:
                        value = int(value.split("value = ")[1].split(")")[0].strip())
                    new_data["lineNumber"] = value
                    break
            stack = []
            data.append(new_data)
    return data


def log_ram_usage():
    proc = psutil.Process(os.getpid())
    """Log current RSS memory in megabytes."""
    rss_bytes = proc.memory_info().rss
    rss_mb = rss_bytes / 1024**2
    # console.log(f"[Step {step}] RAM usage: {rss_mb:.1f} MB")
    return rss_mb


def get_depth(lst):
    if isinstance(lst, list) and lst:
        return 1 + max(get_depth(item) for item in lst)
    else:
        return 0

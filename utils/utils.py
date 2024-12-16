import os
import shlex
import pickle
import random
import requests
import subprocess
import numpy as np
from typing import Dict
from utils.console import log_table


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


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


def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]


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

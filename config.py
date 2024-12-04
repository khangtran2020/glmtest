import argparse


def add_general_group(group):
    group.add_argument(
        "--mode",
        type=str,
        default="train",
        help="mode of the program: train, test, crawl, process",
    )
    group.add_argument("--seed", type=int, default=2605, help="seed value")
    group.add_argument("--debug", type=int, default=1, help="debug mode 1/0")


def add_data_group(group):
    group.add_argument("--data_path", type=str, help="dir path to dataset")
    group.add_argument(
        "--max_pynguin_run_time", type=int, help="time limit for pynguin", default=10
    )
    group.add_argument("--data", type=str, help="name of dataset")


def parse_args():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")

    add_data_group(data_group)
    add_general_group(general_group)
    return parser.parse_args()

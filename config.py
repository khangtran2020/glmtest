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
    group.add_argument(
        "--test_gen_project", type=str, help="name of project to test gen"
    )
    group.add_argument("--docker-image", type=str, help="docker image to use")
    group.add_argument("--graph_type", type=str, default="joern", help="graph type")
    group.add_argument("--num_cpu", type=int, default=-1, help="number of cpus to use")


def add_joern_group(group):
    group.add_argument(
        "--joern_host", type=str, help="host of joern server", default="localhost"
    )
    group.add_argument(
        "--joern_port", type=str, help="port of joern server", default="8080"
    )
    group.add_argument(
        "--joern_path",
        type=str,
        help="path to joern",
        default="./graph/joern/",
    )
    group.add_argument(
        "--joern_install_path",
        type=str,
        help="docker image for joern",
        default="./graph/joern",
    )
    group.add_argument(
        "--graph_path", type=str, help="path to graph", default="./Graphs"
    )
    group.add_argument(
        "--joern_docker_image", type=str, help="docker image for joern", default="joern"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    general_group = parser.add_argument_group(title="General configuration")
    data_group = parser.add_argument_group(title="Data-related configuration")
    joern_group = parser.add_argument_group(title="Joern-related configuration")

    add_joern_group(joern_group)
    add_data_group(data_group)
    add_general_group(general_group)
    return parser.parse_args()

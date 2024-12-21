import argparse
from coverage.sqldata import CoverageData


def run(args):
    data = CoverageData(basename=".coverage", suffix=None, warn=None, debug=None)
    data.read()
    print(data._file_map)
    arcs = data.arcs(filename=args.filename)
    print(arcs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract coverage data")
    parser.add_argument(
        "--filename",
        type=str,
        help="The filename of the coverage data",
    )
    args = parser.parse_args()
    run(args)

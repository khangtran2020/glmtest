# GLMF: Graph Language Model Fuzzer

## Prerequisite

Since GLMF leverage `Pynguin` to create the training data, to clone the project, run the following command:

```shell
git clone --recursive https://github.com/khangtran2020/glmf 
```

## Dataset

### Set up for raw data processing

#### Pynguin container

First of all, we need to build Docker image of Pynguin. To do so, run this command:

```shell
docker build -t pynguin_runner -f pynguin/docker/Dockerfile --platform linux/amd64 ./pynguin
```

#### Coverage.py container

First of all, we need to build Docker image of Coverage.py. To do so, run this command:

```shell
docker build -t coverage -f branch/docker/Dockerfile --platform linux/amd64 ./branch
```

#### Joern server

We leverage Joern to extract CPG for each  module (`.py` file). To install Joern, please follow the instruction [here](https://docs.joern.io/installation/).

If you have `sudo` privilege, we suggest you to create a symlink. On the other heand, you can export the path to Joern by the following command:

```shell
export JOERN_PATH= Path/to/Joern
```

In addition, we also need to initialize Joern server to query CPG and get node's locations. To do so, you can follow the [instruction](https://docs.joern.io/server/), and run the following command:

```shell
cd $JOERN_PATH && ./joern --server --server-port <SERVER-PORT>
```

This command will run the Joern's server at `http://localhost:<SERVER-PORT>`

### OSS-Fuzz Dataset:

OSS-Fuzz is a framework of Fuzzers. It provide a unified framework to evaluate the ability of different fuzzer for real-world projects. You can read more of it [here](https://github.com/google/oss-fuzz).

To crawl the projects of OSS-Fuzz, run the following command:

```shell
python main.py --data ossfuzz --data_path ./Dataset --mode crawl --debug 0
```

Then, to process the raw projects of OSS-Fuzz, run the following command:

```shell
python main.py --data ossfuzz --data_path ./Dataset --mode process_raw --debug 0
```

Then, to generate the test-cases for the projects of OSS-Fuzz, run the following command:

```shell
python main.py --data ossfuzz --data_path ./Dataset --mode test_gen --debug 0
```
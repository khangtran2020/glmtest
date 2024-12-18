# GLMF: Graph Language Model Fuzzer

## Prerequisite

Since GLMF leverage `Pynguin` to create the training data, to clone the project, run the following command:

```shell
git clone --recursive https://github.com/khangtran2020/glmf 
```

## 1. Dataset

### 1.1 Test generate with Pynguin

First of all, we need to build Docker image of Pynguin. To do so, run this command:

```shell
docker build -t pynguin-runner -f pynguin/docker/Dockerfile --platform linux/amd64 ./pynguin
```

### 1.2 Extract Joern CPG graph

We leverage Joern to extract CPG for each  module (`.py` file). To install Joern, please follow the instruction [here](https://docs.joern.io/installation/).

Then, export the path to Joern by the following command:

```shell
export JOERN_PATH= Path/to/Joern
```

In addition, we also need to initialize Joern server to query CPG and get node's locations. To do so, you can follow the [instruction](https://docs.joern.io/server/), and run the following command:

```shell
cd $JOERN_PATH && ./joern --server --server-port <SERVER-PORT>
```

This command will run the Joern's server at `http://localhost:<SERVER-PORT>`

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
docker build -t pynguin-runner -f pynguin/docker/Dockerfile --platform linux/amd64 .
```

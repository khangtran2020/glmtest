FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
WORKDIR /coverage

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda

RUN apt-get update
RUN apt-get install -y wget zip unzip bzip2 libc6-i386 libc6-x32 libfreetype6 \
    ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial \
    subversion vim libasound2 libxi6 libxtst6 && \
    apt-get clean
        
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

RUN conda update conda \
    && conda create -n work python=3.10 -y

COPY ./ ./
RUN chmod +x ./run_coverage.sh

RUN conda init bash
RUN echo "conda activate work" >> ~/.bashrc
RUN export PYTHONHASHSEED=0
ENV PATH /opt/conda/envs/pet/bin:$PATH
ENV CONDA_DEFAULT_ENV $work

RUN pip install coverage

# ENTRYPOINT ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate work && ./run_coverage.sh"]
# CMD []

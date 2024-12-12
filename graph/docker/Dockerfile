# Use nvidia/cuda image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
WORKDIR /pynguin_gen

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# install anaconda

RUN apt-get update
RUN apt-get install -y wget zip unzip bzip2 libc6-i386 libc6-x32 libfreetype6 \
    ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial \
    subversion vim libasound2 libxi6 libxtst6 && \
    apt-get clean

RUN wget https://download.oracle.com/java/19/archive/jdk-19.0.2_linux-x64_bin.deb && \
    dpkg -i jdk-19.0.2_linux-x64_bin.deb &&  \
    update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk-19/bin/java 1 && \
    update-alternatives --install /usr/bin/javac javac /usr/lib/jvm/jdk-19/bin/javac 1 && \
    update-alternatives --config java &&  update-alternatives --config javac

WORKDIR /joern

# Install Joern
RUN cd /joern && \
    curl -L "https://github.com/joernio/joern/releases/latest/download/joern-install.sh" -o joern-install.sh && \
    chmod u+x joern-install.sh && \
    ./joern-install.sh --install-dir=./

# Expose port for Joern server
EXPOSE 8080

# Set working directory
WORKDIR /joern/joern-cli

# Entry point for starting Joern server
CMD ["./joern", "--server", "--server-host", "0.0.0.0", "--server-port", "8080"]
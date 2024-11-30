FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    software-properties-common fish

RUN add-apt-repository "deb http://apt.llvm.org/noble/ llvm-toolchain-noble-19 main"
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    clang-19 clang++-19 llvm-19-dev libmlir-19-dev mlir-19-tools \
    rake python3-pip libzstd-dev

RUN pip install lit filecheck==0.0.24 pre-commit --break-system-packages

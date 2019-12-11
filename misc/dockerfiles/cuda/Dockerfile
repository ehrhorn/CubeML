FROM nvidia/cuda:10.1-cudnn7-devel-centos7

RUN yum install wget -y && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && eval "$($HOME/miniconda/bin/conda shell.bash hook)" && \
    conda init
ADD . /home/CubeML

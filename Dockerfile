# base image
ARG base_image=movesrwth/stormpy:1.9.0
FROM $base_image

# additional arguments for compiling payntbind
ARG setup_args=""
# number of threads to use for parallel compilation
ARG no_threads=4

WORKDIR /opt/

# install dependencies
RUN apt-get update -qq
RUN apt-get install -y graphviz
RUN pip install click z3-solver psutil graphviz

# build paynt
WORKDIR /opt/paynt
COPY . .
WORKDIR /opt/paynt/payntbind
RUN python setup.py build_ext $setup_args -j $no_threads develop

WORKDIR /opt/paynt

# (optional) install paynt
RUN pip install -e .

WORKDIR /opt/payntdev

RUN pip install 

COPY setup.bash GILhack.patch ./

RUN setup.bash


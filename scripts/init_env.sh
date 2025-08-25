#!/bin/bash

pip install ninja packaging
git submodule update --init --recursive
bash ./install_deps.sh
OMP_NUM_THREADS=128 ./build.sh --arch "80;89;90" --nvshmem
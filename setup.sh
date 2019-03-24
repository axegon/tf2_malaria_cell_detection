#!/bin/bash

if nvidia-smi > /dev/null 2>&1; then
    echo "CUDA GPU found!"
    pip install -r requirements-gpu.txt
else
    echo "It appears you don't have a CUDA GPU or it isn't installed correctly."
    pip install -r requirements.txt
fi
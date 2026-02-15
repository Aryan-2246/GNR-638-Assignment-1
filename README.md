# GNR 638 Assignment 1

## Overview
This repository implements a deep learning framework from first principles with a C++ computational backend and a Python frontend. Core tensor operations, convolution, pooling, backward passes, and optimizer updates are implemented in C++ and exposed to Python using pybind11 bindings.

## Requirements
- Python 3.10+ (tested with Python 3.14)
- pybind11
- C++17 compiler (clang++ recommended on macOS)

## Build Instructions (C++ Backend)
From the project root (where `setup.py` is located):
```bash
python3 setup.py build_ext --inplace

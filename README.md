# GNR 638 — Assignment 1  
### Building a Deep Learning Framework from Scratch (C++ Backend + Python Frontend)

<<<<<<< HEAD
## Overview  
This project implements a lightweight deep learning framework from first principles, combining a high-performance **C++ computational backend** with a clean **Python training interface**. Core tensor operations, convolution, pooling, backward passes, and optimizer updates are implemented in C++ and exposed to Python using **pybind11**.

The main objective of this assignment is to demystify how modern deep learning frameworks (such as PyTorch and TensorFlow) work internally by building the full training pipeline from scratch, while still enabling end-to-end CNN training on real image datasets.

---

## Key Features  
- Custom Tensor class implemented in C++  
- Forward and backward passes for Convolution, ReLU, Max Pooling, and Fully Connected layers  
- Cross-entropy loss with numerical stability  
- SGD optimizer  
- Python interface via pybind11 bindings  
- End-to-end CNN training and evaluation  
- Fully reproducible training pipeline  

---

## Requirements  
- Python 3.12 
- pybind11  
- C++17 compatible compiler
- Git  

---

## Setup & Build (C++ Backend)

Install dependency:

```bash
pip install pybind11
```

Build the C++ backend (run this from the project root where `setup.py` is located):
=======
## Overview
This repository implements a deep learning framework from first principles with a C++ computational backend and a Python frontend. Core tensor operations, convolution, pooling, backward passes, and optimizer updates are implemented in C++ and exposed to Python using pybind11 bindings. Add the data_1 and data_2 with smae format of data_2/data_2 for running the setup. 

## Requirements
- Python 3.10+ (tested with Python 3.14)
- pybind11
- C++17 compiler (clang++ recommended on macOS)
- 
>>>>>>> a9c10be80146e6113250ee8a8d0ce8e135f05ebc

```bash
python3 setup.py build_ext --inplace
```


## Dataset Usage (data_1 and data_2)

To train on different datasets, simply change the `--data_path` argument.  
No code changes are required.

Train on **data_1**:

```bash
python3 model.py --data_path data_1
```

Train on **data_2**:

```bash
python3 model.py --data_path data_2
```

The framework automatically infers the number of classes from the dataset directory.
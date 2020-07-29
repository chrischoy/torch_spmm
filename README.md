# CUDA cusparseSpmm example

## Installation

```
export CXX=g++-7
python setup.py install
```

## Unittest

```
python -m unittest tests.test.TestSparseMM.spmm_int32
python -m unittest tests.test.TestSparseMM.spmm_int64
```

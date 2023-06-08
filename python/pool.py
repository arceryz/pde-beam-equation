#!/bin/python
from multiprocessing import Pool

# Function must be defined BEFORE creating the pool.
def f(x):
    return x*x

pool = Pool(processes=4)
result = pool.map(f, range(10))
print(result)

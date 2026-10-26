import numpy as np
from scipy.spatial import cKDTree
import mmap
import os
def f():
    a = os.fstat
    b = mmap.mmap
    print("OK", a, b)
f()

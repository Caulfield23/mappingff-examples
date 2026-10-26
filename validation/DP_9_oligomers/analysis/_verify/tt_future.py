from __future__ import annotations
import mmap
import os
def f():
    a = os.fstat
    b = mmap.mmap
    print("OK", a, b)
f()

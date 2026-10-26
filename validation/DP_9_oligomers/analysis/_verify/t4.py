import mmap
import os
def g():
    yield 1
    mm = mmap.mmap(0,0)
def f():
    print("in globals:", "mmap" in globals())
    return mmap.ACCESS_READ
print(f())

import mmap
import os
def f():
    print("in globals:", "mmap" in globals())
    print("value:", globals()["mmap"])
    return mmap.ACCESS_READ
print(f())

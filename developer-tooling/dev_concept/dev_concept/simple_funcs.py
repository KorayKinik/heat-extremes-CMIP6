from time import sleep

def dask_inc(x):
    print("Worker Running")
    sleep(1)
    return x + 1

def dask_add(x, y):
    sleep(1)
    return x + y

def new_func(x):
    return 1
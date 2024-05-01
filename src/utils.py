import time

# Timer
start_time = None
verbosity = None


def reset_timer():
    global start_time
    start_time = time.time()


def get_time():
    global start_time
    if start_time is None:
        reset_timer()
    return time.time() - start_time


def set_verbosity(v):
    global verbosity
    verbosity = v


def check_verbosity(v):
    global verbosity
    return verbosity == v

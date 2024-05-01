import time

# Timer
start_time = None


def reset_timer():
    global start_time
    start_time = time.time()


def get_time():
    global start_time
    if start_time is None:
        reset_timer()
    return time.time() - start_time

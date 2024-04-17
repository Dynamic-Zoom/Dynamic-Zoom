import time

# Timer
start_time = None
def reset_timer():
    global start_time
    start_time = time.time()
def get_time():
    global start_time
    return time.time() - start_time

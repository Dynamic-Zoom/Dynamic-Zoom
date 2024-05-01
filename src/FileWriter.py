import cv2
import time
import numpy as np
from src.FrameBuffer import FrameBuffer
from src.utils import get_time, check_verbosity

WAIT_TIME = 0.005  # in seconds
LOG_PREFIX = "[FileWriter]"


def log(*s):
    if check_verbosity(1):
        print(LOG_PREFIX, get_time(), *s)


def log2(*s):
    if check_verbosity(2):
        print(LOG_PREFIX, get_time(), *s)


def write_to_file(params, inputBuffer: FrameBuffer):
    filename, fps, frame_shape = params
    _codec = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(filename, _codec, fps, frame_shape)

    while (not inputBuffer.input_exhausted) or (not inputBuffer.isEmpty()):
        if inputBuffer.isEmpty():
            log2("waiting for frame")
            time.sleep(WAIT_TIME)
            continue
        else:
            log2("writing frame")
            frame = inputBuffer.getFrame().numpy().astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output_video.write(frame)
            log("written frame")

    log("file write complete")
    # Release resources
    output_video.release()

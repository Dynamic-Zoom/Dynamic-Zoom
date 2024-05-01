import cv2
import time
import numpy as np

from src.FrameBuffer import FrameBuffer
from src.utils import get_time, check_verbosity


WAIT_TIME = 0.005  # in seconds
LOG_PREFIX = "[OutputStream]"


def log(*s):
    if check_verbosity(1):
        print(LOG_PREFIX, get_time(), *s)


def log2(*s):
    if check_verbosity(2):
        print(LOG_PREFIX, get_time(), *s)


def run_output_stream(inputBuffer: FrameBuffer):
    log("Starting output stream")

    while (not inputBuffer.input_exhausted) or (not inputBuffer.isEmpty()):
        if inputBuffer.isEmpty():
            log2("waiting for frame")
            time.sleep(WAIT_TIME)
            continue
        else:
            log("rendering frame")
            frame = inputBuffer.getFrame().numpy().astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Upscaled Area", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log("User terminated video stream")
            break

    log("Output stream complete")

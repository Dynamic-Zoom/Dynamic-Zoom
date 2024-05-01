import cv2
import time
import numpy as np

from src.FrameBuffer import FrameBuffer
from src.utils import get_time


def log(*s):
    print("[OutputStream]", get_time(), *s)


WAIT_TIME = 0.005  # in seconds


def run_output_stream(inputBuffer: FrameBuffer):
    log("Starting output stream")

    while (not inputBuffer.input_exhausted) or (not inputBuffer.isEmpty()):
        if inputBuffer.isEmpty():
            log("waiting for frame")
            time.sleep(WAIT_TIME)
            continue
        else:
            log("rendering frame")
            frame = inputBuffer.getFrame().numpy().astype(np.uint8)
            cv2.imshow("Upscaled Area", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            log("User terminated video stream")
            break

    log("Output stream complete")

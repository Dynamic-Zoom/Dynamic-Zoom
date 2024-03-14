
import cv2
import time
from src.FrameBuffer import FrameBuffer

WAIT_TIME = 0.01 # in seconds

def write_to_file(params, inputBuffer: FrameBuffer):
    filename, fps, frame_shape = params
    _codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(filename, _codec, fps, frame_shape)

    while (not inputBuffer.input_exhausted) or (not inputBuffer.isEmpty()):
        if inputBuffer.isEmpty():
            time.sleep(WAIT_TIME)
            continue
        else:
            output_video.write(inputBuffer.getFrame())

    # Release resources
    output_video.release()


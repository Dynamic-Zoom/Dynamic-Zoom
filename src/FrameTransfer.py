from typing import List
import time
from src.FrameBuffer import FixedFrameBuffer, FlexibleFrameBuffer
from src.utils import get_time

WAIT_TIME = 0.05 # in seconds

def log(*s):
    print('[FrameTransfer]', get_time(), *s)
        
def transfer_frames(inputBuffer: FixedFrameBuffer, outputBuffers: List[FlexibleFrameBuffer]):
    
    while (not inputBuffer.input_exhausted) or (not inputBuffer.isEmpty()):
        if inputBuffer.isEmpty():
            log("waiting for frame")
            time.sleep(WAIT_TIME)
            continue
        else:
            log("transferring frame")
            frame = inputBuffer.getFrame()
            for outputBuffer in outputBuffers:
                outputBuffer.addFrame(frame)
            log("transferred frame")
    
    # Marking completion for output buffers
    for outputBuffer in outputBuffers:
        outputBuffer.input_exhausted = True
    log("frame transfer completed")

from typing import List
import time
import torch
from src.FrameBuffer import FrameBuffer
from src.utils import get_time, check_verbosity

WAIT_TIME = 0.005  # in seconds
LOG_PREFIX = "[ModelExecutor]"


def log(*s):
    if check_verbosity(1):
        print(LOG_PREFIX, get_time(), *s)


def log2(*s):
    if check_verbosity(2):
        print(LOG_PREFIX, get_time(), *s)


def preprocess_frame(frame):
    # Preprocessing for each frame before sending to model
    pre_frame = frame.permute(2, 0, 1).unsqueeze(0)
    return pre_frame


def postprocess_frame(frame):
    processed_frame = frame.squeeze(0).permute(1, 2, 0)
    return processed_frame


def run_model(model, inputBuffer: FrameBuffer, outputBuffers: List[FrameBuffer]):

    received_frame = None
    processed_frame = None
    unprocessed_buffers = outputBuffers

    while (
        (not inputBuffer.input_exhausted)
        or (not inputBuffer.isEmpty())
        or (processed_frame != None)
        or (received_frame != None)
    ):

        # If processed, write to output buffer
        if processed_frame != None:
            log2("writing to output buffers")
            unfinished_output_buffers = []
            for outputBuffer in unprocessed_buffers:
                if outputBuffer.isFull():
                    unfinished_output_buffers.append(outputBuffer)
                else:
                    outputBuffer.addFrame(processed_frame)
            if len(unfinished_output_buffers) > 0:
                log("Buffer is full.. extra wait added")
                time.sleep(
                    0.05
                )  # should be set something as: next_step_processing_time * next_buffer_size/2
                unprocessed_buffers = (
                    unfinished_output_buffers  # retry only unfinished buffers
                )
            else:
                processed_frame = None
                unprocessed_buffers = (
                    outputBuffers  # all completed, process all buffers next time
                )
            continue

        # Run model if we can process
        if received_frame != None and processed_frame == None:
            log2("upscaling frame")
            with torch.no_grad():
                processed_frame = postprocess_frame(
                    model(preprocess_frame(received_frame))
                )
            log("upscaled frame")
            received_frame = None
            continue

        # If can't process, read from input buffer
        if received_frame == None:
            if inputBuffer.isEmpty():
                log2("waiting for frame")
                time.sleep(WAIT_TIME)
            else:
                received_frame = inputBuffer.getFrame()
            continue

    # Marking completion for output buffers
    for outputBuffer in outputBuffers:
        outputBuffer.input_exhausted = True

    log("execution completed")

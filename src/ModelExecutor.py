import time

from src.FrameBuffer import FrameBuffer

WAIT_TIME = 0.01 # in seconds

def run_model(model, inputBuffer: FrameBuffer, outputBuffers: List[FrameBuffer]):

    received_frame = None
    processed_frame = None
    unprocessed_buffers = outputBuffers

    while (not inputBuffer.input_exhausted) or (not inputBuffer.isEmpty()) or (processed_frame != None) or (received_frame != None):
        
        # If processed, write to output buffer
        if processed_frame != None:
            unfinished_output_buffers = []
            for outputBuffer in unprocessed_buffers:
                if outputBuffer.isFull():
                    unfinished_output_buffers.append(outputBuffer)
                else:
                    outputBuffer.addFrame(processed_frame)
            if len(unfinished_output_buffers) > 0:
                time.sleep(WAIT_TIME)
                unprocessed_buffers = unfinished_output_buffers # retry only unfinished buffers
            else:
                processed_frame = None
                unprocessed_buffers = outputBuffers # all completed, process all buffers next time
            continue

        # Run model if we can process
        if received_frame != None and processed_frame == None:
            processed_frame = model(received_frame)
            received_frame = None
            continue
        
        # If can't process, read from input buffer
        if received_frame == None:
            if inputBuffer.isEmpty(): 
                time.sleep(WAIT_TIME)
            else:
                received_frame = inputBuffer.getFrame()
            continue

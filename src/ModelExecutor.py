import time

WAIT_TIME = 0.01 # in seconds

def run_model(model, inputBuffer: FrameBuffer, outputBuffers: List[FrameBuffer]):

    received_frame = None
    processed_frame = None

    while (not inputBuffer.input_exhausted) or (not inputBuffer.isEmpty()) or (processed_frame != None) or (received_frame != None):
        
        # If processed, write to output buffer
        if processed_frame != None:
            allFull = False
            # for outputBuffer in outputBuffers:
            #     if outputBuffer.isFull():
            #         time.sleep(WAIT_TIME)
            #     else:
            #         outputBuffer.addFrame(processed_frame)
            #         processed_frame = None
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

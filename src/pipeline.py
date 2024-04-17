from threading import Thread, Lock
from multiprocessing import Process
import torch
import time

from models.BicubicPlusPlus import BicubicPlusPlus

from src.FrameBuffer import FixedFrameBuffer, FlexibleFrameBuffer
from src.ModelExecutor import run_model
from src.FrameTransfer import transfer_frames
from src.FileWriter import write_to_file
from src.InputStream import mockfn as inS
from src.OutputStream import mockfn as outS

from src.utils import reset_timer, get_time

def log(*s):
    print('[Pipeline]', get_time(), *s)


def run_pipeline():
    """
    Executes pipeline with configuration parameters passed during invocation

    Initializes parallel instances of InputStream, OutputStream, ModelExecutor, FileWriter linked through FrameBuffer(s)

    Handles exceptions and instance restarts


    """
    reset_timer()
    is_cuda_available = torch.cuda.is_available()
    log('CUDA Available:', is_cuda_available)
    
    cpu = torch.device("cpu")
    gpu = torch.device("cuda" if is_cuda_available else "cpu")
    frame_shape = (720, 1280, 3)
    upscaled_frame_shape = (720*3, 1280*3, 3)
    fps = 24
    
    model = BicubicPlusPlus().to(gpu)
    model.load_state_dict(torch.load('weights/bicubic_pp_x3.pth'))
    model.eval()
    
    modelInBuffer = FixedFrameBuffer(cpu, gpu, frame_shape, buffer_size=40)
    modelOutBuffer = FixedFrameBuffer(gpu, cpu, upscaled_frame_shape, buffer_size=5)
    fileWriteBuffer = FlexibleFrameBuffer()

    t_model = Thread(
        target=run_model, args=(model, modelInBuffer, [modelOutBuffer])
    )
    t_model_transfer = Thread(
        target=transfer_frames, args=(modelOutBuffer, [fileWriteBuffer])
    )
    t_file = Thread(
        target=write_to_file, args=(("output.mp4", fps, upscaled_frame_shape[:2]), fileWriteBuffer)
    )

    t_model.start()
    t_model_transfer.start()
    t_file.start()

    # warmup pipeline
    log('Warmup started')
    modelInBuffer.addFrame(torch.zeros(frame_shape, device=cpu)) # Adding to Start buffer
    while modelOutBuffer.isEmpty(): # First check for filling the End buffer
        # print('Buffer sizes:', len(modelInBuffer), len(modelOutBuffer), len(fileWriteBuffer))
        time.sleep(0.01)
    while modelOutBuffer.isFull(): # Next check for emptying the End buffer
        # print('Buffer sizes:', len(modelInBuffer), len(modelOutBuffer), len(fileWriteBuffer))
        time.sleep(0.01)
    log('Warmup complete')
    
    # time.sleep(0.1) # Extra sleep for testing waiting messages
    
    reset_timer()
    # Testing modelInBuffer + fileOutBuffer
    t_test = Thread(target=test_thread, args=(frame_shape, cpu, modelInBuffer))
    t_test.start()

    t_model.join()
    t_model_transfer.join()
    t_file.join()
    t_test.join()

    return

def test_thread(frame_shape, cpu, modelInBuffer):
    
    for i in range(100):
        log("Adding frame", i)
        test_frame = torch.rand(frame_shape, device=cpu) * 255
        frame_added = modelInBuffer.addFrame(test_frame)
        while not frame_added:
            log("Buffer is full.. extra wait added")
            time.sleep(0.1) # should be set something as: next_step_processing_time * next_buffer_size/2
            frame_added = modelInBuffer.addFrame(test_frame)
        log("Added frame", i)
        time.sleep(0.005)
    modelInBuffer.input_exhausted = True
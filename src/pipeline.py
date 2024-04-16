from threading import Thread, Lock
import torch
import time

from models.BicubicPlusPlus import BicubicPlusPlus

from src.FrameBuffer import FixedFrameBuffer, FlexibleFrameBuffer
from src.ModelExecutor import run_model
from src.FileWriter import write_to_file
from src.InputStream import mockfn as inS
from src.OutputStream import mockfn as outS


def run_pipeline():
    """
    Executes pipeline with configuration parameters passed during invocation

    Initializes parallel instances of InputStream, OutputStream, ModelExecutor, FileWriter linked through FrameBuffer(s)

    Handles exceptions and instance restarts


    """
    print('CUDA Available:', torch.cuda.is_available())
    
    cpu = torch.device("cpu")
    gpu = torch.device("cuda")
    frame_shape = (128, 128, 3)
    upscaled_frame_shape = (128*3, 128*3, 3)
    fps = 24
    
    model = BicubicPlusPlus().to(gpu)
    model.load_state_dict(torch.load('weights/bicubic_pp_x3.pth'))
    model.eval()
    
    modelInBuffer = FixedFrameBuffer(cpu, gpu, frame_shape, buffer_size=50)
    fileOutBuffer = FixedFrameBuffer(gpu, cpu, upscaled_frame_shape, buffer_size=50)

    t_model = Thread(
        target=run_model, args=(model, modelInBuffer, [fileOutBuffer])
    )
    t_file = Thread(
        target=write_to_file, args=(("output.mp4", fps, upscaled_frame_shape[:2]), fileOutBuffer)
    )

    t_model.start()
    t_file.start()

    time.sleep(2) # Extra sleep for testing waiting messages
    
    # Testing modelInBuffer + fileOutBuffer
    for i in range(50):
        print("Adding frame", i)
        test_frame = torch.rand(frame_shape, device=cpu) * 255
        frame_added = modelInBuffer.addFrame(test_frame)
        while not frame_added:
            print("Buffer is full.. extra wait added")
            time.sleep(1)
            frame_added = modelInBuffer.addFrame(test_frame)
        time.sleep(0.05)
    modelInBuffer.input_exhausted = True

    t_model.join()
    t_file.join()

    pass

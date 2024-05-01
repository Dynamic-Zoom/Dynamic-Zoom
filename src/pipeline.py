from threading import Thread, Lock
from multiprocessing import Process
import torch
import time

from models.BicubicPlusPlus import BicubicPlusPlus
from src.FrameBuffer import FixedFrameBuffer, FlexibleFrameBuffer
from src.ModelExecutor import run_model
from src.FrameTransfer import transfer_frames
from src.FileWriter import write_to_file
from src.OutputStream import run_output_stream
from src.utils import reset_timer, get_time
from src.InputStream import run_input_stream


def log(*s):
    print("[Pipeline]", get_time(), *s)


def run_pipeline():
    """
    Executes pipeline with configuration parameters passed during invocation

    Initializes parallel instances of InputStream, OutputStream, ModelExecutor, FileWriter linked through FrameBuffer(s)

    Handles exceptions and instance restarts


    """
    reset_timer()
    is_cuda_available = torch.cuda.is_available()
    log("CUDA Available:", is_cuda_available)

    cpu = torch.device("cpu")
    gpu = torch.device("cuda" if is_cuda_available else "cpu")
    frame_shape = [180, 240, 3]
    upscaled_frame_shape = (frame_shape[0] * 3, frame_shape[1] * 3, 3)
    fps = 24

    model = BicubicPlusPlus().to(gpu)
    model.load_state_dict(torch.load("weights/bicubic_pp_x3.pth"))
    model.eval()

    modelInBuffer = FixedFrameBuffer(cpu, gpu, frame_shape, buffer_size=40)
    modelOutBuffer = FixedFrameBuffer(gpu, cpu, upscaled_frame_shape, buffer_size=5)
    outputRenderBuffer = FlexibleFrameBuffer()
    fileWriteBuffer = FlexibleFrameBuffer()

    t_input_stream = Thread(
        target=run_input_stream,
        args=(("data/input.mp4", frame_shape[0],frame_shape[1]), modelInBuffer),
    )
    t_model = Thread(target=run_model, args=(model, modelInBuffer, [modelOutBuffer]))
    t_model_transfer = Thread(
        target=transfer_frames, args=(modelOutBuffer, [fileWriteBuffer, outputRenderBuffer])
    )
    t_file = Thread(
        target=write_to_file,
        args=(("output.mp4", fps, upscaled_frame_shape[:2]), fileWriteBuffer),
    )
    t_output_stream = Thread(
        target=run_output_stream,
        args=(outputRenderBuffer,),
    )

    t_model.start()
    t_model_transfer.start()
    t_file.start()
    t_output_stream.start()

    # warmup pipeline
    log("Warmup started")
    modelInBuffer.addFrame(
        torch.zeros(frame_shape, device=cpu)
    )  # Adding to Start buffer
    while modelOutBuffer.isEmpty():  # First check for filling the End buffer
        # print('Buffer sizes:', len(modelInBuffer), len(modelOutBuffer), len(fileWriteBuffer))
        time.sleep(0.01)
    while modelOutBuffer.isFull():  # Next check for emptying the End buffer
        # print('Buffer sizes:', len(modelInBuffer), len(modelOutBuffer), len(fileWriteBuffer))
        time.sleep(0.01)
    log("Warmup complete")
    reset_timer()
    
    t_input_stream.start()

    t_input_stream.join()
    t_model.join()
    t_model_transfer.join()
    t_file.join()
    t_output_stream.join()

    return

from threading import Thread, Lock
from multiprocessing import Process
import torch
import time

from models.BicubicPlusPlus import BicubicPlusPlus
from models.SwiftSRGAN import SwiftSRGAN2x, SwiftSRGAN4x

from src.FrameBuffer import FixedFrameBuffer, FlexibleFrameBuffer
from src.ModelExecutor import run_model
from src.FrameTransfer import transfer_frames
from src.FileWriter import write_to_file
from src.OutputStream import run_output_stream
from src.utils import reset_timer, get_time, set_verbosity, check_verbosity
from src.InputStream import run_input_stream


LOG_PREFIX = "[Pipeline]"


def log(*s):
    if check_verbosity(0) or check_verbosity(1):
        print(LOG_PREFIX, get_time(), *s)


def log2(*s):
    if check_verbosity(2):
        print(LOG_PREFIX, get_time(), *s)


def get_model(model_name):
    if model_name == "bicubic++":
        ModelClass = BicubicPlusPlus
        weights_dict = torch.load("weights/bicubic_pp_x3.pth")
        upscale_factor = 3
    elif model_name == "srgan2x":
        ModelClass = SwiftSRGAN2x
        weights_dict = torch.load("weights/swift_srgan_2x.pth.tar")["model"]
        upscale_factor = 2
    elif model_name == "srgan4x":
        ModelClass = SwiftSRGAN4x
        weights_dict = torch.load("weights/swift_srgan_4x.pth.tar")["model"]
        upscale_factor = 4
    else:
        raise Exception(f"Unsupported model: {model_name}")
    return ModelClass, weights_dict, upscale_factor


def run_pipeline(args):
    """
    Executes pipeline with configuration parameters passed during invocation

    Initializes parallel instances of InputStream, OutputStream, ModelExecutor, FileWriter linked through FrameBuffer(s)

    Handles exceptions and instance restarts

    """
    set_verbosity(args.verbosity)
    reset_timer()

    is_cuda_available = torch.cuda.is_available()
    log("CUDA Available:", is_cuda_available)

    cpu = torch.device("cpu")
    gpu = torch.device("cuda" if is_cuda_available else "cpu")
    ModelClass, weights_dict, upscale_factor = get_model(args.model_name)
    model = ModelClass().to(gpu)
    model.load_state_dict(weights_dict)
    model.eval()

    frame_shape = [*args.roi, 3]  # Only RGB channels are supported
    upscaled_frame_shape = (
        frame_shape[0] * upscale_factor,
        frame_shape[1] * upscale_factor,
        frame_shape[2],
    )

    modelInBuffer = FixedFrameBuffer(cpu, gpu, frame_shape, buffer_size=40)
    modelOutBuffer = FixedFrameBuffer(gpu, cpu, upscaled_frame_shape, buffer_size=5)
    outputRenderBuffer = FlexibleFrameBuffer()
    fileWriteBuffer = FlexibleFrameBuffer()

    t_input_stream = Thread(
        target=run_input_stream,
        args=((args.input, frame_shape[0], frame_shape[1]), modelInBuffer),
    )
    t_model = Thread(target=run_model, args=(model, modelInBuffer, [modelOutBuffer]))
    t_model_transfer = Thread(
        target=transfer_frames,
        args=(modelOutBuffer, [fileWriteBuffer, outputRenderBuffer]),
    )
    t_file = Thread(
        target=write_to_file,
        args=((args.output, args.fps, upscaled_frame_shape[1::-1]), fileWriteBuffer),
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
        time.sleep(0.01)
    while modelOutBuffer.isFull():  # Next check for emptying the End buffer
        time.sleep(0.01)
    log("Warmup complete")
    reset_timer()

    t_input_stream.start()

    t_input_stream.join()
    t_model.join()
    t_model_transfer.join()
    t_file.join()
    t_output_stream.join()

    log("Pipeline completed")
    return

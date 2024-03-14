
from threading import Thread, Lock
# from 
from FileWriter import write_to_file
from FrameBuffer import FixedFrameBuffer, FlexibleFrameBuffer
from InputStream import mockfn as inS
from OutputStream import mockfn as outS


def run_pipeline():
    '''
    Executes pipeline with configuration parameters passed during invocation

    Initializes parallel instances of InputStream, OutputStream, ModelExecutor, FileWriter linked through FrameBuffer(s)

    Handles exceptions and instance restarts

    
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BicubicPlusPlus().to(device)
    model.load_state_dict(torch.load('bicubic_pp_x3.pth'))
    model.eval()
    pass

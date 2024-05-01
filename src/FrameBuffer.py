from collections import deque
import torch


class FrameBuffer:

    def __len__(self):
        raise Exception("__len__ not implemented")

    def addFrame(self):
        raise Exception("addFrame not implemented")

    def getFrame(self):
        raise Exception("getFrame not implemented")

    def isFull(self):
        raise Exception("isFull not implemented")

    def isEmpty(self):
        raise Exception("isEmpty not implemented")


class FixedFrameBuffer(FrameBuffer):
    def __init__(
        self,
        in_device,
        out_device,
        tensor_shape,
        tensor_dtype=torch.float32,
        buffer_size=10,
    ):

        self.in_device = in_device
        self.out_device = out_device
        devices = "{in_device.type};{out_device}"
        if "cpu" in devices and "cuda" in devices:
            # If cpu <-> gpu transfers are involved, use pinned memory
            self.is_pinned = True
            self.buffer = [
                torch.zeros(tensor_shape, dtype=tensor_dtype, pin_memory=True)
                for _ in range(buffer_size)
            ]
        else:
            # If no cross device transfers are involved, only maintain simple buffers
            self.is_pinned = False
            self.buffer = [None] * buffer_size
        self.add_idx = 0  # Index for adding new frames
        self.get_idx = 0  # Index for retrieving frames
        self.full = False  # Flag to mark the buffer as full

        self.buffer_size = buffer_size
        self.tensor_shape = tensor_shape
        self.tensor_dtype = tensor_dtype

        self.input_exhausted = False  # Flag to mark no new input frames

    def __len__(self):
        length = self.add_idx - self.get_idx
        if length == 0:
            if self.full:
                return self.buffer_size
            else:
                return 0
        elif length < 0:
            return self.buffer_size + length
        else:
            return length

    def addFrame(self, frame):
        if self.full:
            return False  # Buffer is full, cannot add new frame

        if not self.is_pinned:
            self.buffer[self.add_idx] = frame
        else:
            self.buffer[self.add_idx][:] = frame.to(self.in_device)

        self.add_idx = (self.add_idx + 1) % self.buffer_size

        if self.add_idx == self.get_idx:
            self.full = True

        return True  # Frame added successfully

    def getFrame(self):
        if self.add_idx == self.get_idx and not self.full:
            return None  # Buffer is empty, no frame to retrieve

        frame = self.buffer[self.get_idx].to(self.out_device)
        self.get_idx = (self.get_idx + 1) % self.buffer_size
        self.full = False  # Reset the full flag as we've just consumed a frame
        return frame

    def isFull(self):
        return self.full

    def isEmpty(self):
        return self.add_idx == self.get_idx and not self.full


class FlexibleFrameBuffer(FrameBuffer):
    """
    Flexible buffers for CPU <-> CPU transfers only
    """

    def __init__(self, soft_limit=None, show_warnings=True):
        self.buffer = deque()
        self.input_exhausted = False  # Flag to mark no new input frames
        self.soft_limit = soft_limit
        self.show_warnings = show_warnings

    def __len__(self):
        return len(self.buffer)

    def addFrame(self, frame):
        self.buffer.append(frame)
        if self.show_warnings and self.isFull():
            print("WARN: FlexibleFrameBuffer exceeding limit")
        return True

    def getFrame(self):
        if self.isEmpty():
            return None
        return self.buffer.popleft()

    def isFull(self):
        if self.soft_limit:
            return len(self.buffer) >= self.soft_limit
        return False

    def isEmpty(self):
        return len(self.buffer) == 0

import cv2
import time
import numpy as np
import torch
from src.FrameBuffer import FrameBuffer
from src.utils import get_time, check_verbosity

WINDOW_NAME = "Input Video Stream"
LOG_PREFIX = "[InputStream]"


def log(*s):
    if check_verbosity(1):
        print(LOG_PREFIX, get_time(), *s)


def log2(*s):
    if check_verbosity(2):
        print(LOG_PREFIX, get_time(), *s)


def addTextToImg(text: str, img: np.ndarray):

    # Write some Text
    bottomLeftCornerOfText = (10, img.shape[0] - 12)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    fontColor = (255, 255, 255)
    thickness = 2
    lineType = 2

    cv2.putText(
        img,
        text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType,
    )
    return


class VideoProcessor:
    def __init__(self, params, outputBuffer: FrameBuffer):
        self.filePath, self.crop_height, self.crop_width = params
        self.outputBuffer = outputBuffer
        self.cursor_x, self.cursor_y = (
            self.crop_width // 2,
            self.crop_height // 2,
        )  # Starting position
        self.cap = cv2.VideoCapture(self.filePath)
        self.frame_max_x = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 1
        self.frame_max_y = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 1

        if not self.cap.isOpened():
            log("Error: Could not open video.")
            exit()

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.update_cursor_position)

    def update_cursor_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_x = min(max(x, 0), self.frame_max_x)
            self.cursor_y = min(max(y, 0), self.frame_max_y)
            log2("Mouse event triggered: ", x, y)

    def calculate_bounds(self):
        del_x = self.crop_width // 2
        if self.cursor_x < del_x:
            x_start = 0
        elif self.cursor_x > (self.frame_max_x - del_x):
            x_start = self.frame_max_x - self.crop_width
        else:
            x_start = self.cursor_x - del_x
        x_end = x_start + self.crop_width
        del_y = self.crop_height // 2
        if self.cursor_y < del_y:
            y_start = 0
        elif self.cursor_y > (self.frame_max_y - del_y):
            y_start = self.frame_max_y - self.crop_height
        else:
            y_start = self.cursor_y - del_y
        y_end = y_start + self.crop_height
        return x_start, x_end, y_start, y_end

    def process_frames(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = -1
        while True:
            frame_count += 1
            ret, frame = self.cap.read()
            if not ret:
                # TESTING CODE BEGIN - Infinite loop is useful for testing
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                # TESTING CODE END - Comment out above code for evaluation

                if frame_count >= total_frames:
                    log("Stream exhausted")
                else:
                    log("Error: Could not read frame.")
                break

            x_start, x_end, y_start, y_end = self.calculate_bounds()

            if x_end > x_start and y_end > y_start:
                cropped_frame = frame[y_start:y_end, x_start:x_end]
                cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                cropped_tensor = torch.tensor(cropped_frame, dtype=torch.float32)
                self.outputBuffer.addFrame(cropped_tensor)

                frame_with_rect = frame.copy()
                cv2.rectangle(
                    frame_with_rect, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2
                )
                addTextToImg("Press 'q' to exit", frame_with_rect)
                cv2.imshow(WINDOW_NAME, frame_with_rect)
                # cv2.imshow("Cropped Area", cropped_frame) # Disabled redundant cropped stream

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                log("User terminated video stream")
                break

        self.outputBuffer.input_exhausted = True

        self.cap.release()
        # cv2.destroyAllWindows() # Disabled this as it attempts to hang output windows
        log("Input Stream complete")


def run_input_stream(params, output_buffer):
    VideoProcessor(params, output_buffer).process_frames()

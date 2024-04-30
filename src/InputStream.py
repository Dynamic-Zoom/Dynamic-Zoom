import cv2
import time
import numpy as np
import torch
from src.FrameBuffer import FrameBuffer, FixedFrameBuffer
from src.utils import get_time

class VideoProcessor:
    def __init__(self, filePath, outputBuffer):
        self.filePath = filePath
        self.outputBuffer = outputBuffer
        self.cursor_x, self.cursor_y = 100, 100  # Starting position
        self.crop_width, self.crop_height = 200, 200  # Size of the cropped area
        self.cap = cv2.VideoCapture(self.filePath)
        
        if not self.cap.isOpened():
            print("[InputStream]", get_time(), "Error: Could not open video.")
            exit()

        cv2.namedWindow('Video Stream')
        cv2.setMouseCallback('Video Stream', self.update_cursor_position)
    
    def update_cursor_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            max_x = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 1
            max_y = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 1
            self.cursor_x = min(max(x, 0), max_x)
            self.cursor_y = min(max(y, 0), max_y)
            print("[InputStream]", get_time(), "Mouse event triggered: ", x, y)

    def process_frames(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in range(total_frames):
            ret, frame = self.cap.read()
            if not ret:
                print("[InputStream]", get_time(), "Error: Could not read frame.")
                break

            x_start = max(0, self.cursor_x - self.crop_width // 2)
            y_start = max(0, self.cursor_y - self.crop_height // 2)
            x_end = min(frame.shape[1], self.cursor_x + self.crop_width // 2)
            y_end = min(frame.shape[0], self.cursor_y + self.crop_height // 2)

            if x_end > x_start and y_end > y_start:
                cropped_frame = frame[y_start:y_end, x_start:x_end]
                frame_with_rect = frame.copy()
                cv2.rectangle(frame_with_rect, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.imshow('Video Stream', frame_with_rect)
                cv2.imshow('Cropped Area', cropped_frame)
                cropped_tensor = torch.tensor(cropped_frame, dtype=torch.float32)
                self.outputBuffer.addFrame(cropped_tensor)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.outputBuffer.input_exhausted = True
        print("[InputStream]", get_time(), "Input Stream complete")
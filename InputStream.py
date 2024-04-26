import cv2
import time
import numpy as np
from src.FrameBuffer import FrameBuffer
from src.utils import get_time

WAIT_TIME = 0.01 # in seconds

def log(*s):
    print('[InputStream]', get_time(), *s)

def InputStream(filePath, inputBuffer: FrameBuffer):
    # Initialize global variables for the cursor position. Should these go into utils.py?   
    cursor_x, cursor_y = 100, 100  # Starting position
    crop_width, crop_height = 200, 200  # Size of the cropped area

    # Mouse callback function to update cursor position
    def update_cursor_position(event, x, y, flags, param):
        global cursor_x, cursor_y
        if event == cv2.EVENT_MOUSEMOVE:
            # Clamp the cursor position within the window size
            max_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 1
            max_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) - 1
            cursor_x = min(max(x, 0), max_x)
            cursor_y = min(max(y, 0), max_y)

    path = filePath
    cap = cv2.VideoCapture(path)  # path to video file

    # Check if the video capture has been initialized correctly
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Calculate the delay between frames in milliseconds
    delay = int(1000 / fps)

    cv2.namedWindow('Video Stream')
    cv2.setMouseCallback('Video Stream', update_cursor_position)

    while True:
        # Read a new frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Ensure the cropped area does not exceed the frame boundaries
        x_start = max(0, cursor_x - crop_width // 2)
        y_start = max(0, cursor_y - crop_height // 2)
        x_end = min(frame.shape[1], cursor_x + crop_width // 2)
        y_end = min(frame.shape[0], cursor_y + crop_height // 2)

        # Crop the frame if valid dimensions exist
        if (x_end > x_start) and (y_end > y_start):
            cropped_frame = frame[y_start:y_end, x_start:x_end]
            # Optionally resize the cropped frame if needed here
            
            # Display the original frame with a rectangle showing the crop area
            frame_with_rect = frame.copy()
            cv2.rectangle(frame_with_rect, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

            # Optionally concatenate and display frames
            cv2.imshow('Video Stream', frame_with_rect)
            cv2.imshow('Cropped Area', cropped_frame)

            # Write cropped frame to frame buffer
            log("writing frame")
            inputBuffer.addFrame(cropped_frame) # should be inputBuffer.FixedFrameBuffer.addFrame(cropped_frame)?
            log("written frame")

        # Exit loop when 'q' is pressed
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    log("Input Stream complete")

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


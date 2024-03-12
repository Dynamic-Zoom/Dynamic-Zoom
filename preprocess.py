import cv2
import numpy as np

# Initialize global variables for the cursor position
cursor_x, cursor_y = 100, 100  # Starting position
crop_width, crop_height = 200, 200  # Size of the cropped area

# Mouse callback function to update cursor position
def update_cursor_position(event, x, y, flags, param):
    global cursor_x, cursor_y
    if event == cv2.EVENT_MOUSEMOVE:
        cursor_x, cursor_y = x, y

# Initialize video capture
# path: "C:\Users\Nicor\OneDrive\Documents\Sem 10\CS 766\project\nico's code\1996 Chicago Bulls 44th Straight Home Win.mp4"
cap = cv2.VideoCapture("1996 Chicago Bulls 44th Straight Home Win.mp4") # Replace with the path to your video file

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

    # Crop the frame
    cropped_frame = frame[y_start:y_end, x_start:x_end]

    # # Display the original frame with a rectangle showing the crop area
    # frame_with_rect = frame.copy()
    # cv2.rectangle(frame_with_rect, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
    # cv2.imshow('Video Stream', frame_with_rect)

    # # Optionally display the cropped area in a separate window
    # cv2.imshow('Cropped Area', cropped_frame)
    
     # Resize the cropped frame to match the original frame height, if necessary
    if cropped_frame.shape[0] != frame.shape[0]:
        cropped_frame = cv2.resize(cropped_frame, (int(frame.shape[0] * (cropped_frame.shape[1] / cropped_frame.shape[0])), frame.shape[0]))

    # Display the original frame with a rectangle showing the crop area
    frame_with_rect = frame.copy()
    cv2.rectangle(frame_with_rect, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    # Concatenate the original frame with the cropped frame
    combined_frame = np.hstack((frame_with_rect, cropped_frame))

    # Display the combined frame
    cv2.imshow('Video Stream', combined_frame)


    # Exit loop when 'q' is pressed
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

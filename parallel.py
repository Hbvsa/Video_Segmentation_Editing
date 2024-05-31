import gradio as gr
import cv2
import numpy as np
from src.components.model_setup import model_setup
from src.components.return_mask import return_mask
def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


    return cap, fps, width, height, fourcc

def apply_mask_to_frame_v2(frame_bgr, alpha_mask, padded_image):
    """
    Apply a mask to a frame using an alpha blend.

    Parameters:
    - frame_bgr: numpy array, the frame in BGR format
    - alpha_mask: numpy array, the alpha mask
    - padded_image: numpy array, the image to be applied where the mask is true

    Returns:
    - blended_frame: numpy array, the frame with the mask applied
    """
    alpha_frame = 1.0 - alpha_mask
    blended_frame = (alpha_frame * frame_bgr + alpha_mask * padded_image).astype(np.uint8)
    return blended_frame

video_path, ad_image_path = 'video_lacy.mp4', 'images/clix.png'

# Model load
predictor, ort_session = model_setup()

# Get video capture and properties
cap, fps, width, height, fourcc = get_video_properties(video_path)
print("fps",fps)
# To save altered video
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

# Load the ad
ad_image = cv2.imread(ad_image_path)
ret, frame = cap.read()  # read one frame
input_points = np.array([[frame.shape[1] // 3, frame.shape[0] // 10]])  # Example input point
input_labels = np.array([1])  # Positive label
mask_image, alpha_mask = return_mask(frame, predictor, ort_session, ad_image, input_points, input_labels)
import cv2
import threading
from queue import Queue

# Function to display frames from the output queue
def display_frames(output_queue, max_frames):
    global start_time  # Access the start_time variable from the outer scope
    frame_count = 0
    while True:
        frame = output_queue.get()
        if frame is None:
            break
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Calculate the time elapsed since the last frame
        time_elapsed = time.time() - start_time

        # Calculate the time needed to wait for the next frame
        target_time_per_frame = 1 / 24  # 60 fps
        time_to_wait = max(0, target_time_per_frame - time_elapsed)

        # Wait for the calculated time
        time.sleep(time_to_wait)

        # Update the start time for the next frame
        start_time = time.time()

        frame_count += 1

    cv2.destroyAllWindows()
# Define a thread class to process frames
class FrameProcessingThread(threading.Thread):
    def __init__(self, input_queue, output_queue, predictor, ort_session, ad_image, input_points, input_labels):
        super(FrameProcessingThread, self).__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.predictor = predictor
        self.ort_session = ort_session
        self.ad_image = ad_image
        self.input_points = input_points
        self.input_labels = input_labels
        self.cancel_processing = False
        self.mask_image = mask_image
        self.alpha_mask = alpha_mask

    def run(self):
        while not self.cancel_processing:
            frame = self.input_queue.get()
            if frame is None:
                break
            mask_image, alpha_mask = return_mask(frame, self.predictor, self.ort_session, self.ad_image,
                                                 self.input_points, self.input_labels)
            processed_frame = apply_mask_to_frame_v2(frame, alpha_mask, mask_image)
            self.output_queue.put(processed_frame)

# Initialize variables
max_frames = 24*10  # Example max frames
cancel_processing = False  # Example cancel flag
input_queue = Queue()
output_queue = Queue()
threads = []

# Start frame processing threads
num_threads = 1  # Example number of threads
for _ in range(num_threads):
    thread = FrameProcessingThread(input_queue, output_queue, predictor, ort_session, ad_image, input_points, input_labels)
    thread.start()

# Start a thread to display frames
display_thread = threading.Thread(target=display_frames, args=(output_queue, max_frames))
display_thread.start()

import time


start_time = time.time()
# Read and distribute frames to threads
frame_count = 0
while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    input_queue.put(frame)
    frame_count += 1

# Signal threads to finish processing
for _ in range(num_threads):
    input_queue.put(None)

# Wait for threads to finish
for thread in threads:
    thread.join()

# Wait for the display thread to finish
display_thread.join()

end_time = time.time()
execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")
# Release video capture
cap.release()

frame = output_queue.get()
cv2.imshow('Processed Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
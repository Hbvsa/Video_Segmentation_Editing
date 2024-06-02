import threading
import queue
import time
import gradio as gr
import cv2
from src.components.model_setup import model_setup
from src.components.return_mask import return_mask
import numpy as np

# Global variables for input points and labels
input_points = []
input_labels = []
cancel_processing = False
start_time = time.time()  # Initialize start time

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cap, fps, width, height, fourcc

def apply_mask_to_frame_v2(frame_bgr, alpha_mask, padded_image):
    alpha_frame = 1.0 - alpha_mask
    blended_frame = (alpha_frame * frame_bgr + alpha_mask * padded_image).astype(np.uint8)
    return blended_frame

def show_points(coords, labels, frame, marker_size=15):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    for point in pos_points:
        cv2.drawMarker(frame, (point[0], point[1]), color=(0, 255, 0), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2)
    for point in neg_points:
        cv2.drawMarker(frame, (point[0], point[1]), color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2)
    return frame

def set_input_points(points):
    global input_points
    input_points = points
    return "Input points set."

def reset_input_points():
    global input_points
    input_points = []
    return "Input points reset."

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cap, fps, width, height, fourcc

def apply_mask_to_frame_v2(frame_bgr, alpha_mask, padded_image):
    alpha_frame = 1.0 - alpha_mask
    blended_frame = (alpha_frame * frame_bgr + alpha_mask * padded_image).astype(np.uint8)
    return blended_frame

def show_points(coords, labels, frame, marker_size=15):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    for point in pos_points:
        cv2.drawMarker(frame, (point[0], point[1]), color=(0, 255, 0), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2)
    for point in neg_points:
        cv2.drawMarker(frame, (point[0], point[1]), color=(0, 0, 255), markerType=cv2.MARKER_STAR,
                       markerSize=marker_size, thickness=2)
    return frame

def set_input_points(points):
    global input_points
    input_points = points
    return "Input points set."

def reset_input_points():
    global input_points
    input_points = []
    return "Input points reset."

def visualize_initial_frame(video_path, ad_image_path):
    global input_points, input_points_array
    global input_labels

    # Model load
    predictor, ort_session = model_setup()

    # Get video capture and properties
    cap, fps, width, height, fourcc = get_video_properties(video_path)

    # Load the ad
    ad_image = cv2.imread(ad_image_path)

    # Read one frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read the first frame of the video.")

    # Display the frame for user input
    cv2.imshow('Select Points', frame)
    cv2.setMouseCallback('Select Points', record_clicks)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert points to numpy array
    input_points_array = np.array(input_points)
    input_labels = np.ones(len(input_points_array))  # Assume all points are positive

    # Apply mask to frame
    mask_image, alpha_mask = return_mask(frame, predictor, ort_session, ad_image, input_points_array, input_labels)
    frame_with_mask = apply_mask_to_frame_v2(frame, alpha_mask, mask_image)

    # Show points
    frame_with_points_and_mask = show_points(input_points_array, input_labels, frame_with_mask)

    # Release resources
    cap.release()

    return cv2.cvtColor(frame_with_points_and_mask, cv2.COLOR_BGR2RGB)

def record_clicks(event, x, y, flags, param):
    global input_points
    if event == cv2.EVENT_LBUTTONDOWN:
        input_points.append((x, y))

def run_return_mask(frame, predictor, ort_session, ad_image, input_points, input_labels):
    global mask_image, alpha_mask
    mask_image, alpha_mask = return_mask(frame, predictor, ort_session, ad_image, input_points, input_labels)
    print("ended my mask update")

def video_pipeline(video_path, ad_image_path, seg_per_sec, seconds):
    global cancel_processing, input_points, input_labels, input_points_array
    cancel_processing = False  # Reset cancellation flag at the start

    # Model load
    predictor, ort_session = model_setup()

    # Get video capture and properties
    cap, fps, width, height, fourcc = get_video_properties(video_path)
    print("FPS", fps)
    # Load the ad
    ad_image = cv2.imread(ad_image_path)

    # Writer to output frames to a video file
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    # Process each frame
    frame_count = 0
    ret, frame = cap.read()
    mask_image, alpha_mask = return_mask(frame, predictor, ort_session, ad_image, input_points_array, input_labels)
    max_frames = fps * int(seconds)
    while cap.isOpened() and frame_count < max_frames:
        if cancel_processing:
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % int(fps / float(seg_per_sec)) == 0:
            mask_image, alpha_mask = return_mask(frame, predictor, ort_session, ad_image, input_points_array,
                                                 input_labels)
        frame = apply_mask_to_frame_v2(frame, alpha_mask, mask_image)
        out.write(frame)

    print("Processed frames", frame_count)
    print("----------Ended processing------------")
    # Release resources
    cap.release()

    return 'output_video.mp4'

def cancel_video_processing():
    global cancel_processing
    cancel_processing = True
    return "Processing cancelled."

# Gradio Blocks interface
with gr.Blocks() as iface:

    seg_per_sec = gr.Textbox(label="Segmentations per sec", interactive=True)
    max_frames = gr.Textbox(label="Seconds processed", interactive=True)
    video_input = gr.Video(label="Upload Video")
    image_input = gr.Image(label="Upload Image", type="filepath")
    visualize_button = gr.Button("Visualize Initial Frame")
    output_image = gr.Image(label="Initial Frame with Points and Mask", type="numpy")
    reset_button = gr.Button("Reset Input Points")
    run_button = gr.Button("Start Processing")
    status = gr.Textbox(label="Status")
    cancel_button = gr.Button("Cancel Processing")
    output_video = gr.Video(label="Processed video")

    run_button.click(fn=video_pipeline, inputs=[video_input, image_input, seg_per_sec, max_frames], outputs=output_video)
    cancel_button.click(fn=cancel_video_processing, inputs=[], outputs=status)
    visualize_button.click(fn=visualize_initial_frame, inputs=[video_input, image_input], outputs=output_image)
    reset_button.click(fn=reset_input_points, inputs=[], outputs=status)

# Initialize the output queue
output_queue = queue.Queue()
iface.launch()

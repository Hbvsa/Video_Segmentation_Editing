import cv2
import numpy as np
from src.components.model_setup import model_setup
from src.components.return_mask import return_mask
import matplotlib.pyplot as plt
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
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def video_pipeline(video_path, ad_image_path):

    # Model load
    predictor, ort_session = model_setup()

    # Get video capture and properties
    cap, fps, width, height, fourcc = get_video_properties(video_path)

    #To save altered video
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

    # Load the ad
    ad_image = cv2.imread(ad_image_path)

    #Visualize before executing
    fig, ax = plt.subplots()
    ret, frame = cap.read() #read one frame

    # Input prompt to choose segmentation location
    input_point = np.array([[frame.shape[1] // 3, frame.shape[0] // 10]]) # Example input point
    input_label = np.array([1]) #Positive label

    #Show points and mask
    show_points(input_point, input_label, plt.gca())
    mask_image, alpha_mask = return_mask(frame, predictor, ort_session, ad_image, input_point, input_label)
    frame = apply_mask_to_frame_v2(frame, alpha_mask, mask_image)
    ax.imshow(frame)

    #Limit for testing
    frame_count = 0
    max_frames = 60*10

    # Process each frame
    while cap.isOpened() and frame_count < max_frames:

        ret, frame = cap.read()
        mask_image, alpha_mask = return_mask(frame, predictor, ort_session, ad_image, input_point, input_label )
        frame = apply_mask_to_frame_v2(frame, alpha_mask, mask_image)

        cv2.imshow('Processed Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()



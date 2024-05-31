import numpy as np
import cv2
import skimage



def input_with_points(input_point, input_label, predictor, image, image_embedding):


    # Transform inputs
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }

    return ort_inputs

def create_masked_image(mask, image):
    mask = mask.reshape(mask.shape[-2], mask.shape[-1])
    mask = mask.astype(bool)
    mask_int = mask.astype(np.uint8)
    from skimage import measure
    # Find the bounding box of the masked region
    bbox = measure.regionprops(mask_int)[0].bbox
    min_row, min_col, max_row, max_col = bbox

    from skimage.transform import resize
    # Resize the image to match the Mask dimensions where it is True
    resized_image_float = resize(image, (max_row - min_row, max_col - min_col), anti_aliasing=True)
    resized_image = (resized_image_float * 255).astype(np.uint8)

    # Pad the resized image to match the size of the full original mask
    padded_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    padded_image[min_row:max_row, min_col:max_col, :] = resized_image

    return padded_image

def return_mask(frame, predictor, ort_session, ad_image, input_point, input_label):

    # Convert frame from BGR (OpenCV format) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Frame embedding (encoder encoding the image)
    predictor.set_image(frame_rgb)
    frame_embedding = predictor.get_image_embedding().cpu().numpy()

    #Function which returns the inputs ready for the model
    ort_inputs = input_with_points(input_point, input_label, predictor, frame, frame_embedding)

    # Get the mask using the segmentation model which takes the points as input prompt
    masks, _, what = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold


    #Resize the ad image to match mask size and create an image with just the ad on the mask
    mask_image = create_masked_image(masks, ad_image)
    #The alpha mask indicates where the ad image should overlap the original image
    alpha_mask = np.squeeze(masks)
    alpha_mask = alpha_mask[..., np.newaxis]

    return mask_image, alpha_mask
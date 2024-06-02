import warnings
import yaml
from onnxruntime.quantization import quantize_dynamic
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import torch
from onnxruntime.quantization import QuantType
import onnxruntime
from pathlib import Path
import os

def load_sam():
    with open('./src/components/config_model.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    checkpoint = config["checkpoint"]
    model_type = config["model_type"]

    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint):
        # Download the checkpoint file using wget
        os.system(f'wget -O {checkpoint} https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return sam
def initialize_predictor(sam):
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    return predictor

def create_onnx_model(sam):
    onnx_model_path = "./src/models/sam_onnx_example.onnx"
    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        return onnx_model_path

def quantize_model(onnx_model_path):

    onnx_model_quantized_path = "./src/models/sam_onnx_quantized_example.onnx"
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=onnx_model_quantized_path,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )
    return onnx_model_quantized_path

def model_setup():

    quantized_model_path = r'./src/models/sam_onnx_quantized_example.onnx'
    sam = load_sam()
    predictor = initialize_predictor(sam)

    # Create quantized onnx model if not already created
    if not Path(quantized_model_path).exists():
        onnx_model_path = create_onnx_model(sam)
        quantized_model_path = quantize_model(onnx_model_path)

    # Prepare for inference
    ort_session = onnxruntime.InferenceSession(quantized_model_path)

    return predictor, ort_session

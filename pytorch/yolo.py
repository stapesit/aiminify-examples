import torch
from ultralytics import YOLO
from ultralytics.engine.exporter import Exporter
from onnxruntime.quantization import quantize_dynamic, QuantType

from aiminify import minify
from aiminify.minify_pytorch.utils import get_flops


def quantize_int8(model: torch.nn.Module) -> torch.nn.Module:
    """
    Applies static INT8 quantization to a YOLOv8 model.

    Args:
        model (torch.nn.Module): The YOLOv8 model to quantize.

    Returns:
        torch.nn.Module: The INT8 quantized model.
    """
    model.eval()
    model.cpu()

    custom = {
        "imgsz": model.args["imgsz"],
        "batch": 1,
        "data": None,
        "device": None,  # reset to avoid multi-GPU errors
        "verbose": False,
        "format": "onnx",
        "simplify": True,
        "opset": 13,
        "int8": True,
        "mode": "export",
    }

    onnx_model = Exporter(overrides=custom, )(model=model)

    quantize_dynamic(
        model_input=onnx_model,
        model_output='model_int8.onnx',
        weight_type=QuantType.QInt8  # or QuantType.QUInt8
    )

    return 'model_int8.onnx'


if __name__ == "__main__":
    # Try any of the following
    model = YOLO('yolov8s-seg.pt').model.to('cpu')
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to('cpu')
    # model = YOLO('yolov5s.pt').model.to('cpu')

    rand_input = torch.randn(1, 3, 640, 640)

    normal_flops = get_flops(model, rand_input)
    normal_params = sum(m.numel() for m in model.parameters())

    compressed_model, _ = minify(
        model,
        compression_strength=5,  # From 0 (no compression) to 5 (strongest)

        # Everything below is optional
        fine_tune=False,
        quantization=False,
        training_generator=None,
        validation_generator=None,
        loss_function=None,
        optimizer=None,
        precision='mixed',
        accumulation_steps=1,
        debug_mode=False,
    )

    compressed_flops = get_flops(compressed_model, rand_input)
    compressed_params = sum(m.numel() for m in compressed_model.parameters())

    print(f'Normal flops: {normal_flops:,}')
    print(f'Compressed flops: {compressed_flops:,}')

    print(f'Normal params: {normal_params:,}')
    print(f'Compressed params: {compressed_params:,}')

    int8_model = quantize_int8(compressed_model)

import copy

from ultralytics import YOLO
import torch

from aiminify import minify
from aiminify.minify_pytorch.utils import get_flops

# Try any of the following
# model = YOLO('yolov5s.pt').model.to('cpu')
# model = YOLO('yolov8s.pt').model.to('cpu')
model = YOLO('yolo11s.pt').model.to('cpu')

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

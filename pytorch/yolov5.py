import copy

from ultralytics import YOLO
import torch

from aiminify import minify
from aiminify.minify_pytorch.utils import get_flops

model = YOLO('yolov5s.pt').model.to('cpu')

rand_input = torch.randn(1, 3, 640, 640)

normal_flops = get_flops(model, rand_input)
normal_params = sum(m.numel() for m in model.parameters())

compressed_model, _ = minify(
    model,
    compression_strength=5,
    fine_tune=False,
    debug_mode=False,
)

compressed_flops = get_flops(compressed_model, rand_input)
compressed_params = sum(m.numel() for m in compressed_model.parameters())

print(f'Normal flops: {normal_flops:,}')
print(f'Compressed flops: {compressed_flops:,}')

print(f'Normal params: {normal_params:,}')
print(f'Compressed params: {compressed_params:,}')

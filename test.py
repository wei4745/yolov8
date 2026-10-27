import torch
print("torch版本：", torch.__version__)
print("支持最大opset：", torch.onnx.operators.version())

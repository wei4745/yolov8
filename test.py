import torch
from ultralytics.utils.metrics import bbox_iou

box1 = torch.tensor([[10, 10, 50, 50]], dtype=torch.float32)  # xyxy
box2 = torch.tensor([[15, 15, 55, 55]], dtype=torch.float32)

iou_ciou = bbox_iou(box1, box2, xywh=False, CIoU=True)
iou_siou = bbox_iou(box1, box2, xywh=False, SIoU=True)

print(f"CIoU: {iou_ciou.item():.6f}")
print(f"SIoU: {iou_siou.item():.6f}")
# 两者数值应不同（SIoU 会额外减去角度+距离+形状代价）
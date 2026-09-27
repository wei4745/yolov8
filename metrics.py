from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
model.fuse()

# 1. 参数量 & FLOPs
print("=" * 40)
print("模型复杂度：")
model.info()

# 2. 验证集精度 + 速度
print("\n验证集性能：")
metrics = model.val(batch=1, plots=False)

print(f"mAP50-95 : {metrics.box.map:.4f}")
print(f"mAP50    : {metrics.box.map50:.4f}")
print(f"推理时间 : {metrics.speed['inference']:.2f} ms")
print(f"总耗时   : {metrics.speed['total']:.2f} ms")
print(f"帧率 FPS : {1000 / metrics.speed['inference']:.1f}")

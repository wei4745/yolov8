from ultralytics import YOLO

# 加载训练好的模型（通常是 best.pt）
model = YOLO("runs/detect/train/weights/best.pt")  # 改成你的路径

# 在验证集上验证（会自动使用训练时的 data.yaml）
metrics = model.val()  # 默认用 val 集
# 或者显式指定
# metrics = model.val(data="your_data.yaml", split="val")  # 或 split="test"

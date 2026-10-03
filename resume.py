from ultralytics import YOLO

# 加载上次中断保存的 last.pt
model = YOLO("/kaggle/working/runs/yolov8s_p2_ema_256e/weights/last.pt")

# resume=True 开启断点续训
# 不要写 data、epochs、batch等大部分参数！会自动从last.pt读取
results = model.train(resume=True)

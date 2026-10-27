from ultralytics import YOLO

def export():
    # 加载预训练yolov8n，自动下载yolov8n.pt
    model = YOLO("runs/detect/runs/blueberry/yolov8n_nwdc10point5_300e/weights/best.pt")

    # 导出参数配置
    export_result = model.export(
        format="rknn",  # 可选: onnx, torchscript, tflite, coreml

        opset=11,  # ONNX opset版本，RKNN推荐17

    )

    print("✅ 模型导出完成！")
    print(f"输出路径: {export_result}")

if __name__ == "__main__":
    export()

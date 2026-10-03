from ultralytics import YOLO

# 统一超参数
# 放在ultralytics同级目录
MODEL_CFG = "ultralytics/cfg/models/v8/yolov8s-p2-mobileone.yaml"  # 不加载预训练权重
NAME = "yolov8s_p2_mobileone_test"  # 改名字区分

PROJECT = "runs/blueberry"
DATA_YAML = "ultralytics/cfg/datasets/blueberry_local.yaml"
EPOCHS = 256
IMGSZ = 1024
BATCH = 2
DEVICE = "cpu"  # Tesla T4 单卡
WORKERS = 0


def main():
    model = YOLO(MODEL_CFG)
    criterion = model.model.init_criterion()

    print("=" * 50)
    print("NWD 启用状态检查：")
    print(f"nwd_loss   : {criterion.bbox_loss.nwd_loss}")
    print(f"iou_ratio  : {criterion.bbox_loss.iou_ratio}")
    print(f"constant   : {criterion.bbox_loss.constant}")
    print("=" * 50)

    # 开始训练
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        pretrained=False,  # 明确不使用预训练
        optimizer="SGD",  # 统一优化器
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率 = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        hsv_h=0.015,  # 数据增强（小目标可适当减弱）
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,  # 小目标建议关闭大幅旋转
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,  # 小目标建议关闭 mixup
        copy_paste=0.0,
        close_mosaic=15,  # 关闭 mosaic
        patience=50,  # early stopping
        save=True,
        save_period=-1,  # 只保存 best 和 last
        exist_ok=True,
        verbose=True,
        seed=17,  # 固定随机种子，方便复现
        deterministic=True,

    )


if __name__ == "__main__":
    main()

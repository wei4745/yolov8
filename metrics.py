import argparse

import torch
from thop import profile

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="模型路径，如 best.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.path

    model = YOLO(model_path)
    # 只 fuse 一次，融合Conv+BN
    model.fuse()
    model.model.eval()

    # 获取模型所在设备
    device = next(model.model.parameters()).device
    print(f"模型运行设备: {device}")

    print("=" * 40)
    print("模型复杂度(model.info):")
    model.info()

    print("\n测试集性能：")
    # 注意：数据集yaml必须有test字段，没有请改成 split="val"
    metrics = model.val(batch=1, plots=False, imgsz=1024, split="test")

    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"预处理   : {metrics.speed['preprocess']:.2f} ms")
    print(f"推理时间 : {metrics.speed['inference']:.2f} ms")
    print(f"后处理   : {metrics.speed['postprocess']:.2f} ms")

    total_ms = metrics.speed["preprocess"] + metrics.speed["inference"] + metrics.speed["postprocess"]
    print(f"单图总耗时 : {total_ms:.2f} ms")
    # 真实FPS：包含预处理推理后处理
    print(f"FPS: {1000 / total_ms:.1f}")

    # 构造和模型同设备的dummy输入
    dummy_input = torch.randn(1, 3, 1024, 1024).to(device)
    macs, params = profile(model.model, inputs=(dummy_input,), verbose=False)

    print("\n==== thop计算结果 ====")
    print(f"参数量 Params: {params / 1e6:.2f} M")
    print(f"MACs: {macs / 1e9:.2f} GMac")
    print(f"FLOPs(≈2×MACs): {2 * macs / 1e9:.2f} GFLOPs")

    mp = metrics.box.mp.item()
    mr = metrics.box.mr.item()
    print(f"\n平均精度 mp(Precision): {mp:.4f}")
    print(f"平均召回 mr(Recall): {mr:.4f}")

    print("\n每一类别的mAP50‑95：")
    for idx, ap in enumerate(metrics.box.maps.tolist()):
        print(f"类别{idx:2d} mAP50‑95: {ap:.4f}")


if __name__ == "__main__":
    main()

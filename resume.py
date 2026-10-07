import argparse

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="模型路径")

    return parser.parse_args()


def main():
    args = parse_args()
    # 加载上次中断保存的 last.pt
    model_path = args.path
    model = YOLO(model_path)

    # resume=True 开启断点续训
    # 不要写 data、epochs、batch等大部分参数！会自动从last.pt读取
    results = model.train(resume=True)


if __name__ == "__main__":
    main()

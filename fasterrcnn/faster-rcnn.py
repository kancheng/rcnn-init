import torch
import torchvision
import argparse
import cv2
import numpy as np
import sys
sys.path.append('./')
import coco_names
import random
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')
    parser.add_argument('image_path', type=str, help='image path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
    parser.add_argument('--output', type=str, default='output_image.jpg', help='output image file path')
    args = parser.parse_args()
    return args

def random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def main():
    args = get_args()
    num_classes = 91
    names = coco_names.names  # Ensure coco_names.names is a dictionary

    # 创建模型
    print("Creating model")
    # model = torchvision.models.detection.__dict__[args.model](pretrained=True)
    model = torchvision.models.detection.__dict__[args.model](
        num_classes=num_classes,
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1  # 或者使用 `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT`
    )
    # model = model.cuda()
    model = model.cpu()
    model.eval()

    # 加载和预处理图片
    src_img = cv2.imread(args.image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    # img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().unsqueeze(0).cuda()
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().unsqueeze(0).cpu()

    # 模型推理
    with torch.no_grad():
        outputs = model(img_tensor)

    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    # 绘制检测框
    for idx in range(len(boxes)):
        if scores[idx] >= args.score:
            x1, y1, x2, y2 = boxes[idx]
            x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
            name = names.get(str(labels[idx].item()), "Unknown")

            cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
            cv2.putText(src_img, text=name, org=(x1, y1 + 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    cv2.imwrite(args.output, cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR))  # 保存为BGR格式
    # 显示结果
    cv2.imshow('Detection Result', src_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

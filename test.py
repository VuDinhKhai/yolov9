import time
import cv2
import argparse
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

cap = cv2.VideoCapture("rtsp://admin:admin-12345@192.168.3.15:10555//Streaming/Channels/101")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights= "yolov9s.pt"
dnn=False
data="data/coco.yaml"
half=False
imgsz=(640, 640)
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size
img_size=640
conf_thres=0.25
iou_thres=0.45
classes=None
agnostic_nms=False
max_det=1000
stride=32
auto=True
augment=False,  # augmented inference
visualize=False,
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def detect_yolov9(frame):
    im = letterbox(frame, img_size, stride, auto)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)  # expand for batch dim
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(im, augment=False, visualize=False)
    im0 = frame.copy()
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    for i, det in enumerate(pred):  # per image
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = '%s %.2f' % (model.names[int(cls)], conf)
                x1, y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                annotator.box_label(xyxy, label, color=colors(100, True))

        # Stream results
        im0 = annotator.result()
    return im0
@smart_inference_mode()
def main():
    # Mở camera (0 là chỉ số của camera mặc định)
    if not cap.isOpened():
        print("Không thể mở camera.")
        return
    count = 0
    while True:
        # Đọc khung hình từ camera
        ret, frame = cap.read()

        if not ret:
            print("Không thể đọc khung hình.")
            break
        if count % 3 == 0:
            to = time.time()
            frame = detect_yolov9(frame)
            print(time.time() - to)
            # Hiển thị khung hình
            frame = cv2.resize(frame,(1280,720))
            cv2.imshow('Camera Feed', frame)
        count +=1
        # Thoát khi nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên và đóng tất cả các cửa sổ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

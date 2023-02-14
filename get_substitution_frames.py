# 截取视频中的特定帧并保存
import os

import cv2 as cv
import numpy as np
import time
from collections import defaultdict
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import torch
from utils.general import non_max_suppression
from models.experimental import attempt_load
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from numpy import random
import argparse
from utils.plots import plot_one_box
from utils.datasets import letterbox


weights = 'runs/train/exp15/weights/best.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dnn = False

imgsz = 640
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
# imgsz = check_img_size(imgsz, s=stride)  # check img_size

# trace = False
# if trace:
#     model = TracedModel(model, device, opt.img_size)

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
old_img_w = old_img_h = imgsz
old_img_b = 1





BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


EVENT_DIR = "D:/dataset/event"
# VIDEO_DIR = "D:/dataset/MAKE_OCR_DATA/Small_PadOCR_videos"
# Output_big_frames_path = "D:/dataset/MAKE_OCR_DATA/small_frames"

VIDEO_DIR = "D:/dataset/temp_videos"
Output_big_frames_path = "D:/dataset/substitution_image/result_substitution"
makedir(Output_big_frames_path)


def get_substitution_index(VIDEO_DIR):
    for root, _, files in os.walk(VIDEO_DIR):
        for video_index in files:
            time1 = time.time()
            video_index1 = video_index.split('_')

            if video_index1[1] not in ['0059', '0060', '0061', '0062',
                                       '0063', '0064', '0065', '0066', '0067', '0068', '0069', '0070',
                                       '0071', '0072', '0073', '1040', '1041', '1045', '1046', '1047', '1048',
                                       '1049', '1050', '1051', '1052', '1054',
                                       '1055', '1056', '1057', '1058', '1059', '1171',
                                       '1212', '1216', '1218', '1221', '1223', '1224', '1225',
                                       '1226', '1228', '1230', '1231', '1233', '1236', '1237', '1238', '1239', '1242',
                                       '1243', '1244', '1245']:
                score_dic = {}
                score_record = []
                player_score_dic = {}
                player_score_record = []
                videoCap = cv.VideoCapture(VIDEO_DIR + "\\" + video_index)

                frame_height = videoCap.get(cv.CAP_PROP_FRAME_HEIGHT)
                frame_width = videoCap.get(cv.CAP_PROP_FRAME_WIDTH)
                fps = int(videoCap.get(cv.CAP_PROP_FPS))

                # big_h_index = int(frame_height * 6 / 10)
                big_h_index = int(frame_height * 2 / 10)

                big_w_index = int(frame_width / 2)
                x1_big_center = big_w_index - 50
                x2_big_center = big_w_index + 50

                frame_count = videoCap.get(cv.CAP_PROP_FRAME_COUNT)
                print("-----------------------------")
                print("video:{}".format(video_index1[1]))


                # i = frame_count - 20000
                # i = 147503
                # i_index = [58620, 58750]
                i = 0
                i = 19435

                not_found_consecutive_nums = 0
                found_consecutive_nums = 0
                lost_nums = 0
                substitution_frame_index = []
                # for i in range(i_index[0], i_index[1]):
                while i < frame_count:
                    # print("deal with {}".format(i))
                    videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
                    # i += 100;
                    boolFrame, matFrame = videoCap.read()

                    substitution_flag = False
                    # 不截取就保存
                    if boolFrame:

                        find_flag = False
                        temp_jpgframe = np.asarray(matFrame)


                        # Padded resize
                        img = letterbox(matFrame, imgsz, stride)[0]

                        # Convert
                        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                        img = np.ascontiguousarray(img)

                        img = torch.from_numpy(img.copy()).to(device)
                        img = img.float()
                        img /= 255.0  # 0 - 255 to 0.0 - 1.0
                        if img.ndimension() == 3:
                            img = img.unsqueeze(0)


                        # Inference
                        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                            pred = model(img, augment=opt.augment)[0]

                        # Apply NMS
                        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                                   agnostic=opt.agnostic_nms)

                        # Process detections
                        for k, det in enumerate(pred):  # detections per image
                            if len(det):
                                # p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                                s = ''

                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], temp_jpgframe.shape).round()



                                # Write results
                                for *xyxy, conf, cls in reversed(det):

                                    if conf >= 0.60:
                                        substitution_flag = True

                                        substitution_frame_index.append(i)
                                        # Print results
                                        for c in det[:, -1].unique():
                                            n = (det[:, -1] == c).sum()  # detections per class
                                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                                        label = f'{names[int(cls)]} {conf:.2f}'
                                        plot_one_box(xyxy, temp_jpgframe, label=label, color=colors[int(cls)], line_thickness=1)

                                if substitution_flag:
                                    print(s)
                                    save_img_path = os.path.join(Output_big_frames_path,video_index1[1] + "_" + str(i) + ".jpg")
                                    cv.imwrite(save_img_path, temp_jpgframe)
                                    print("save {} *************".format(i))

                    if substitution_flag:
                        i += 1
                        lost_nums = 0
                    else:
                        if 6 < lost_nums < 888:
                            # i = min(i + 10000, frame_count - 5000)
                            i = i+fps*2
                            lost_nums = 999
                        elif lost_nums > 888:
                            # i += 5
                            i += 10
                        else:
                            i += 5
                            lost_nums += 1



                time2 = time.time()
                print("finish video:{}  , cost {} s".format(video_index1[1], time2 - time1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp15/weights/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='D:/dataset/video_5/soccer_1314_video.mp4',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    get_substitution_index(VIDEO_DIR)
    # test_playback_algo()
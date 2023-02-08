# 使用yolov5对视频中的 比分大字幕 进行检测，并将帧保存
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
from models.common import DetectMultiBackend
from utils.augmentations import letterbox

weights = 'runs/train/exp11/weights/best.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dnn = False

model = DetectMultiBackend(weights, device=device)  # local model
imgsz = [640, 640]
model.warmup(imgsz=(1, 3, *imgsz))  # warmup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


EVENT_DIR = "D:/dataset/event"
# VIDEO_DIR = "D:/dataset/MAKE_OCR_DATA/Small_PadOCR_videos"
# Output_big_frames_path = "D:/dataset/MAKE_OCR_DATA/small_frames"

VIDEO_DIR = "D:/dataset/label_videos"
Output_big_frames_path = "D:/dataset/yolo_images"
makedir(Output_big_frames_path)

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

            lost_nums = 0
            miss_times = 0
            # for i in range(i_index[0], i_index[1]):
            while i < frame_count:
                videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
                # i += 100;
                boolFrame, matFrame = videoCap.read()

                # 截取后再保存
                # if boolFrame:
                #     temp_jpgframe = np.asarray(matFrame)
                #     # 截取上面0-90区域进行OCR检测
                #     if big_h_index < 300:
                #         jpgframe = temp_jpgframe[0:big_h_index]
                #     else:
                #         jpgframe = temp_jpgframe[big_h_index:]
                #
                #     save_img_path = os.path.join(Output_big_frames_path,
                #                                  video_index1[1] + "_" + str(i) + ".jpg")
                #     cv.imwrite(save_img_path, jpgframe)

                big_flag = False
                # 不截取就保存
                if boolFrame:

                    temp_jpgframe = np.asarray(matFrame)

                    img = letterbox(matFrame, 640, stride=32, auto=True)[0]

                    # Convert
                    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    img = np.ascontiguousarray(img)

                    im = torch.from_numpy(img).to(device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    result = model(im)
                    result = non_max_suppression(result, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)
                    # print("{}".format(i))
                    for k, det in enumerate(result):
                        if len(det):
                            # print("deal with {}".format(i))
                            for *xyxy, conf, cls in reversed(det):
                    # a = result.pandas().xyxy.confidence
                                if conf >= 0.50:
                                    big_flag = True
                                    save_img_path = os.path.join(Output_big_frames_path,
                                                                 video_index1[1] + "_" + str(i) + ".jpg")
                                    print("save {} *************".format(i))

                                    cv.imwrite(save_img_path, temp_jpgframe)
                if big_flag:
                    i += 2
                    lost_nums = 0
                else:
                    if 10 < lost_nums < 888:
                        # i = min(i + 10000, frame_count - 5000)
                        i = i+200
                        lost_nums = 999
                    elif lost_nums > 888:
                        i += 50
                    else:
                        i += 2
                        lost_nums += 1

            time2 = time.time()
            print("finish video:{}  , cost {} s".format(video_index1[1], time2 - time1))


            # print("save images from {} to {},  totoal {} frames".format(i_index[0], i_index[1], i_index[1] - i_index[0]))
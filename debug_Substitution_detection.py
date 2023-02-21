# coding=utf-8
# 替换事件检测
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
import sys


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


EVENT_DIR = "D:/dataset/event"
# VIDEO_DIR = "D:/dataset/MAKE_OCR_DATA/Small_PadOCR_videos"
# Output_big_frames_path = "D:/dataset/MAKE_OCR_DATA/small_frames"

VIDEO_DIR = "D:/dataset/temp_videos"
Output_big_frames_path = "D:/dataset/A_graduate_experiment/sub/debug_sub_frames_soccernet_2-15"
raw_Output_big_frames_path = "D:/dataset/A_graduate_experiment/sub/debug_raw_sub_frames_soccernet_2-15"
wrong_Output_big_frames_path = "D:/dataset/A_graduate_experiment/sub/debug_wrong_sub_frames_soccernet_2-15"

makedir(Output_big_frames_path)
makedir(raw_Output_big_frames_path)
makedir(wrong_Output_big_frames_path)


VIDEO_DIR = "D:/dataset/SoccerNet/SoccerNet_test_hq/"

import logging



def get_substitution_index(VIDEO_DIR):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    imgsz = 640
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    for root, matches, files in os.walk(VIDEO_DIR):
        for match in matches:
            # if match != "germany_bundesliga":
            #     continue
            match_path = os.path.join(VIDEO_DIR, match)
            for match_root, years, ffiles in os.walk(match_path):
                for year in years:
                    year_path = os.path.join(match_path, year)
                    for year_root, games, fffiles in os.walk(year_path):
                        for game in games:
                            game_path = os.path.join(year_path, game)
                            game_split = game.split(' ')
                            for game_root, _, videos in os.walk(game_path):
                                for video_index in videos:
                                    video_index1 = video_index.split('_')
                                    if not video_index.endswith('mkv'):
                                        continue
                                    time1 = time.time()
                                    video_name = match + ' ' + game_split[0] + ' ' + game_split[-1] + ' ' + \
                                                 video_index1[0]

                                    if video_name != "italy_serie-a 2016-08-27 Milan 2":
                                        continue

                                    video_path = os.path.join(game_path, video_index)
                                    videoCap = cv.VideoCapture(video_path)

                                    frame_height = videoCap.get(cv.CAP_PROP_FRAME_HEIGHT)
                                    frame_width = videoCap.get(cv.CAP_PROP_FRAME_WIDTH)
                                    frame_count = videoCap.get(cv.CAP_PROP_FRAME_COUNT)
                                    fps = int(videoCap.get(cv.CAP_PROP_FPS))

                                    logger.info("  ")
                                    logger.info(
                                        "--------------------------------------------------------------------------------------------------------------")
                                    logger.info("video:{}".format(video_name))
                                    logger.info(
                                        "视频属性：   总帧数：{}   FPS:{}   width:{}   height:{}".format(frame_count, fps,
                                                                                                frame_width,
                                                                                                frame_height))

                                    # big_h_index = int(frame_height * 6 / 10)
                                    big_h_index = int(frame_height * 2 / 10)

                                    big_w_index = int(frame_width / 2)
                                    x1_big_center = big_w_index - 50
                                    x2_big_center = big_w_index + 50



                                    labels = json.load(open(os.path.join(game_path, "Labels-v2.json")))
                                    gt_sub_frame_indexes = []
                                    sub_frame_indexes = []
                                    for annotation in labels["annotations"]:
                                        if annotation["label"] == "Substitution" and annotation["gameTime"][0] == video_index[0]:
                                            label_time = annotation["gameTime"]
                                            goal_minutes = int(label_time[-5:-3])
                                            goal_seconds = int(label_time[-2::])
                                            gt_frame = fps * (goal_seconds + 60 * goal_minutes)
                                            gt_sub_frame_indexes.append(gt_frame)
                                    logger.info("真实替换时间（帧号）：{}".format(gt_sub_frame_indexes))
                                    if not gt_sub_frame_indexes:
                                        continue




                                    # **跳帧算法**
                                    # 检测数阈值：表示确定检测到某个事件所需要的多帧检测次数
                                    # 在大字幕检测（进球得分事件检测）等事件中，因为信息会动态变化，此时这个阈值可以定为MAX_INT，即检测到字幕彻底消失为止
                                    # 在红黄牌事件或换人事件中，可设置阈值以降低检测成本
                                    find_nums_thresh = 12

                                    # 未检测阈值：表示没有检测到某个事件后，确定可以恢复常态化检测状态，所需要的多帧检测次数
                                    notfind_nums_thresh = 5

                                    find_nums = 0
                                    notfind_nums = 0
                                    find_flag = False
                                    MAX_INT = sys.maxsize

                                    # i = frame_count - 20000
                                    # i_index = [58620, 58750]
                                    i = 0
                                    i = 73574
                                    # i = int(frame_count * (7 / 10))


                                    substitution_frame_index = []
                                    while i < frame_count:
                                        logger.info("deal with {} : {}".format(video_name, i))
                                        videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
                                        # i += 100;
                                        boolFrame, matFrame = videoCap.read()

                                        find_flag = False
                                        # 不截取就保存
                                        if boolFrame:

                                            temp_jpgframe = np.asarray(matFrame)
                                            raw_jpg = temp_jpgframe.copy()

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
                                                            find_flag = True

                                                            # substitution_frame_index.append(i)
                                                            # logger.info results
                                                            for c in det[:, -1].unique():
                                                                n = (det[:, -1] == c).sum()  # detections per class
                                                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                                                            label = f'{names[int(cls)]} {conf:.2f}'
                                                            plot_one_box(xyxy, temp_jpgframe, label=label, color=colors[int(cls)], line_thickness=1)

                                                    # if find_flag:
                                                    #     save_img_path = os.path.join(Output_big_frames_path,video_index1[1] + "_" + str(i) + ".jpg")
                                                    #     cv.imwrite(save_img_path, temp_jpgframe)
                                                    #     logger.info("save:{} , result:{}*************".format(i, s))

                                            # 很久没有检测到-->检测到
                                            if find_flag and find_nums == 0 and notfind_nums == MAX_INT:
                                                # 检测数+1，未检测数归零
                                                find_nums += 1
                                                notfind_nums = 0
                                                # 检测到则连续检测
                                                i += 1

                                            # 并不是很久没有检测到，中间有几帧没有检测到，现在又检测到了
                                            elif find_flag:
                                                # 检测数+1，未检测数归零
                                                find_nums += 1
                                                notfind_nums = 0
                                                # flag需要及时设置为False，不然会一直卡在这里
                                                find_flag = False

                                                if find_nums == 3:
                                                    sub_frame_indexes.append(i)
                                                    save_img_path = os.path.join(Output_big_frames_path,
                                                                                 video_name + "_" + str(i) + ".jpg")
                                                    cv.imwrite(save_img_path, temp_jpgframe)
                                                    logger.info("save:{} , result:{}".format(i, s))

                                                    save_raw_img_path = os.path.join(raw_Output_big_frames_path,
                                                                                 video_name + "_" + str(i) + ".jpg")
                                                    cv.imwrite(save_raw_img_path, raw_jpg)
                                                    # logger.info("save:{} , result:{}".format(i, s))

                                                # 如果检测数未达到阈值，则继续连续检测
                                                if find_nums < find_nums_thresh:
                                                    i += 2
                                                # 如果检测数已达到阈值，可以确定发生了某个事件，则进行一长段跳帧，即5s的长度（可变）
                                                # 5s的长度，一般可以保证已检测到的信息的字幕消失
                                                # 如果有连续的包含新信息的字幕出现，5s也不会完全跳过
                                                else:
                                                    i += 5 * fps
                                                    find_nums = 0


                                            # 本次没有检测到
                                            else:
                                                # 只给未检测数加1，暂时不把检测数归零
                                                if notfind_nums != MAX_INT:
                                                    notfind_nums += 1

                                                # 常态化检测
                                                # 如果是未检测数为极大，则处于常态化检测中，每隔视频的1s（即fps帧）检测一次
                                                # 1s检测1次，属于很细粒度的检测了，因为字幕一般出现时间不会小于3s，可根据情况加大这个间隔
                                                if notfind_nums == MAX_INT:
                                                    i += 2 * fps

                                                # 如果未达到未检测数阈值，则不能完全确定字幕已经消失
                                                # 可能是画质等因素影响检测结果，那么每隔2帧（可变）再检测一次
                                                # 每隔2帧是为了降低少数帧画质问题带来的影响，同时能保证不跳过太多信息
                                                elif notfind_nums <= notfind_nums_thresh:
                                                    i += 2

                                                # 此时未检测数已经达到阈值，说明字幕已经消失，本次字幕的检测结束
                                                # 此时可进行一段跳帧，未避免字幕连续出现，只跳2s（可变），可根据情况加大这个间隔
                                                # 将未检测数设置为极大，进入常态化检测
                                                # 此时检测数也可以归零
                                                else:
                                                    i += 100 * fps
                                                    notfind_nums = MAX_INT
                                                    find_nums = 0
                                        else:
                                            i += 1

                                    right_sub_result = []
                                    rest_gt_sub = gt_sub_frame_indexes.copy()
                                    for gt_idx in gt_sub_frame_indexes:
                                        for idx in sub_frame_indexes:
                                            if abs(idx - gt_idx) < fps * 60 * 2:
                                                right_sub_result.append(str(gt_idx) + " -> " + str(idx))
                                                if gt_idx in rest_gt_sub:
                                                    rest_gt_sub.remove(gt_idx)
                                            else:
                                                save_wrong_img_path = os.path.join(wrong_Output_big_frames_path,
                                                                                 video_name + "_" + str(i) + ".jpg")
                                                cv.imwrite(save_wrong_img_path, raw_jpg)

                                    logger.info("*********************************")
                                    logger.info("本视频总结：")
                                    logger.info("检测正确结果：{}".format(right_sub_result))
                                    logger.info(
                                        "GT : {}， 共找到：{}， 其中正确的：{}".format(len(gt_sub_frame_indexes), len(sub_frame_indexes),
                                                                           len(right_sub_result)))
                                    logger.info("没被找到的：{}".format(rest_gt_sub))


                                    time2 = time.time()
                                    logger.info("finish video:{}  , cost {} s".format(video_index1[1], time2 - time1))


if __name__ == '__main__':

    logger = logging.getLogger('sub')
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s : [line:%(lineno)d] - %(message)s")

    file_handler = logging.FileHandler('log/sub/debug_Substitution_detection2-15.log')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

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
    logger.info(opt)
    get_substitution_index(VIDEO_DIR)
    # test_playback_algo()
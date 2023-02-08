# 使用yolov5对视频中的 logo帧 进行检测，并根据配对生成回放镜头的区间，保存为json
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
from tqdm import tqdm

weights = 'runs/train/exp24/weights/best.pt'
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

VIDEO_DIR = "D:/dataset/test_video"
Output_big_frames_path = "D:/dataset/yolo_logo_18Russia_images_s"
makedir(Output_big_frames_path)
Json_18Russia_path = os.path.join(BASE_DIR, "18Russia_json_s/")
makedir(Json_18Russia_path)

def test_playback_algo():
    fps = 25
    with open('1337_logo_frame_index' + '.json', 'r') as fd:
        logo_data = json.load(fd)
    fd.close()

    last = logo_data[0]
    start = logo_data[0]
    logo_blocks = []
    count = 0
    for index in logo_data:
        if index - last <= 5:
            last = index
            count += 1
        else:
            if count >= 4:
                logo_blocks.append(int((start + last)/2))

            start = index
            last = index
            count = 0

    playback = []
    ii = 0
    while ii < len(logo_blocks) - 2:
        if (logo_blocks[ii + 1] - logo_blocks[ii]) < fps * 45 and (logo_blocks[ii + 1] - logo_blocks[ii]) < (logo_blocks[ii + 2] - logo_blocks[ii + 1]):
            playback.append((logo_blocks[ii], logo_blocks[ii + 1]))
            ii += 2
        else:
            ii += 1
    playback.append((logo_blocks[-2], logo_blocks[-1]))
    # for ii, block in enumerate(logo_blocks):
    #     if block - last > fps * 35:
    #         flag[ii] = 1
    #         flag[ii - 1] = 0
    #         last = block
    # for ii in range(1, len(flag)):
    #     if flag[ii] == -1 and flag[ii-1] == 1:
    #         flag[ii] = 0
    #     elif flag[ii] == -1 and flag[ii+1] == 0:
    #         flag[ii] = 1

def get_playback_index(video_name, videoCap):
    '''
    调用yolov5模型进行logo检测，然后匹配logo序列得到回放镜头
    :param video_name:
    :param videoCap:
    :return:
    '''
    time1 = time.time()
    frame_height = videoCap.get(cv.CAP_PROP_FRAME_HEIGHT)
    frame_width = videoCap.get(cv.CAP_PROP_FRAME_WIDTH)
    fps = int(videoCap.get(cv.CAP_PROP_FPS))

    # big_h_index = int(frame_height * 6 / 10)
    big_h_index = int(frame_height * 2 / 10)

    big_w_index = int(frame_width / 2)

    frame_count = int(videoCap.get(cv.CAP_PROP_FRAME_COUNT))

    i = 0

    lost_nums = 0
    logo_frame_index = []
    for i in tqdm(range(frame_count)):
        videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
        # i += 100;
        boolFrame, matFrame = videoCap.read()

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
                        if conf >= 0.80:
                            big_flag = True

                            logo_frame_index.append(i)

        if big_flag:
            i += 1
            lost_nums = 0
        else:
            if 10 < lost_nums < 888:
                # i = min(i + 10000, frame_count - 5000)
                i = i + fps
                lost_nums = 999
            elif lost_nums > 888:
                i += 5
            else:
                i += 5
                lost_nums += 1

    last = logo_frame_index[0]
    start = logo_frame_index[0]
    logo_blocks = []
    count = 0
    for index in logo_frame_index:
        if index - last <= 5:
            last = index
            count += 1
        else:
            if count >= 4:
                logo_blocks.append(int((start + last) / 2))

            start = index
            last = index
            count = 0

    playback = []
    ii = 0
    while ii < len(logo_blocks) - 2:
        if (logo_blocks[ii + 1] - logo_blocks[ii]) < fps * 45 and (logo_blocks[ii + 1] - logo_blocks[ii]) < (
                logo_blocks[ii + 2] - logo_blocks[ii + 1]):
            playback.append((logo_blocks[ii], logo_blocks[ii + 1]))
            ii += 2
        else:
            ii += 1
    playback.append((logo_blocks[-2], logo_blocks[-1]))

    # # 根据logo确定回放镜头区间
    # playback = []
    # last_index = logo_frame_index[0]
    # start_index = last_index
    # end_index = 0
    # logo_frames_nums = len(logo_frame_index)
    # for i_logo in range(0, logo_frames_nums):
    #     cha = logo_frame_index[i_logo] - last_index
    #     if cha < fps:
    #         last_index = logo_frame_index[i_logo]
    #     elif cha <= fps * 20:
    #         end_index = logo_frame_index[i_logo]
    #         playback.append((start_index, end_index))
    #         last_index = logo_frame_index[i_logo]
    #     else:
    #         start_index = logo_frame_index[i_logo]
    #         last_index = start_index

    # with open(video_name + '_playback' + '.json', 'w+') as fd:
    #     json.dump(playback, fd)
    # fd.close()

    time2 = time.time()
    print("finish find playbacks of video:{}  , cost {} s".format(video_name, time2 - time1))

    return playback

def get_logo_index(VIDEO_DIR):
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

                lost_nums = 0
                miss_times = 0
                logo_frame_index = []
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
                                    if conf >= 0.80:
                                        big_flag = True

                                        logo_frame_index.append(i)

                                        # 保存帧
                                        # save_img_path = os.path.join(Output_big_frames_path,
                                        #                              video_index1[1] + "_" + str(i) + ".jpg")
                                        # print("save {} *************".format(i))
                                        # cv.imwrite(save_img_path, temp_jpgframe)
                    if big_flag:
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

                with open(Json_18Russia_path + video_index1[1] + '_logo_frame_index' + '.json', 'w+') as fd:
                    json.dump(logo_frame_index, fd)
                fd.close()

                # 根据logo确定回放镜头区间
                last = logo_frame_index[0]
                start = logo_frame_index[0]
                logo_blocks = []
                count = 0
                for index in logo_frame_index:
                    if index - last <= 5:
                        last = index
                        count += 1
                    else:
                        if count >= 4:
                            logo_blocks.append(int((start + last) / 2))

                        start = index
                        last = index
                        count = 0

                playback = []
                ii = 0
                while ii < len(logo_blocks) - 2:
                    if (logo_blocks[ii + 1] - logo_blocks[ii]) < fps * 45 and (
                            logo_blocks[ii + 1] - logo_blocks[ii]) < (
                            logo_blocks[ii + 2] - logo_blocks[ii + 1]):
                        playback.append((logo_blocks[ii], logo_blocks[ii + 1]))
                        ii += 2
                    else:
                        ii += 1
                playback.append((logo_blocks[-2], logo_blocks[-1]))

                with open(Json_18Russia_path + video_index1[1] + '_playback' + '.json', 'w+') as fd:
                    json.dump(playback, fd)
                fd.close()

                print(playback)



                time2 = time.time()
                print("finish video:{}  , cost {} s".format(video_index1[1], time2 - time1))

                # return playback
                # print("save images from {} to {},  totoal {} frames".format(i_index[0], i_index[1], i_index[1] - i_index[0]))

if __name__ == '__main__':
    get_logo_index(VIDEO_DIR)
    # test_playback_algo()
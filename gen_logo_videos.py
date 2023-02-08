# 读取回放镜头的json，根据帧号生成视频
import os

import cv2 as cv
import numpy as np
import time
from collections import defaultdict
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import torch
from models.common import DetectMultiBackend

weights = 'runs/train/exp21/weights/best.pt'
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

VIDEO_DIR = "D:/dataset/logo_videos"
Output_big_frames_path = "D:/dataset/yolo_logo_5m_images"
makedir(Output_big_frames_path)

playback_videos_path = "D:/dataset/playback_logo_videos/"
makedir(playback_videos_path)


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
            videoClip = VideoFileClip(VIDEO_DIR + "\\" + video_index)

            frame_height = int(videoCap.get(cv.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(videoCap.get(cv.CAP_PROP_FRAME_WIDTH))
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

            with open(video_index1[1] + '_playback' + '.json', 'r') as fd:
                playback_index = json.load(fd)
            fd.close()

            fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_width, video_height = frame_width, frame_height

            for k, interval in enumerate(playback_index):
                save_path = playback_videos_path + video_index1[1] + '_' + str(k) + '.mp4'
                videoWriter = cv.VideoWriter(save_path, fourcc, fps, (video_width, video_height))
                for f_index in range(interval[0], interval[1]+1):
                    videoCap.set(cv.CAP_PROP_POS_FRAMES, f_index)
                    # i += 100;
                    boolFrame, matFrame = videoCap.read()
                    if boolFrame:
                        temp_jpgframe = np.asarray(matFrame)
                        videoWriter.write(temp_jpgframe)
                videoWriter.release()

            # imgs = glob.glob(frames_path + "/*.jpg")
            # # print(imgs)
            # # frames_num = len(imgs)
            # for i in range(2851, 2894):
            #     if os.path.isfile("%s/%d.jpg" % (frames_path, i)):
            #         # print(i)
            #         frame = cv2.imread("%s/%d.jpg" % (frames_path, i))
            #         # print(frame.shape)
            #         videoWriter.write(frame)
            # videoWriter.release()

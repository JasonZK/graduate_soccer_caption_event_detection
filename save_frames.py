# 截取视频中的特定帧并保存
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2 as cv
import numpy as np
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


# VIDEO_DIR = "D:/dataset/MAKE_OCR_DATA/Small_PadOCR_videos"
# Output_big_frames_path = "D:/dataset/MAKE_OCR_DATA/small_frames"

VIDEO_DIR = "D:/dataset/A_graduate_experiment/yolo_dataset/card_dataset/card_videos"
Output_big_frames_path = "D:/dataset/A_graduate_experiment/yolo_dataset/card_dataset/raw_card_frames"

# VIDEO_DIR = "D:/dataset/A_graduate_experiment/yolo_dataset/sub_dataset/sub_videos"
# Output_big_frames_path = "D:/dataset/A_graduate_experiment/yolo_dataset/sub_dataset/yolo_sub_dataset3/raw_sub_frames"

makedir(Output_big_frames_path)

for root, _, files in os.walk(VIDEO_DIR):
    for video_index in files:
        time1 = time.time()
        video_index1 = video_index.split('_')
        if video_index1[1] != "4017":
            continue


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
        i_index = [62255, 62282]
        for i in range(i_index[0], i_index[1], 2):
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

            # 不截取就保存
            if boolFrame:
                temp_jpgframe = np.asarray(matFrame)
                save_img_path = os.path.join(Output_big_frames_path,
                                             video_index1[1] + "_" + str(i) + ".jpg")

                cv.imwrite(save_img_path, temp_jpgframe)

        print("save images from {} to {},  totoal {} frames".format(i_index[0], i_index[1], i_index[1] - i_index[0]))
    break
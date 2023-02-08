from moviepy import *
from moviepy.editor import *
import re
import tqdm
import json
import cv2
# VIDEO_DIR = "D:/dataset/test_a_video"

from SoccerNet.Downloader import getListGames
listGames = getListGames("test")
path = "D:/dataset/SoccerNet/SoccerNet_test_hq"
label_name = "1_720p.mkv"

for game in listGames:
    labels = json.load(open(os.path.join(path, game, label_name)))
    print(game)


# for video_dir in VIDEO_DIR:
#     video = VideoFileClip(video_dir)# 读入视频
#     audio = video.audio
#
#     video = video.set_audio(audio)# 将音轨合成到视频中
#     video.write_videofile("result/" + video_dir.split("\\")[-1])# 输出


# video = VideoFileClip('D:/dataset/temp_videos/Godfather.mp4')
# audio = video.audio
# video = video.set_audio(audio)
# video_clip = video.subclip(1, 51)
# video_clip.write_videofile("C:/Users/pc/Desktop/Godfather720.mp4")

ss = "asgfyga123T34"
sss = re.sub(r'(\d)T', r"\1'1'", ss)
print(sss)


# import torch
#
# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
#
# # Images
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
#
# # Inference
# results = model(img)
#
# # Results
# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
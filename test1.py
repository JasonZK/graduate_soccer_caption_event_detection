from moviepy import *
from moviepy.editor import *
import re
import tqdm
import json
import cv2
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
# VIDEO_DIR = "D:/dataset/test_a_video"

# from SoccerNet.Downloader import getListGames
# listGames = getListGames("test")
# path = "D:/dataset/SoccerNet/SoccerNet_test_hq"
# label_name = "1_720p.mkv"
#
# for game in listGames:
#     labels = json.load(open(os.path.join(path, game, label_name)))
#     print(game)


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

team_name = []
f = open("team_name_full.txt", "r")
for line in f:
    line = line[:-1]
    team_name.append(line)

game = "2015-02-24 - 22-45 Manchester City 1 - 2 Barcelona"
two_team_names = game[19:]
divid_index = two_team_names.find(" - ")
team1_name = two_team_names[:divid_index-2]
team2_name = two_team_names[divid_index+5:]

print(team1_name)
print(team2_name)
team_name.append(team1_name)
team_name.append(team2_name)
block_team1_names = team1_name.split(' ')
block_team2_names = team2_name.split(' ')
team_name.extend(block_team1_names)
team_name.extend(block_team2_names)

strr1 = "MAN"
strr2 = "Barcelona"
team1 = process.extractOne(strr1, team_name, scorer=fuzz.ratio,
                                       score_cutoff=70)
team2 = process.extractOne(strr2, team_name, scorer=fuzz.ratio,
                                       score_cutoff=70)
print(team1)
print(team2)


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
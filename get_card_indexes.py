# 读取数据集的标注信息，找到有红黄牌的视频，并输出其时间
import os
import json
import cv2 as cv

video_dir = "D:/dataset/SoccerNet/SoccerNet_test_hq/"

for root, matches, files in os.walk(video_dir):
    for match in matches:
        # if match != "italy_serie-a":
        #     continue
        match_path = os.path.join(video_dir, match)
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

                                video_path = os.path.join(game_path, video_index)
                                videoCap = cv.VideoCapture(video_path)
                                fps = videoCap.get(cv.CAP_PROP_FPS)
                                video_name = match + ' ' + game_split[0] + ' ' + game_split[-1] + ' ' + \
                                             video_index1[0]

                                labels = json.load(open(os.path.join(game_path, "Labels-v2.json")))
                                gt_goal_frame_indexes = []
                                for annotation in labels["annotations"]:
                                    if (annotation["label"] == "Yellow card" or annotation["label"] == "Red card") \
                                            and annotation["gameTime"][0] == video_index[0]:
                                        label_time = annotation["gameTime"]
                                        goal_minutes = int(label_time[-5:-3])
                                        goal_seconds = int(label_time[-2::])
                                        gt_frame = fps * (goal_seconds + 60 * goal_minutes)
                                        gt_goal_frame_indexes.append([label_time[-5::], gt_frame, annotation["label"]])
                                        # print("video time:{}   frame index:{}".format(str(goal_minutes) + ':' + str(goal_seconds), gt_frame))

                                if gt_goal_frame_indexes:
                                    print("")
                                    print("------------------------------------------------------")
                                    print("video:{}".format(video_name))
                                    for item in gt_goal_frame_indexes:
                                        if item[2] == "Red card":
                                            print("*****")
                                        print("{}    video time:{}   frame index:{}".format(item[2], item[0], item[1]))
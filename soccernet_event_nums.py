import os
import json

VIDEO_DIR = "D:/dataset/SoccerNet/SoccerNet_test_hq/"

for root, matches, files in os.walk(VIDEO_DIR):
    total = 0
    for match in matches:
        # if match != "germany_bundesliga":
        #     continue
        match_path = os.path.join(VIDEO_DIR, match)
        match_gt = 0
        for match_root, years, ffiles in os.walk(match_path):
            for year in years:
                year_gt = 0
                year_path = os.path.join(match_path, year)
                for year_root, games, fffiles in os.walk(year_path):
                    for game in games:

                        game_path = os.path.join(year_path, game)
                        game_split = game.split(' ')
                        # 根据文件名，提取出队伍名字
                        two_team_names = game[19:]
                        divid_index = two_team_names.find(" - ")
                        team1_name = two_team_names[:divid_index - 2]
                        team2_name = two_team_names[divid_index + 5:]

                        for game_root, _, videos in os.walk(game_path):
                            for video_index in videos:
                                gt_goal_times = []
                                video_index1 = video_index.split('_')
                                if not video_index.endswith('mkv'):
                                    continue
                                video_name = match + ' ' + game_split[0] + ' ' + game_split[-1] + ' ' + \
                                             video_index1[0]

                                # if video_name != "italy_serie-a 2015-08-29 Empoli 1":
                                #     continue

                                video_path = os.path.join(game_path, video_index)

                                labels = json.load(open(os.path.join(game_path, "Labels-v2.json")))


                                for annotation in labels["annotations"]:
                                    if "card" in annotation["label"] and annotation["gameTime"][0] == video_index[0]:
                                        label_time = annotation["gameTime"]
                                        gt_goal_times.append(label_time)
                                year_gt += len(gt_goal_times)
                                print(video_name + " : {}".format(gt_goal_times))

                print("{} : {} : {}".format(match, year, year_gt))
                match_gt += year_gt

        print("{} : {}".format(match, match_gt))
        total += match_gt

    print("total : {}".format(total))
    break

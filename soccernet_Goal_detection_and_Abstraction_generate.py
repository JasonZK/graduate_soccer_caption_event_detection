# 检测进球得分事件，生成相应的文字摘要和视频摘要
# 需要提前有该视频的回放镜头json文件
import os
# -*- coding:utf-8 -*-

from my_utils.tools_in_Goal_detection import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

keyword1 = "goal"

if_gen_video = False

EVENT_DIR = "D:/dataset/event"

# VIDEO_DIR = "D:/dataset/video_6"
# Output_big_frames_path = "D:/dataset/A_graduate_experiment/big_frames_19Asia/"

VIDEO_DIR = "D:/dataset/SoccerNet/SoccerNet_test_hq/"
Output_big_frames_path = "D:/dataset/A_graduate_experiment/big_frames_SoccerNet_2-11/"

makedir(Output_big_frames_path)

video_output_path = "D:/dataset/output_goal_video/"
makedir(video_output_path)

Json_18Russia_path = "D:/study/proj/yolo/yolov5-master/18Russia_json/"

ocr = PaddleOCR(lang="en", gpu_mem=5000, det_model_dir="./padOCR_inference/en_PP-OCRv3_det_infer",
                rec_model_dir="./padOCR_inference/en_PP-OCRv3_rec_infer/", show_log=False)  # 首次执行会自动下载模型文件

import logging

# 默认的warning级别，只输出warning以上的
# 使用basicConfig()来指定日志级别和相关信息

logging.basicConfig(level=logging.DEBUG  # 设置日志输出格式
                    , filename="log/Goal_detection2-11.log"  # log日志输出的文件位置和文件名
                    , filemode="w+"  # 文件的写入格式，w为重新写入文件，默认是追加
                    , format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                    # 日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    , datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
                    )


def get_frames(video_dir):
    get = 0
    miss = 0
    gt_total_goal = 0
    get_total_goal = 0
    get_right_total_goal = 0

    more = 0
    # with open('video_6_score_change.json', 'r') as fd:
    #     Video_Score_dic = json.load(fd)
    SCORES = {}
    makedir(Output_big_frames_path)

    video_index = ''

    for root, matches, files in os.walk(video_dir):
        for match in matches:
            if match != "italy_serie-a":
                continue
            match_path = os.path.join(video_dir, match)
            for match_root, years, ffiles in os.walk(match_path):
                for year in years:
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
                            # 将从文件名中得到的队名，以及队名的拆分 加入到队名库中
                            team_name.append(team1_name)
                            team_name.append(team2_name)
                            block_team1_names = team1_name.split(' ')
                            block_team2_names = team2_name.split(' ')
                            team_name.extend(block_team1_names)
                            team_name.extend(block_team2_names)
                            for game_root, _, videos in os.walk(game_path):
                                for video_index in videos:
                                    video_index1 = video_index.split('_')
                                    if not video_index.endswith('mkv'):
                                        continue
                                    time1 = time.time()
                                    video_name = match + ' ' + game_split[0] + ' ' + game_split[-1] + ' ' + \
                                                 video_index1[0]

                                    # if video_name != "germany_bundesliga 2015-08-29 Leverkusen 2":
                                    #     continue

                                    video_path = os.path.join(game_path, video_index)
                                    videoCap = cv.VideoCapture(video_path)
                                    videoClip = VideoFileClip(video_path)

                                    frame_height = videoCap.get(cv.CAP_PROP_FRAME_HEIGHT)
                                    frame_width = videoCap.get(cv.CAP_PROP_FRAME_WIDTH)
                                    small_h_index = int(frame_height * 3 / 10)
                                    fps = videoCap.get(cv.CAP_PROP_FPS)

                                    big_h_index = int(frame_height * 6 / 10)
                                    big_w_index = int(frame_width / 2)
                                    x1_big_center = big_w_index - 50
                                    x2_big_center = big_w_index + 50
                                    frame_count = videoCap.get(cv.CAP_PROP_FRAME_COUNT)

                                    labels = json.load(open(os.path.join(game_path, "Labels-v2.json")))
                                    gt_goal_frame_indexes = []
                                    for annotation in labels["annotations"]:
                                        if annotation["label"] == "Goal" and annotation["gameTime"][0] == video_index[
                                            0]:
                                            label_time = annotation["gameTime"]
                                            goal_minutes = int(label_time[-5:-3])
                                            goal_seconds = int(label_time[-2::])
                                            gt_frame = fps * (goal_seconds + 60 * goal_minutes)
                                            gt_goal_frame_indexes.append(gt_frame)

                                    gt_total_goal += len(gt_goal_frame_indexes)

                                    logging.info("  ")
                                    logging.info(
                                        "--------------------------------------------------------------------------------------------------------------")
                                    logging.info("video:{}".format(video_name))
                                    logging.info(
                                        "视频属性：   总帧数：{}   FPS:{}   width:{}   height:{}".format(frame_count, fps,
                                                                                                frame_width,
                                                                                                frame_height))

                                    player_name_dic = defaultdict(int)
                                    score_time_dic = defaultdict(int)

                                    left_team = ''
                                    right_team = ''
                                    left_team_x = 0
                                    right_team_x = 0
                                    team_y1 = 10000
                                    team_y2 = 0
                                    Player_Time_list = []

                                    # 获取视频类型与比赛时间参照
                                    # 视频类型：上半场、下半场、整场(first, second, full)
                                    # time_refer: [(12335, "16:64"), (54674, "40:37")]
                                    video_type, time_refer = game_time(videoCap, frame_count, small_h_index, fps)

                                    # gen_goal_video(video_name, videoCap, video_type, time_refer, '45', 'hhh', [],
                                    #                video_output_path)

                                    # 获取回放镜头，有两种方式：
                                    playback_index = []
                                    # 1、调用另一个文件里面的get_playback_index函数，获取回放镜头
                                    # playback_index = get_playback_index(video_name, videoCap)

                                    # 2、从json文件中，读取logo_frame_index，然后匹配得到回放镜头
                                    # playback_index = get_playback_index_from_json(Json_18Russia_path, video_name, fps)

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

                                    i = int(frame_count * (6 / 10))
                                    # i = int(frame_count * (99 / 100))
                                    # i = 0
                                    # i = 38760

                                    # 遍历视频
                                    while i < frame_count:
                                        # if i > 73499:
                                        #     logging.info("a")
                                        # logging.info("deal with {}".format(i))
                                        print("deal with {} on {}".format(video_name, i))

                                        # 评估候选比分
                                        # 获取ocr第i帧的直接结果temp_result， 以及字符串连接后的result
                                        big_result, big_temp_result = get_ocr_result(videoCap, i, big_h_index)

                                        big_flag = False
                                        team_nums = 0

                                        left_line = []
                                        right_line = []

                                        if big_result:
                                            # logging.info(big_result, i)
                                            # 首先检测是否存在两个队名，同时将合适位置的字符串按左右队伍分类
                                            two_teams, left_big_temp_result, right_big_temp_result, left_team, left_team_x, \
                                            right_team, right_team_x = check_teams(big_temp_result, big_w_index,
                                                                                   frame_width, left_team, left_team_x,
                                                                                   right_team, right_team_x)

                                            # 检测出了两个球队名，说明是大字幕，此时team_y1，team_y2都有了
                                            if two_teams:

                                                # 下面这一段是将确定为大字幕的帧保存下来，debug的时候方便看
                                                save_a_frame(i, videoCap, big_h_index, video_name,
                                                             Output_big_frames_path)

                                                # # doc = nlp(big_result)
                                                # # logging.info(big_result)
                                                # # logging.info([(ent.text, ent.label_) for ent in doc.ents])

                                                # 接下来分别对 左候选集 和 右候选集 进行分析处理
                                                if left_big_temp_result:
                                                    left_line = divid_rows("left", left_big_temp_result, team_y1,
                                                                           team_y2, left_team_x)


                                                    player_name_dic, score_time_dic, Player_Time_list, find_flag = check_line2(
                                                        left_line, 1, player_name_dic, score_time_dic, i,
                                                        Player_Time_list,
                                                        find_flag)

                                                if right_big_temp_result:
                                                    right_line = divid_rows("right", right_big_temp_result, team_y1,
                                                                            team_y2, right_team_x)

                                                    # 排序、分行完成后，进行逐行细致分析
                                                    player_name_dic, score_time_dic, Player_Time_list, find_flag = check_line2(
                                                        right_line, -1, player_name_dic, score_time_dic, i,
                                                        Player_Time_list, find_flag)




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

                                    # 根据player_name_dic和score_time_dic，整合Player_Time_list
                                    # 确定最终的球员名和进球时间
                                    merge_player_time_list(Player_Time_list, player_name_dic, score_time_dic)

                                    Summary_Json = {}
                                    Summary_Json['teamA'] = left_team
                                    Summary_Json['teamB'] = right_team
                                    time_line = {}

                                    logging.info(
                                        "##################### video:{}  goal events ####################".format(
                                            video_name))
                                    logging.info("### showed by players ###")

                                    # 根据球员展示进球情况，生成球员文本摘要
                                    time_line_sorted = show_player_text_summarization(Player_Time_list, time_line, left_team,
                                                                               right_team)

                                    get_total_goal += len(time_line_sorted)

                                    # 生成时间线的文本摘要，以及视频中的文字水印
                                    time_line_texts, team_order, video_texts = show_time_line_and_gen_video(time_line_sorted, left_team, right_team)

                                    get_right_goal_frame_indexes = []
                                    for key in time_line_sorted:
                                        goal_time_minute = key[0]
                                        name = key[1].split('_')[0]
                                        team = key[1].split('_')[1]
                                        # 将进球时间（分钟）转换成视频帧数
                                        goal_frame_index = time_to_frame_index(goal_time_minute, video_type, time_refer,
                                                                               fps)
                                        if goal_frame_index < 1 or goal_frame_index >= frame_count:
                                            logging.info("goal time error! goal time:{}".format(goal_time_minute))
                                        else:
                                            # 统计找到的正确的数量
                                            for gt_goal_frame_index in gt_goal_frame_indexes:
                                                if abs(goal_frame_index - gt_goal_frame_index) < (fps * 60 * 2):
                                                    get_right_goal_frame_indexes.append(
                                                        str(gt_goal_frame_index) + " - " + str(goal_frame_index))

                                            get_right_total_goal += len(get_right_goal_frame_indexes)

                                        # 如果要生成视频
                                        if if_gen_video:
                                            # 针对这个时间的进球，结合回放镜头合集，找出对应的回放镜头作为视频摘要
                                            gen_one_goal_video(video_name, videoClip, goal_time_minute, name, team,
                                                               playback_index, video_output_path,
                                                               video_texts[goal_time_minute], goal_frame_index, fps)

                                    left_score = len(team_order[left_team])
                                    right_score = len(team_order[right_team])
                                    game_result = 1
                                    summary = []
                                    if left_score > right_score:
                                        win_team = left_team
                                        win_score = left_score
                                        lose_team = right_team
                                        lose_score = right_score
                                    elif right_score > left_score:
                                        win_team = right_team
                                        win_score = right_score
                                        lose_team = left_team
                                        lose_score = left_score
                                    else:
                                        game_result = -1

                                    if video_type == "first":
                                        m1 = "本场比赛的双方是" + left_team + "和" + right_team + "。"
                                        summary.append(m1)
                                        if game_result == -1:
                                            if left_score >= 2:
                                                m2 = "上半场结束，目前结果是" + left_team + str(left_score) + "比" + str(
                                                    right_score) + "战平" + right_team + "，双方贡献了一场精彩的比赛！"
                                            else:
                                                m2 = "上半场结束，目前结果是" + left_team + str(left_score) + "比" + str(
                                                    right_score) + "战平" + right_team + "。"

                                        else:
                                            if win_score - lose_score >= 2:
                                                m2 = "上半场结束，目前结果是" + win_team + str(win_score) + "比" + str(
                                                    lose_score) + "大幅领先" + lose_team + "。"
                                            else:
                                                m2 = "上半场结束，目前结果是" + win_team + str(win_score) + "比" + str(
                                                    lose_score) + "领先" + lose_team + "。"

                                        summary.append(m2)
                                    else:
                                        m1 = "本场比赛的双方是" + left_team + "和" + right_team + "。"
                                        summary.append(m1)
                                        if game_result == -1:
                                            if left_score >= 2:
                                                m2 = "最终结果是" + left_team + str(left_score) + "比" + str(
                                                    right_score) + "战平" + right_team + "，双方贡献了一场精彩的比赛！"
                                            else:
                                                m2 = "最终结果是" + left_team + str(left_score) + "比" + str(
                                                    right_score) + "战平" + right_team + "。"

                                        else:
                                            if win_score - lose_score >= 2:
                                                m2 = "最终结果是" + win_team + str(win_score) + "比" + str(
                                                    lose_score) + "大胜" + lose_team + "。"
                                            else:
                                                m2 = "最终结果是" + win_team + str(win_score) + "比" + str(
                                                    lose_score) + "小胜" + lose_team + "，双方贡献了一场精彩的比赛！"

                                        summary.append(m2)

                                    m4 = "以下是得分详情："
                                    summary.append(m4)
                                    for sens in summary:
                                        logging.info(sens)

                                    logging.info("### showed by time line ###")
                                    logging.info(
                                        "               {}                     time                     {}".format(
                                            left_team, right_team))
                                    for text in list(time_line_texts.keys()):
                                        logging.info(text)

                                    logging.info(" ")
                                    # for gt_goal_frame_index in gt_goal_frame_indexes:
                                    logging.info("真实进球时间（帧号）：{}".format(gt_goal_frame_indexes))
                                    logging.info("检测正确结果：{}".format(get_right_goal_frame_indexes))
                                    logging.info(
                                        "GT : {}， 共找到：{}， 其中正确的：{}".format(len(gt_goal_frame_indexes), len(time_line),
                                                                           len(get_right_goal_frame_indexes)))
                                    logging.info("####################################################")

                                    time2 = time.time()
                                    logging.info("time for this video:{}".format(time2 - time1))

                # SCORES[video_index] = score_record

                # with open('scores_3' + '.json', 'w') as fd:
                #     json.dump(SCORES, fd)

    logging.info("所有真实得分数:{}   检测到的得分数:{}   检测正确的得分数:{}  ".format(gt_total_goal, get_total_goal, get_right_total_goal))


get_frames(video_dir=VIDEO_DIR)

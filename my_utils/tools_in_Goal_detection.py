# 得分事件检测所需要的工具类函数
import os

# -*- coding:utf-8 -*-
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
from paddleocr import PaddleOCR, draw_ocr
import cv2 as cv
import numpy as np
import time
from collections import defaultdict
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import torch
import spacy
# from get_logo_frames import get_playback_index
from moviepy import *
from moviepy.editor import *
from my_utils import *

ocr = PaddleOCR(lang="en", gpu_mem=5000, det_model_dir="./padOCR_inference/en_PP-OCRv3_det_infer",
                rec_model_dir="./padOCR_inference/en_PP-OCRv3_rec_infer/", show_log=False)  # 首次执行会自动下载模型文件


class PlayerTime:
    def __init__(self, playername=''):
        self.goaltime = defaultdict(str)
        self.playername = playername
        self.location = 0
        self.use = 1


class OneLine:
    def __init__(self, y1, y2):
        self.y1 = y1
        self.y2 = y2
        self.strr = ""
        self.times = []
        self.location = ""


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


import logging
# 默认的warning级别，只输出warning以上的
# 使用basicConfig()来指定日志级别和相关信息

# logging.basicConfig(level=logging.DEBUG  # 设置日志输出格式
#                     , filename="log/debug_Goal_detection2-10.log"  # log日志输出的文件位置和文件名
#                     , filemode="w+"  # 文件的写入格式，w为重新写入文件，默认是追加
#                     , format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
#                     # 日志输出的格式
#                     # -8表示占位符，让输出左对齐，输出长度都为8位
#                     , datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
#                     )

# 获取队名文件
team_name = []
f = open("team_name_full.txt", "r")
for line in f:
    line = line[:-1]
    team_name.append(line)


def print_json(data):
    print(json.dumps(data, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))


def get_ocr_result(videoCap, i, h_index):
    '''
    调用ocr接口，对第i帧进行字符检测识别
    :param videoCap: opencv对视频生成的cap
    :param i:
    :param h_index: 高度阈值，用来截取区域，以减少计算成本
    :return: 返回检测字符的拼接结果result，和ocr原始结果temp_result（带有框坐标和置信率）
    '''
    result = ''
    ocr_list = []
    temp_result = []

    videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
    # i += 100;
    boolFrame, matFrame = videoCap.read()

    if boolFrame:
        temp_jpgframe = np.asarray(matFrame)
        # 截取上面0-90区域进行OCR检测
        if h_index < 250:
            jpgframe = temp_jpgframe[0:h_index]
        else:
            jpgframe = temp_jpgframe[h_index:]

        # 得到OCR识别结果
        temp_result = ocr(jpgframe)

        for mystr, ration in temp_result[1]:
            result += mystr
            ocr_list.append(mystr)
            result += ' '
    return result, temp_result


def game_time(videoCap, frame_count, small_h_index, fps):
    '''

    :param videoCap:
    :param frame_count:
    :param small_h_index:
    :param fps:
    :return:
    '''
    frame_index1 = int(frame_count * 1 / 5)
    frame_index2 = int(frame_count * 4 / 5)
    i1 = frame_index1
    time1_candidate = []
    time_12 = []
    for i1 in [frame_index1, frame_index2]:
        ttemp = i1
        while i1 < ttemp + fps * 120 or len(time_12) < 2:
            result, temp_result = get_ocr_result(videoCap, i1, small_h_index)
            if result:
                game_time = re.findall("\d\d:\d\d", result)
                if game_time:
                    # logging.info("frames: {}    time: {}".format(i1, game_time))
                    time1_candidate.append((i1, game_time[0]))
                    if len(time1_candidate) >= 3:
                        ii0, game_time0 = time1_candidate[0]
                        ii1, game_time1 = time1_candidate[1]
                        ii2, game_time2 = time1_candidate[2]

                        t0 = int(game_time0.split(':')[0]) * 60 + int(game_time0.split(':')[1])
                        t1 = int(game_time1.split(':')[0]) * 60 + int(game_time1.split(':')[1])
                        t2 = int(game_time2.split(':')[0]) * 60 + int(game_time2.split(':')[1])
                        if (ii1 - ii0) <= (t1 - t0 + 1) * fps and (ii2 - ii1) <= (t2 - t1 + 1) * fps:
                            time_12.append(time1_candidate[0])
                            time1_candidate.clear()
                            break
                        else:
                            time1_candidate.clear()
            i1 += (fps + 1)

    time1_fen = int(time_12[0][1].split(':')[0])
    time2_fen = int(time_12[1][1].split(':')[0])
    # logging.info("finished finding the game time!*****************************************************************")
    if time2_fen < 45:
        # shang
        logging.info("本视频为上半场比赛")
        return 'first', time_12
    elif time1_fen > 45:
        # xia
        logging.info("本视频为下半场比赛")
        return 'second', time_12
    else:
        logging.info("本视频为全场比赛")
        return 'full', time_12


def gen_one_goal_video(video_name, videoClip, goal_time_minute, player, playback_index, video_output_path, text, goal_frame_index, fps):

        save_path = video_output_path + video_name + '_' + player + goal_time_minute + '_Playback_audio' + '.mp4'
        bias = 100000
        start = 0
        end = 0
        nearest = 0
        before_len = 0
        after_len = 0
        nearest_len = 0
        for ii, (ps, pe) in enumerate(playback_index):
            if abs(goal_frame_index - (ps + pe) / 2) < bias:
                bias = abs(goal_frame_index - (ps + pe) / 2)
                nearest = ii
                nearest_len = pe - ps
        if nearest - 1 > 0:
            before_len = playback_index[nearest - 1][1] - playback_index[nearest - 1][0]
        if nearest + 1 < len(playback_index):
            after_len = playback_index[nearest + 1][1] - playback_index[nearest + 1][0]
        if nearest_len >= before_len and nearest_len >= after_len:
            start = playback_index[nearest][0]
            end = playback_index[nearest][1]
        elif before_len >= nearest_len and before_len >= after_len:
            start = playback_index[nearest - 1][0]
            end = playback_index[nearest - 1][1]
        else:
            start = playback_index[nearest + 1][0]
            end = playback_index[nearest + 1][1]

        video_clip = videoClip.subclip(start // fps - 2, end // fps + 2)
        # text = player + " from " + team + " scores a goal at " + goal_time + "!"
        font2_B = "Fonts/arialbd.ttf"
        texpClip = TextClip(text, font=font2_B, fontsize=35, color='red').set_position('top').set_duration(
            video_clip.duration).set_start(0)
        video_clip = CompositeVideoClip([video_clip, texpClip])
        video_clip.write_videofile(save_path, logger=None)

        logging.info("save video to {}".format(save_path))


def check_line2(result_line, LorR, player_name_dic, score_time_dic, i, Player_Time_list, big_flag):
    for lline in result_line:
        str_line = lline.strr
        str_line1 = lline.strr
        len_line = len(str_line1)

        # 如果没有字符或者没有数字，就跳过这一行
        string_list = re.findall("[A-Za-z]+?[A-Za-z]+", lline.strr)
        number_list = re.findall("\d+", lline.strr)
        if (not string_list) or (not number_list):
            continue

        player_name = ''
        score_time = ''
        score_frame_type = ''
        pre_mode = 'none'
        pre_c = ''
        temp_goaltime = defaultdict(str)
        start_str = 0
        end_str = 0
        OG_flag = False

        # 针对19_Asia里面，把1识别成T的案例，将数字后面紧跟T的，全部把T换成1
        for jj, c in enumerate(str_line1):
            if str_line1[jj] == 'T' and jj - 1 > 0 and str_line1[jj - 1].isdigit():
                temp_list = list(str_line1)
                temp_list[jj] = '1'
                str_line = ''.join(temp_list)

            if str_line1[jj] == 'O' and jj - 1 >= 0 and str_line1[jj - 1].isdigit() and jj+1 < len_line and not str_line1[jj + 1].isdigit():
                temp_list = list(str_line1)
                temp_list[jj] = '0'
                str_line = ''.join(temp_list)

        # 球员名在得分时间之前：Messi 34’
        if str_line[0].isalpha():
            ii = 0
            while ii < len_line:

                if str_line[ii].isalpha():
                    # 绝对是P
                    if str_line[ii].upper() == 'P' \
                            and ((pre_mode == "score"
                                  and ((ii + 1 == len_line) or (ii + 1 < len_line and not str_line[ii + 1].isalpha())))
                                 or ((ii - 1) >= 0 and str_line[ii - 1] == '(')
                                 or ((ii + 1 < len_line) and str_line[ii + 1] == ')')):
                        score_frame_type = str(int(i)) + "_" + "P"
                        # 不是P，是新的P开头的人名
                        ii += 1
                    # 绝对是OG
                    elif pre_mode == "score" and str_line[ii].upper() == 'O' and \
                            ((ii + 2 == len_line and str_line[ii + 1].upper() == "G")
                             or (ii + 2 < len_line and str_line[ii + 1].upper() == "G" and not str_line[
                                        ii + 2].isalpha())):
                        score_frame_type = str(int(i)) + "_" + "OG"
                        OG_flag = True
                        ii += 1
                    else:
                        start_str = ii
                        while ii < len_line and (str_line[ii].isalpha() or str_line[ii] == ' ' or str_line[ii] == '.'):
                            ii += 1
                        end_str = ii
                        # 之前有一个完整的球员 时间记录了，说明是一行两个球员记录
                        if pre_mode == "score":
                            # 先把之前的记录下来
                            if len(player_name) > 2 and score_time and eval(score_time) <= 120:
                                # 暂时没定这个得分的类型，说明是N
                                if not score_frame_type:
                                    score_frame_type = str(int(i)) + "_" + "N"
                                temp_goaltime[score_time] = score_frame_type
                                score_frame_type = ''
                                one_playertime = PlayerTime(player_name)
                                one_playertime.goaltime = temp_goaltime
                                temp_goaltime = defaultdict(str)
                                if one_playertime.location == 0:
                                    if OG_flag:
                                        one_playertime.location = -LorR
                                        OG_flag = False
                                    else:
                                        one_playertime.location = LorR

                                player_name_dic[player_name] += 1
                                score_time_dic[score_time] += 1

                                score_time = ""

                                Player_Time_list.append(one_playertime)
                                del one_playertime
                                big_flag = True

                        player_name = str_line[start_str:end_str]
                        pre_mode = "player"

                elif str_line[ii].isdigit():
                    start_str = ii
                    # 得分时间的开头不能是0
                    if str_line[start_str] == '0':
                        break

                    while ii < len_line and str_line[ii].isdigit() and ii - start_str < 2:
                        ii += 1
                    end_str = ii
                    # 第一个得分时间
                    if pre_mode == "player":
                        score_time = str_line[start_str:end_str]
                    # 还有别的得分时间
                    elif pre_mode == "score":

                        # 这里为了解决连续时间的分割错误问题，如9，11，55  被OCR识别成91，1，55
                        # 采取的规则是，连续时间应该是递增的，不符合就跳过
                        # 记得这里改了后面也要改
                        now_time = str_line[start_str:end_str]
                        if eval(score_time) > eval(now_time):
                            score_time = ''
                            # 这里是break好还是continue好呢
                            break

                        # 暂时没定这个得分的类型，说明是N
                        if not score_frame_type:
                            score_frame_type = str(int(i)) + "_" + "N"
                        # 先把之前的得分记录下来
                        temp_goaltime[score_time] = score_frame_type
                        score_frame_type = ''
                        score_time_dic[score_time] += 1
                        # 这是新的得分
                        score_time = str_line[start_str:end_str]
                    pre_mode = "score"

                elif str_line[ii] == '+':
                    ii += 1
                    start_str = ii
                    while ii < len_line and str_line[ii].isdigit():
                        ii += 1
                    end_str = ii
                    plus_score_time = str_line[start_str:end_str]
                    if plus_score_time and score_time and pre_mode == "score":
                        score_time = score_time + '+' + plus_score_time
                        score_frame_type = str(int(i)) + "_" + "add"
                else:
                    ii += 1

            if len(player_name) > 2 and score_time and 0 < eval(score_time) <= 120:
                if not score_frame_type:
                    score_frame_type = str(int(i)) + "_" + "N"
                temp_goaltime[score_time] = score_frame_type
                score_frame_type = ''
                one_playertime = PlayerTime(player_name)
                one_playertime.goaltime = temp_goaltime
                temp_goaltime = defaultdict(str)
                if one_playertime.location == 0:
                    one_playertime.location = LorR

                player_name_dic[player_name] += 1
                score_time_dic[score_time] += 1

                Player_Time_list.append(one_playertime)
                del one_playertime
                big_flag = True

        # 得分时间在球员名之前：34’ Messi
        elif str_line[0].isdigit():
            ii = 0
            while ii < len_line:
                if str_line[ii].isalpha():
                    # 绝对是P
                    if str_line[ii].upper() == 'P' \
                            and ((pre_mode == "score" and((ii + 1 == len_line) or (ii + 1 < len_line and not str_line[
                                                             ii + 1].isalpha())))
                                 or ((ii - 1) >= 0 and str_line[ii - 1] == '(')
                                 or ((ii + 1 < len_line) and str_line[ii + 1] == ')') ):
                        score_frame_type = str(int(i)) + "_" + "P"
                        # 不是P，是新的P开头的人名
                        ii += 1
                    # 绝对是OG
                    elif pre_mode == "score" and str_line[ii].upper() == 'O' and \
                            ((ii + 2 == len_line and str_line[ii + 1].upper() == "G")
                             or (ii + 2 < len_line and str_line[ii + 1].upper() == "G" and not str_line[
                                        ii + 2].isalpha())):
                        score_frame_type = str(int(i)) + "_" + "OG"
                        OG_flag = True
                        ii += 1
                    else:
                        start_str = ii
                        while ii < len_line and (str_line[ii].isalpha() or str_line[ii] == ' ' or str_line[ii] == '.'):
                            ii += 1
                        end_str = ii

                        player_name = str_line[start_str:end_str]
                        pre_mode = "player"

                elif str_line[ii].isdigit():
                    start_str = ii
                    # 得分时间的开头不能是0
                    if str_line[start_str] == '0':
                        break

                    while ii < len_line and str_line[ii].isdigit() and ii - start_str < 2:
                        ii += 1
                    end_str = ii
                    # 之前有一个完整的球员 时间记录了，说明是一行两个球员记录
                    if pre_mode == "player":
                        # 先把之前的记录下来
                        if len(player_name) > 2 and score_time and eval(score_time) <= 120:
                            # 暂时没定这个得分的类型，说明是N
                            if not score_frame_type:
                                score_frame_type = str(int(i)) + "_" + "N"
                            temp_goaltime[score_time] = score_frame_type
                            score_frame_type = ''
                            one_playertime = PlayerTime(player_name)
                            one_playertime.goaltime = temp_goaltime
                            temp_goaltime = defaultdict(str)
                            if one_playertime.location == 0:
                                if OG_flag:
                                    one_playertime.location = -LorR
                                    OG_flag = False
                                else:
                                    one_playertime.location = LorR

                            player_name_dic[player_name] += 1
                            score_time_dic[score_time] += 1

                            score_time = ""

                            Player_Time_list.append(one_playertime)
                            del one_playertime
                            big_flag = True

                    # 还有别的得分时间
                    elif pre_mode == "score":

                        # 这里为了解决连续时间的分割错误问题，如9，11，55  被OCR识别成91，1，55
                        # 采取的规则是，连续时间应该是递增的，不符合就跳过
                        # 记得这里改了后面也要改
                        now_time = str_line[start_str:end_str]
                        if eval(score_time) > eval(now_time):
                            score_time = ''
                            # 这里是break好还是continue好呢
                            break

                        # 暂时没定这个得分的类型，说明是N
                        if not score_frame_type:
                            score_frame_type = str(int(i)) + "_" + "N"
                        # 先把之前的得分记录下来
                        temp_goaltime[score_time] = score_frame_type
                        score_frame_type = ''
                        score_time_dic[score_time] += 1

                    # 这是新的得分
                    score_time = str_line[start_str:end_str]
                    pre_mode = "score"

                elif str_line[ii] == '+':
                    ii += 1
                    start_str = ii
                    while ii < len_line and str_line[ii].isdigit():
                        ii += 1
                    end_str = ii
                    plus_score_time = str_line[start_str:end_str]
                    if plus_score_time and score_time and pre_mode == "score":
                        score_time = score_time + '+' + plus_score_time
                        score_frame_type = str(int(i)) + "_" + "add"
                else:
                    ii += 1

            if len(player_name) > 2 and score_time and 0 < eval(score_time) <= 120:
                # 暂时没定这个得分的类型，说明是N
                if not score_frame_type:
                    score_frame_type = str(int(i)) + "_" + "N"
                temp_goaltime[score_time] = score_frame_type
                one_playertime = PlayerTime(player_name)
                one_playertime.goaltime = temp_goaltime
                temp_goaltime = defaultdict(str)
                if one_playertime.location == 0:
                    one_playertime.location = LorR

                player_name_dic[player_name] += 1
                score_time_dic[score_time] += 1

                Player_Time_list.append(one_playertime)
                del one_playertime
                big_flag = True

    return player_name_dic, score_time_dic, Player_Time_list, big_flag


def get_playback_index_from_json(json_path, video_index, fps):
    '''
    读取logo序列的json文件，匹配得到回放镜头
    :param json_path: json文件路径
    :param video_index: 视频名字（编号）
    :param fps: 视频帧率
    :return:
    '''
    with open(json_path + video_index + '_logo_frame_index.json', 'r') as fd:
        logo_frame_index = json.load(fd)
    fd.close()

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

    playback_index = []
    ii = 0
    cnt = 0
    while ii < len(logo_blocks) - 2:
        if cnt == 2:
            playback_index.append((logo_blocks[ii - 2], logo_blocks[ii - 1]))
            cnt = 0
        if (logo_blocks[ii + 1] - logo_blocks[ii]) < fps * 45 and (
                logo_blocks[ii + 1] - logo_blocks[ii]) < (
                logo_blocks[ii + 2] - logo_blocks[ii + 1]):
            playback_index.append((logo_blocks[ii], logo_blocks[ii + 1]))
            ii += 2
            cnt = 0
        else:
            ii += 1
            cnt += 1

    playback_index.append((logo_blocks[-2], logo_blocks[-1]))
    return playback_index


def check_teams(big_temp_result, big_w_index, frame_width, left_team, left_team_x, right_team, right_team_x):
    '''
    检查big_temp_result中的字符串，是否含有两个关于中轴对称的球队名
    同时将合适位置的字符串按左右队伍分类
    :param big_temp_result:
    :param big_w_index:
    :param frame_width:
    :return:
    '''
    team_candidates_list = []
    left_big_temp_result = []
    right_big_temp_result = []
    two_teams = False
    for ii, [strr, ration] in enumerate(big_temp_result[1]):
        x1_big = big_temp_result[0][ii][0][0]
        x2_big = big_temp_result[0][ii][2][0]
        y1_big = big_temp_result[0][ii][0][1]
        y2_big = big_temp_result[0][ii][3][1]

        # 排除在中间位置，以及长度不合理的字符串
        if abs((x1_big + x2_big) / 2 - big_w_index) < (frame_width * (1 / 25)):
            continue

        # 使用fuzzwuzz模糊匹配判断是否为球队名
        team1 = 0
        if len(strr) > 2:
            team1 = process.extractOne(strr, team_name, scorer=fuzz.ratio,
                                       score_cutoff=70)
        if team1:
            if team_candidates_list:
                for team_candidate in team_candidates_list:
                    # 两个检测出的队名必须满足：y坐标近似，x坐标关于中间对称
                    if abs(team_candidate[2] - (y1_big + y2_big) / 2) < 3 and \
                            abs((team_candidate[1] + (x1_big + x2_big) / 2) / 2 - big_w_index) < 20:
                        two_teams = True
                        if (x1_big + x2_big) / 2 > big_w_index:
                            right_team = team1[0]
                            left_team = team_candidate[0]
                            # 确定队名的x坐标（中心）用于后面排除太过边缘的字符串
                            left_team_x = team_candidate[1]
                            right_team_x = (x1_big + x2_big) / 2
                        if x2_big < big_w_index:
                            left_team = team1[0]
                            right_team = team_candidate[0]
                            left_team_x = (x1_big + x2_big) / 2
                            right_team_x = team_candidate[1]
                        team_y1 = y1_big
                        team_y2 = y2_big
                        break
            team_candidates_list.append([team1[0], (x1_big + x2_big) / 2, (y1_big + y2_big) / 2])
            continue

        # 根据文字框的坐标，将字符串及其坐标分别记录在 左候选集 和 右候选集 中
        if (x1_big + x2_big) / 2 < big_w_index:
            left_big_temp_result.append([strr, x1_big, x2_big, y1_big, y2_big])
        else:
            right_big_temp_result.append([strr, x1_big, x2_big, y1_big, y2_big])

    return two_teams, left_big_temp_result, right_big_temp_result, left_team, left_team_x, right_team, right_team_x


def save_a_frame(i, videoCap, big_h_index, video_name, Output_big_frames_path):
    '''
    保存第i帧的图像，便于debug
    :param i:
    :param videoCap:
    :param big_h_index:
    :param video_name:
    :return:
    '''
    videoCap.set(cv.CAP_PROP_POS_FRAMES, i)
    # i += 100;
    boolFrame, matFrame = videoCap.read()
    if boolFrame:
        temp_jpgframe = np.asarray(matFrame)
        # 截取上面0-90区域进行OCR检测
        if big_h_index < 250:
            jpgframe = temp_jpgframe[0:big_h_index]
        else:
            jpgframe = temp_jpgframe[big_h_index:]

        save_img_path = os.path.join(Output_big_frames_path,
                                     video_name + "_" + str(int(i)) + ".jpg")
        cv.imwrite(save_img_path, jpgframe)


def mysort(list):
    length = len(list)
    for i in range(0, length):
        for j in range(i + 1, length):
            if list[i][3] - list[j][3] > 3:
                list[i], list[j] = list[j], list[i]
            elif abs(list[i][3] - list[j][3]) < 3:
                if list[i][2] > list[j][2]:
                    list[i], list[j] = list[j], list[i]
    return list


def divid_rows(location, big_temp_result, team_y1, team_y2, team_x):
    '''
    把字符串分行放在一起
    :param location:
    :param big_temp_result:
    :param team_y1:
    :param team_y2:
    :param team_x:
    :return:
    '''
    # 对候选集中的字符串，首先根据y坐标进行排序，y坐标相同的根据x坐标排序
    # sorted_left_big_result = sorted(big_temp_result,
    #                                 key=lambda item: (item[3], item[2]))

    sorted_left_big_result = mysort(big_temp_result)

    yy1 = sorted_left_big_result[0][3]
    yy2 = sorted_left_big_result[0][4]
    oneline_0 = OneLine(sorted_left_big_result[0][3], sorted_left_big_result[0][4])
    line_index = 0
    result_line = []
    for ii, [strr, x1_big, x2_big, y1_big, y2_big] in enumerate(sorted_left_big_result):
        # 两种情况的字符串排除：
        # 1、和球队名在同一水平线上（差距小于4）；
        # 2、对于左边球队，球队名在球员名的右边且超过100
        if abs((y1_big + y2_big) / 2 - (team_y1 + team_y2) / 2) < 4 or \
                (location == "left" and team_x - (x1_big + x2_big) / 2 > 130) \
                or (location == "right" and (x1_big + x2_big) / 2 - team_x > 130):
            continue

        # 因为之前先按y排序，再按x排序
        # 所以如果当前字符串的y值和本行的差别较大，就需要另起一行存
        if abs((yy1 + yy2) / 2 - (y1_big + y2_big) / 2) > 3:
            # locals()函数可以将字符串转换为变量名
            result_line.append(locals()['oneline_' + str(line_index)])
            del locals()['oneline_' + str(line_index)]
            line_index += 1
            locals()['oneline_' + str(line_index)] = OneLine(
                sorted_left_big_result[ii][3],
                sorted_left_big_result[ii][4])
            yy1 = y1_big
            yy2 = y2_big

        # 给当前行加上当前字符串，补个空格
        locals()['oneline_' + str(line_index)].strr += (strr + ' ')
    result_line.append(locals()['oneline_' + str(line_index)])
    del locals()['oneline_' + str(line_index)]

    return result_line


def merge_player_time_list(Player_Time_list, player_name_dic, score_time_dic):
    # 把Player_Time_list里的值（set）取交集，如果不为空，就在player_name_dic里面看谁的key次数多
    # 次数少的key就在big_candidate中删除
    nn = len(Player_Time_list)
    # names = list(Player_Time_list.keys())
    score_times = list(score_time_dic.keys())
    for i in range(nn):
        if Player_Time_list[i].use == 1:
            i_score_times = list(Player_Time_list[i].goaltime.keys())
            i_playername = Player_Time_list[i].playername

            # 对于其他球员
            for j in range(i + 1, nn):
                if Player_Time_list[j].use == 1:
                    j_score_times = list(Player_Time_list[j].goaltime.keys())
                    j_playername = Player_Time_list[j].playername
                    set_temp = set(i_score_times) & set(j_score_times)
                    if set_temp:
                        if player_name_dic[i_playername] > player_name_dic[
                            j_playername]:
                            # Player_Time_list[i].goaltime.update(Player_Time_list[j].goaltime)
                            Player_Time_list[j].use = 0
                        else:
                            # Player_Time_list[j].goaltime.update(Player_Time_list[i].goaltime)
                            Player_Time_list[i].use = 0
                            break
                    elif fuzz.token_set_ratio(i_playername, j_playername) >= 70:
                        # 处理进球时间微调的情况，+1或者-1，需要删除前面的，而不是直接并集
                        i_times = list(Player_Time_list[i].goaltime.keys())
                        j_times = list(Player_Time_list[j].goaltime.keys())
                        for j_time in j_times:
                            jj_time = eval(j_time)
                            for i_time in i_times:
                                if Player_Time_list[i].goaltime[i_time]:
                                    if j_time == i_time:
                                        continue
                                    else:

                                        ii_time = eval(i_time)
                                        if abs(jj_time - ii_time) == 1:
                                            Player_Time_list[i].goaltime.pop(i_time)
                        # 根据次数多少选择保留的对象
                        if player_name_dic[i_playername] > player_name_dic[
                            j_playername]:
                            Player_Time_list[i].goaltime.update(
                                Player_Time_list[j].goaltime)
                            Player_Time_list[j].use = 0
                        else:
                            Player_Time_list[j].goaltime.update(
                                Player_Time_list[i].goaltime)
                            Player_Time_list[i].use = 0
                            break
            # 与其他的比较后：
            if Player_Time_list[i].use == 1:
                # 如果该运动员的某个得分时间次数出现少于5，就删除这个得分时间
                for score_time in list(Player_Time_list[i].goaltime.keys()):
                    if score_time_dic[score_time] < 5:
                        Player_Time_list[i].goaltime.pop(score_time)
                # 如果该球员没有得分时间，就删除该球员的记录
                if not Player_Time_list[i].goaltime:
                    Player_Time_list[i].use = 0
                # 如果该球员名字出现次数较少，就删除该球员的记录（有问题）
                if player_name_dic[i_playername] < 3:
                    Player_Time_list[i].use = 0

    # return Player_Time_list

def show_player_text_summarization(Player_Time_list, time_line, left_team, right_team):
    nn = len(Player_Time_list)
    for i in range(nn):
        if Player_Time_list[i].use:
            name = Player_Time_list[i].playername
            if Player_Time_list[i].location == 1:
                team = left_team
            else:
                team = right_team
            # logging.info("name:{}  team:{}".format(name, team))
            # logging.info("姓名:{}  队伍:{}".format(name, team))
            # logging.info(Player_Time_list[i].goaltime)

            goal_num_player = len(Player_Time_list[i].goaltime)
            if goal_num_player == 1:
                key = list(Player_Time_list[i].goaltime.keys())[0]
                goal_type = Player_Time_list[i].goaltime[key].split('_')[1]
                time_line[key] = name + '_' + team + '_' + goal_type
                if goal_type == 'N':
                    # logging.info("{} from {} shot a goal at {}\'".format(name, team, key))
                    logging.info("来自 {} 的 {} 在第 {} 分钟射入一球！\'".format(team, name, key))
                elif goal_type == 'P':
                    # logging.info("{} from {} shot a goal at {}\', and is a penalty kick!".format(name, team, key))
                    logging.info("来自 {} 的 {} 在第 {} 分钟点球得分！\'".format(team, name, key))
                elif goal_type == 'OG':
                    # logging.info("{} from {} shot a OWN GOAL at {}\', what a pity!".format(name, team, key))
                    logging.info(
                        "来自 {} 的 {} 在第 {} 分钟造成一粒乌龙球，真可惜！\'".format(team, name, key))
                elif goal_type == 'add':
                    # logging.info("{} from {} shot a goal at {}\' in added time".format(name, team, key))
                    logging.info(
                        "来自 {} 的 {} 在伤停补时阶段第 {} 分钟射入一球！\'".format(team, name, key))
                # logging.info("**********************************************************")
            elif goal_num_player == 2:
                keys = list(Player_Time_list[i].goaltime.keys())
                goal_type_0 = Player_Time_list[i].goaltime[keys[0]].split('_')[1]
                goal_type_1 = Player_Time_list[i].goaltime[keys[1]].split('_')[1]
                time_line[keys[0]] = name + '_' + team + '_' + goal_type_0
                time_line[keys[1]] = name + '_' + team + '_' + goal_type_1

                # logging.info(
                #     "梅开二度! {} from {} shot two goals! One is at {}\', another is at {}\'".format(name, team,
                #                                                                                  keys[0],
                #                                                                                  keys[1]))
                logging.info(
                    "梅开二度! 来自 {} 的 {} 分别在比赛第 {} 分钟和第 {} 分钟射门得分！".format(team, name,
                                                                        keys[0],
                                                                        keys[1]))
                # logging.info("**********************************************************")
            elif goal_num_player == 3:
                keys = list(Player_Time_list[i].goaltime.keys())
                goal_type_0 = Player_Time_list[i].goaltime[keys[0]].split('_')[1]
                goal_type_1 = Player_Time_list[i].goaltime[keys[1]].split('_')[1]
                goal_type_2 = Player_Time_list[i].goaltime[keys[2]].split('_')[1]
                time_line[keys[0]] = name + '_' + team + '_' + goal_type_0
                time_line[keys[1]] = name + '_' + team + '_' + goal_type_1
                time_line[keys[2]] = name + '_' + team + '_' + goal_type_2
                # logging.info(
                #     "帽子戏法! {} from {} shot three goals! first is at {}\', second is at {}\', and the "
                #     "third is at {}\'".format(name, team,
                #                               keys[0],
                #                               keys[1], keys[2]))
                logging.info(
                    "帽子戏法! 来自 {} 的 {} 分别在比赛第 {} 分钟、第 {} 分钟，和第 {} 分钟射门得分！".format(team,
                                                                                 name,
                                                                                 keys[
                                                                                     0],
                                                                                 keys[
                                                                                     1],
                                                                                 keys[
                                                                                     2]))
                # logging.info("**********************************************************")
            elif goal_num_player == 4:
                keys = list(Player_Time_list[i].goaltime.keys())
                goal_type_0 = Player_Time_list[i].goaltime[keys[0]].split('_')[1]
                goal_type_1 = Player_Time_list[i].goaltime[keys[1]].split('_')[1]
                goal_type_2 = Player_Time_list[i].goaltime[keys[2]].split('_')[1]
                goal_type_3 = Player_Time_list[i].goaltime[keys[3]].split('_')[1]
                time_line[keys[0]] = name + '_' + team + '_' + goal_type_0
                time_line[keys[1]] = name + '_' + team + '_' + goal_type_1
                time_line[keys[2]] = name + '_' + team + '_' + goal_type_2
                time_line[keys[3]] = name + '_' + team + '_' + goal_type_3

                logging.info(
                    "大四喜! 来自 {} 的 {} 分别在比赛第 {} 分钟、第 {} 分钟、第 {} 分钟，和第 {} 分钟射门得分！".format(
                        team, name,
                        keys[0],
                        keys[1], keys[2], keys[3]))
            elif goal_num_player == 5:
                keys = list(Player_Time_list[i].goaltime.keys())
                goal_type_0 = Player_Time_list[i].goaltime[keys[0]].split('_')[1]
                goal_type_1 = Player_Time_list[i].goaltime[keys[1]].split('_')[1]
                goal_type_2 = Player_Time_list[i].goaltime[keys[2]].split('_')[1]
                goal_type_3 = Player_Time_list[i].goaltime[keys[3]].split('_')[1]
                goal_type_4 = Player_Time_list[i].goaltime[keys[4]].split('_')[1]
                time_line[keys[0]] = name + '_' + team + '_' + goal_type_0
                time_line[keys[1]] = name + '_' + team + '_' + goal_type_1
                time_line[keys[2]] = name + '_' + team + '_' + goal_type_2
                time_line[keys[3]] = name + '_' + team + '_' + goal_type_3
                time_line[keys[4]] = name + '_' + team + '_' + goal_type_4
                logging.info(
                    "五子登科! 来自 {} 的 {} 分别在比赛第 {} 分钟、第 {} 分钟、第 {} 分钟、第 {} 分钟，和第 {} 分钟射门得分！".format(
                        team, name,
                        keys[0],
                        keys[1], keys[2],
                        keys[3], keys[4]))

    time_line_sorted = sorted(time_line.items(), key=lambda d: eval(d[0]),
                              reverse=False)

    return time_line_sorted

def show_time_line_and_gen_video(time_line_sorted, left_team, right_team):
    # 根据进球时间排序，生成进球时间时间线

    time_line_texts = {}
    team_order = defaultdict(list)
    video_texts = {}

    logging.info("------Generating goal vides------")
    logging.info("------生成进球视频片段------")
    for key in time_line_sorted:
        goal_time_minute = key[0]
        name = key[1].split('_')[0]
        team = key[1].split('_')[1]
        goal_type = key[1].split('_')[2]
        time_line_text = ' '
        if goal_type == 'N':
            if team == left_team:
                team_order[left_team].append(key)
                time_line_text = name + " shot a goal" + " At " + key[0] + '\''
                video_text = name + " from " + team + " shot a goal!" + " At " + goal_time_minute + '\''
            else:
                team_order[right_team].append(key)
                time_line_text = "                                          At " + goal_time_minute + '\' ' + name + " shot a goal"
                video_text = "At " + goal_time_minute + '\' ' + name + " from " + team + " shot a goal!"
        elif goal_type == 'P':
            if team == left_team:
                team_order[left_team].append(key)
                time_line_text = name + " shot a penalty goal" + " At " + goal_time_minute + '\''
                video_text = name + " from " + team + " shot a goal!, and is a penalty kick!" + " At " + \
                             goal_time_minute + '\''
            else:
                team_order[right_team].append(key)
                time_line_text = "                                          At " + goal_time_minute + '\', ' + name + " shot a penalty goal"
                video_text = "At " + goal_time_minute + '\', ' + name + " from " + team + " shot a goal!, and is a penalty kick!"
        elif goal_type == 'OG':
            if team == left_team:
                team_order[right_team].append(key)
                time_line_text = name + " shot a OWN GOAL" + " At " + goal_time_minute + '\''
                video_text = name + " from " + team + " shot a OWN GOAL, what a pity!" + " At " + \
                             goal_time_minute + '\''
            else:
                team_order[left_team].append(key)
                time_line_text = "                                          At " + goal_time_minute + '\', ' + name + " shot a OWN GOAL"
                video_text = "At " + goal_time_minute + '\', ' + name + " from " + team + " shot a OWN GOAL, what a pity!"
        elif goal_type == 'add':
            if team == left_team:
                team_order[left_team].append(key)
                time_line_text = name + " shot a goal!" + " At " + goal_time_minute + '\''
                video_text = name + " from " + team + " shot a goal in added time" + " At " + \
                             goal_time_minute + '\''
            else:
                team_order[right_team].append(key)
                time_line_text = "                                          At " + goal_time_minute + '\' ' + name + " shot a goal!"
                video_text = "At " + goal_time_minute + '\' ' + name + " from " + team + " shot a goal in added time"
        time_line_texts[time_line_text] = team
        video_texts[goal_time_minute] = video_text
        # logging.info("At {}\', {} from {} shot a goal!".format(key[0], name, team))

    return time_line_texts, team_order, video_texts


def time_to_frame_index(goal_time_minute, video_type, time_refer, fps):
    int_goal_time = eval(goal_time_minute)
    goal_frame_index = 0
    if video_type == 'second' and int_goal_time < 45:
        logging.info(
            "goal time {} out of this video, this video is a second part of tha game!".format(
                int_goal_time))
    else:
        if int_goal_time < 45:
            time_ref = time_refer[0]
        else:
            time_ref = time_refer[1]
        ref_frame_index = time_ref[0]
        ref_minute = int(time_ref[1].split(':')[0])
        if ref_minute < int_goal_time:
            goal_frame_index = int(
                ref_frame_index + (int_goal_time - ref_minute) * 60 * fps)
        else:
            goal_frame_index = int(
                ref_frame_index - (ref_minute - int_goal_time) * 60 * fps)

    return goal_frame_index



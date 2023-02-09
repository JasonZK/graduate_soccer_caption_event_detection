# 检测进球得分事件，生成相应的文字摘要和视频摘要
# 需要提前有该视频的回放镜头json文件
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

keyword1 = "goal"

if_gen_video = False


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


EVENT_DIR = "D:/dataset/event"

# VIDEO_DIR = "D:/dataset/video_6"
# Output_big_frames_path = "D:/dataset/A_graduate_experiment/big_frames_19Asia/"

VIDEO_DIR = "D:/dataset/video_6_error"
Output_big_frames_path = "D:/dataset/A_graduate_experiment/big_frames_19Asia_error/"

video_output_path = "D:/dataset/output_goal_video/"
makedir(video_output_path)

Json_18Russia_path = "D:/study/proj/yolo/yolov5-master/18Russia_json/"

ocr = PaddleOCR(lang="en", gpu_mem=5000, det_model_dir="./padOCR_inference/en_PP-OCRv3_det_infer",
                rec_model_dir="./padOCR_inference/en_PP-OCRv3_rec_infer/", show_log=False)  # 首次执行会自动下载模型文件

# nlp = spacy.load("en_core_web_md")

# 获取队名文件
team_name = []
f = open("team_name_full.txt", "r")
for line in f:
    line = line[:-1]
    team_name.append(line)


import logging

#默认的warning级别，只输出warning以上的
#使用basicConfig()来指定日志级别和相关信息

logging.basicConfig(level=logging.DEBUG #设置日志输出格式
                    ,filename="log/Goal_detection.log" #log日志输出的文件位置和文件名
                    ,filemode="w+" #文件的写入格式，w为重新写入文件，默认是追加
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )


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


def gen_goal_video(video_name, videoCap, videoClip, video_type, time_refer, goal_time, player, team, playback_index,
                   video_output_path, text):
    frame_height = int(videoCap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(videoCap.get(cv.CAP_PROP_FRAME_WIDTH))
    fps = int(videoCap.get(cv.CAP_PROP_FPS))
    frame_count = int(videoCap.get(cv.CAP_PROP_FRAME_COUNT))
    int_goal_time = eval(goal_time)
    if video_type == 'second' and int_goal_time < 45:
        logging.info("goal time {} out of this video, this video is a second part of tha game!".format(int_goal_time))
    else:
        if int_goal_time < 45:
            time_ref = time_refer[0]
        else:
            time_ref = time_refer[1]
        ref_frame_index = time_ref[0]
        ref_minute = int(time_ref[1].split(':')[0])
        if ref_minute < int_goal_time:
            goal_frame_index = int(ref_frame_index + (int_goal_time - ref_minute) * 60 * fps)
        else:
            goal_frame_index = int(ref_frame_index - (ref_minute - int_goal_time) * 60 * fps)

        if goal_frame_index < 1 or goal_frame_index >= frame_count:
            logging.info("generate video failed! goal time:{}".format(goal_time))
        else:
            fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
            video_width, video_height = frame_width, frame_height

            save_path = video_output_path + video_name + '_' + player + goal_time + '_Playback_audio' + '.mp4'
            # videoWriter = cv.VideoWriter(save_path, fourcc, fps, (video_width, video_height))
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

            # for f_index in range(start - fps * 2, end + fps * 2):
            #     videoCap.set(cv.CAP_PROP_POS_FRAMES, f_index)
            #     # i += 100;
            #     boolFrame, matFrame = videoCap.read()
            #     if boolFrame:
            #         temp_jpgframe = np.asarray(matFrame)
            #         text = player + " from " + team + " scores a goal at " + goal_time + "!"
            #         cv.putText(temp_jpgframe, text, (200, 50), cv.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
            #
            #         videoWriter.write(temp_jpgframe)
            # videoWriter.release()
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

        # 球员名在得分时间之前：Messi 34’
        if str_line[0].isalpha():
            ii = 0
            while ii < len_line:

                if str_line[ii].isalpha():
                    # 绝对是P
                    if pre_mode == "score" and str_line[ii].upper() == 'P' and \
                            ((ii + 1 == len_line) or (ii + 1 < len_line and not str_line[ii + 1].isalpha()) ):
                        score_frame_type = str(int(i)) + "_" + "P"
                        # 不是P，是新的P开头的人名
                        ii += 1
                    # 绝对是OG
                    elif pre_mode == "score" and str_line[ii].upper() == 'O' and \
                            ((ii + 2 == len_line and str_line[ii + 1].upper() == "G")
                             or (ii + 2 < len_line and str_line[ii + 1].upper() == "G" and not str_line[ii + 2].isalpha())):
                        score_frame_type = str(int(i)) + "_" + "OG"
                        OG_flag = True
                        ii += 1
                    else:
                        start_str = ii
                        while ii < len_line and (str_line[ii].isalpha() or str_line[ii] == ' '):
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
                    if pre_mode == "score" and str_line[ii].upper() == 'P' and \
                            ((ii + 1 == len_line) or (ii + 1 < len_line and not str_line[ii + 1].isalpha())):
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
                        while ii < len_line and (str_line[ii].isalpha() or str_line[ii] == ' '):
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
                    start_str = ii + 1
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

    # for ii, c in enumerate(str_line):
    #
    #     # 字母
    #     if c.isalpha():
    #         # 判断是否是P
    #         if pre_mode == 'score' and c.upper() == 'P':
    #             # 后面还有字符，说明不是P，而是后面P开头的人名
    #             if ii < len_line-4 and str_line[ii+1].isalpha():
    #                 # 如果已经有球员名，而之前mode为score，说明信息足够，可以新建对象
    #                 if len(player_name) > 2:
    #                     one_playertime = PlayerTime(player_name)
    #                     one_playertime.location = LorR
    #                     score_frame_type = str(int(i)) + "_" + "N"
    #                     one_playertime.goaltime[score_time] = score_frame_type
    #                     score_time_dic[score_time] += 1
    #
    #                     # 初始化下一个球员，mode为player
    #                     player_name = c
    #                     score_time = ""
    #                     score_frame_type = ""
    #                     pre_mode = "player"
    #
    #                 # 没有球员名，球员名在进球时间后面
    #                 else:
    #                     player_name = c
    #                     pre_mode = 'player'
    #             # 说明是P
    #             else:
    #                 score_frame_type = str(int(i)) + "_" + 'P'
    #                 if one_playertime:
    #                     one_playertime.goaltime[score_time] = score_frame_type
    #                 # elif player_name
    #
    #         # 当前字符是字母，之前为空或者之前是球员名，那么这个字母归在球员名中
    #         elif pre_mode == 'player' or pre_mode == 'none':
    #             if pre_mode == 'none':
    #                 pre_mode == 'player'
    #             if pre_c == ' ' and pre_mode == 'player':
    #                 player_name += ' '
    #             player_name += c
    #         # 当前字符是字母，之前是数字，说明是一行两个
    #         elif pre_mode == 'score' and player_name != '':
    #             continue
    #
    #     # 数字
    #     elif c.isdigit():
    #         continue


def check_line(result_line, LorR, player_name_dic, big_candidate, score_time_dic, i, Player_Time_list, big_flag):
    for lline in result_line:
        P = 0
        OG = 0
        score_time = '0'
        # string_list = re.findall("[A-Za-z]+", lline.strr)
        string_list = re.findall("[A-Za-z]+?[A-Za-z]+", lline.strr)
        number_list = re.findall("\d+", lline.strr)

        if (not string_list) or (not number_list):
            continue

        player_name = ''
        # 这里目前只处理了普通的P、90加时的P以及普通的OG
        # 还可能有45加时的P，以及45/90加时的OG
        # for string_one in string_list:
        #     if string_one.upper() == 'P':
        #         P = 1
        #         P_add90_score_time_list = re.findall("90\D+\d\'? ?\(?P|90\D+\d\'? ?\[?P", lline.strr)
        #         if P_add90_score_time_list:
        #             add_number = re.findall("\d", P_add90_score_time_list[0])[0]
        #             score_time = '90+' + add_number
        #         score_time_list = re.findall("\d+\'? ?\(?P|\d+\'? ?\[?P", lline.strr)
        #         if score_time_list:
        #             score_time = re.findall("\d+", score_time_list[0])[0]
        #     elif string_one.upper() == 'OG':
        #         OG = 1
        #         score_time_list = re.findall("\d+\'? ?\(?OG|\d+\'? ?\[?OG", lline.strr)
        #         if score_time_list:
        #             score_time = re.findall("\d+", score_time_list[0])[0]
        #     else:
        #         player_name += (string_one + ' ')

        for string_one in string_list:
            if string_one.upper() != 'P' and string_one.upper() != 'OG':
                player_name += (string_one + ' ')

        if not player_name or len(player_name) < 3:
            continue

        player_name_dic[player_name] += 1

        # 检测P
        # 检测90+ P
        P_add90_score_time_list = re.findall("90\D+\d\'? ?\(?P|90\D+\d\'? ?\[?P", lline.strr)
        if P_add90_score_time_list:
            add_number = re.findall("\d", P_add90_score_time_list[0])[0]
            score_time = '90+' + add_number
            P = 1
        score_time_list = re.findall("\d+\'? ?\(?P|\d+\'? ?\[?P", lline.strr)
        if score_time_list:
            score_time = re.findall("\d+", score_time_list[0])[0]
            P = 1
        # 检测OG
        score_time_list = re.findall("\d+\'? ?\(?OG|\d+\'? ?\[?OG", lline.strr)
        if score_time_list:
            score_time = re.findall("\d+", score_time_list[0])[0]
            OG = 1

        one_playertime = PlayerTime(player_name)
        one_playertime.location = LorR
        if P and score_time != '0':
            one_playertime.goaltime[score_time] = str(int(i)) + "_" + 'P'
        elif OG and score_time != '0':
            one_playertime.goaltime[score_time] = str(int(i)) + "_" + 'OG'
            one_playertime.location = -LorR

        add_number = ''
        for number_one in number_list:
            # if number_one == '3':
            #     logging.info("a")
            num_start = lline.strr.find(number_one)  # 0
            end = num_start + len(number_one) - 1
            if end + 1 < len(lline.strr) and lline.strr[end + 1].isalpha():
                continue

            # 如果这个数字是90后面加的，就跳过
            if number_one == add_number:
                continue
            # 如果数字是90，说明有加时
            if number_one == '90' or number_one == '45':
                add_time = 1
                # add_score_time_list = re.findall("\+\d\'?", lline.strr)
                add90_score_time_list = re.findall("90\D+\d", lline.strr)
                add45_score_time_list = re.findall("45\D+\d", lline.strr)
                if add90_score_time_list:
                    add_number = add90_score_time_list[0][-1]
                    score_time = '90+' + add_number
                    # 如果这个进球不是P或者OG
                    if not one_playertime.goaltime[score_time]:
                        one_playertime.goaltime[score_time] = str(int(i)) + "_" + "add"
                elif add45_score_time_list:
                    add_number = add45_score_time_list[0][-1]
                    score_time = '45+' + add_number
                    # 如果这个进球不是P或者OG
                    if not one_playertime.goaltime[score_time]:
                        one_playertime.goaltime[score_time] = str(int(i)) + "_" + "add"
                else:
                    score_time = number_one
                    # 如果这个进球不是P或者OG
                    if not one_playertime.goaltime[score_time]:
                        one_playertime.goaltime[score_time] = str(int(i)) + "_" + "N"
                big_candidate[player_name].add(score_time)
                score_time_dic[score_time] += 1
                big_flag = True
            # 正常进球情况
            # 理论上应该小于3，但是有1302特殊情况时间有115，101
            elif eval(number_one) <= 120:
                score_time = number_one
                big_candidate[player_name].add(score_time)
                # 如果这个进球不是P或者OG
                if not one_playertime.goaltime[score_time]:
                    one_playertime.goaltime[score_time] = str(int(i)) + "_" + "N"
                score_time_dic[score_time] += 1
                big_flag = True

        Player_Time_list.append(one_playertime)
        # logging.info("source string: {}   check players name: {}  goal_time: {}  frame:{}  goaltime:{}".format(
        #     lline.strr, player_name, big_candidate[player_name], i, one_playertime.goaltime))
        del one_playertime
        continue

    return player_name_dic, big_candidate, score_time_dic, Player_Time_list, big_flag


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
                            abs((team_candidate[1] + (x1_big + x2_big) / 2) / 2 - big_w_index) < 10:
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


def save_a_frame(i, videoCap, big_h_index, video_name):
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
                (location == "left" and team_x - (x1_big + x2_big) / 2 > 100) \
                or (location == "right" and (x1_big + x2_big) / 2 - team_x > 100):
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


def get_frames(video_dir):
    get = 0
    miss = 0
    total_num = 0
    more = 0
    # with open('video_6_score_change.json', 'r') as fd:
    #     Video_Score_dic = json.load(fd)
    SCORES = {}
    makedir(Output_big_frames_path)

    for root, _, files in os.walk(video_dir):
        for video_index in files:
            time1 = time.time()
            video_index1 = video_index.split('_')
            video_name = video_index1[1]

            if video_name not in ['0059', '0060', '0061', '0062',
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
                videoCap = cv.VideoCapture(video_dir + "\\" + video_index)
                videoClip = VideoFileClip(video_dir + "\\" + video_index)

                frame_height = videoCap.get(cv.CAP_PROP_FRAME_HEIGHT)
                frame_width = videoCap.get(cv.CAP_PROP_FRAME_WIDTH)
                small_h_index = int(frame_height * 3 / 10)
                fps = videoCap.get(cv.CAP_PROP_FPS)

                big_h_index = int(frame_height * 6 / 10)
                big_w_index = int(frame_width / 2)
                x1_big_center = big_w_index - 50
                x2_big_center = big_w_index + 50

                frame_count = videoCap.get(cv.CAP_PROP_FRAME_COUNT)
                logging.info("  ")
                logging.info(
                    "--------------------------------------------------------------------------------------------------------------")
                logging.info("video:{}".format(video_name))



                init_candidate = defaultdict(int)
                player_name_dic = defaultdict(int)
                score_time_dic = defaultdict(int)
                big_candidate = defaultdict(set)

                location_player_score = 0  # 得分球员位置，为0代表在队名下面，为1代表在队名上面
                tempResult_index = 0
                mode = 0
                modify_candidate = defaultdict(int)
                temp_result = []
                gt_y2 = 0
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

                i = frame_count - 30000
                # i = 0
                i = 58292

                # 遍历视频
                while i < frame_count:
                    # if i > 73499:
                    #     logging.info("a")
                    # logging.info("deal with {}".format(i))

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
                        right_team, right_team_x = check_teams(big_temp_result, big_w_index, frame_width, left_team, left_team_x, right_team, right_team_x)

                        # 检测出了两个球队名，说明是大字幕，此时team_y1，team_y2都有了
                        if two_teams:

                            # 下面这一段是将确定为大字幕的帧保存下来，debug的时候方便看
                            save_a_frame(i, videoCap, big_h_index, video_name)

                            # # doc = nlp(big_result)
                            # # logging.info(big_result)
                            # # logging.info([(ent.text, ent.label_) for ent in doc.ents])

                            # 接下来分别对 左候选集 和 右候选集 进行分析处理
                            if left_big_temp_result:
                                left_line = divid_rows("left", left_big_temp_result, team_y1, team_y2, left_team_x)

                            if right_big_temp_result:
                                right_line = divid_rows("right", right_big_temp_result, team_y1, team_y2, right_team_x)

                            # 排序、分行完成后，进行逐行细致分析
                            # player_name_dic, big_candidate, score_time_dic, Player_Time_list, find_flag = check_line(
                            #     left_line, 1, player_name_dic, big_candidate, score_time_dic, i, Player_Time_list,
                            #     find_flag)
                            #
                            # player_name_dic, big_candidate, score_time_dic, Player_Time_list, find_flag = check_line(
                            #     right_line, -1, player_name_dic, big_candidate, score_time_dic, i,
                            #     Player_Time_list,
                            #     find_flag)

                            player_name_dic, score_time_dic, Player_Time_list, find_flag = check_line2(
                                left_line, 1, player_name_dic, score_time_dic, i, Player_Time_list, find_flag)

                            player_name_dic, score_time_dic, Player_Time_list, find_flag = check_line2(
                                right_line, -1, player_name_dic, score_time_dic, i, Player_Time_list, find_flag)

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

                # 把big_candidate里的值（set）取交集，如果不为空，就在player_name_dic里面看谁的key次数多
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
                                    if player_name_dic[i_playername] > player_name_dic[j_playername]:
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
                                    if player_name_dic[i_playername] > player_name_dic[j_playername]:
                                        Player_Time_list[i].goaltime.update(Player_Time_list[j].goaltime)
                                        Player_Time_list[j].use = 0
                                    else:
                                        Player_Time_list[j].goaltime.update(Player_Time_list[i].goaltime)
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

                Summary_Json = {}
                Summary_Json['teamA'] = left_team
                Summary_Json['teamB'] = right_team
                random_events = {}
                goal_event = []
                time_goal_event = []
                time_line = {}

                logging.info("##################### video:{}  goal events ####################".format(video_index1[1]))
                logging.info("### showed by players ###")
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
                                logging.info("来自 {} 的 {} 在第 {} 分钟造成一粒乌龙球，真可惜！\'".format(team, name, key))
                            elif goal_type == 'add':
                                # logging.info("{} from {} shot a goal at {}\' in added time".format(name, team, key))
                                logging.info("来自 {} 的 {} 在伤停补时阶段第 {} 分钟射入一球！\'".format(team, name, key))
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
                                "帽子戏法! 来自 {} 的 {} 分别在比赛第 {} 分钟、第 {} 分钟，和第 {} 分钟射门得分！".format(team, name,
                                                                                             keys[0],
                                                                                             keys[1], keys[2]))
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
                                "大四喜! 来自 {} 的 {} 分别在比赛第 {} 分钟、第 {} 分钟、第 {} 分钟，和第 {} 分钟射门得分！".format(team, name,
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
                                "五子登科! 来自 {} 的 {} 分别在比赛第 {} 分钟、第 {} 分钟、第 {} 分钟、第 {} 分钟，和第 {} 分钟射门得分！".format(team, name,
                                                                                                    keys[0],
                                                                                                    keys[1], keys[2],
                                                                                                    keys[3], keys[4]))
                        # for goal_time in list(Player_Time_list[i].goaltime.keys()):
                        #     time_goal_event.append([goal_time, Player_Time_list[i].goaltime[goal_time], name, team])
                        #
                        # gv = {'team': team, 'player': name, 'goal_time': Player_Time_list[i].goaltime}
                        # goal_event.append(gv)
                time_line_sorted = sorted(time_line.items(), key=lambda d: eval(d[0]), reverse=False)
                time_line_texts = {}
                team_order = defaultdict(list)

                logging.info("------Generating goal vides------")
                logging.info("------生成进球视频片段------")
                for key in time_line_sorted:
                    name = key[1].split('_')[0]
                    team = key[1].split('_')[1]
                    goal_type = key[1].split('_')[2]
                    time_line_text = ' '
                    if goal_type == 'N':
                        if team == left_team:
                            team_order[left_team].append(key)
                            time_line_text = name + " shot a goal" + " At " + key[0] + '\''
                            video_text = name + " from " + team + " shot a goal!" + " At " + key[0] + '\''
                        else:
                            team_order[right_team].append(key)
                            time_line_text = "                                          At " + key[
                                0] + '\' ' + name + " shot a goal"
                            video_text = "At " + key[0] + '\' ' + name + " from " + team + " shot a goal!"
                    elif goal_type == 'P':
                        if team == left_team:
                            team_order[left_team].append(key)
                            time_line_text = name + " shot a penalty goal" + " At " + key[0] + '\''
                            video_text = name + " from " + team + " shot a goal!, and is a penalty kick!" + " At " + \
                                         key[0] + '\''
                        else:
                            team_order[right_team].append(key)
                            time_line_text = "                                          At " + key[
                                0] + '\', ' + name + " shot a penalty goal"
                            video_text = "At " + key[
                                0] + '\', ' + name + " from " + team + " shot a goal!, and is a penalty kick!"
                    elif goal_type == 'OG':
                        if team == left_team:
                            team_order[right_team].append(key)
                            time_line_text = name + " shot a OWN GOAL" + " At " + key[0] + '\''
                            video_text = name + " from " + team + " shot a OWN GOAL, what a pity!" + " At " + key[
                                0] + '\''
                        else:
                            team_order[left_team].append(key)
                            time_line_text = "                                          At " + key[
                                0] + '\', ' + name + " shot a OWN GOAL"
                            video_text = "At " + key[
                                0] + '\', ' + name + " from " + team + " shot a OWN GOAL, what a pity!"
                    elif goal_type == 'add':
                        if team == left_team:
                            team_order[left_team].append(key)
                            time_line_text = name + " shot a goal!" + " At " + key[0] + '\''
                            video_text = name + " from " + team + " shot a goal in added time" + " At " + key[0] + '\''
                        else:
                            team_order[right_team].append(key)
                            time_line_text = "                                          At " + key[
                                0] + '\' ' + name + " shot a goal!"
                            video_text = "At " + key[0] + '\' ' + name + " from " + team + " shot a goal in added time"
                    time_line_texts[time_line_text] = team
                    # logging.info("At {}\', {} from {} shot a goal!".format(key[0], name, team))
                    if if_gen_video:
                        gen_goal_video(video_index1[1], videoCap, videoClip, video_type, time_refer, key[0], name, team,
                                       playback_index, video_output_path, video_text)
                logging.info("------ done ------")

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
                            m2 = "最终结果是" + win_team + str(win_score) + "比" + str(lose_score) + "大胜" + lose_team + "。"
                        else:
                            m2 = "最终结果是" + win_team + str(win_score) + "比" + str(
                                lose_score) + "小胜" + lose_team + "，双方贡献了一场精彩的比赛！"

                    summary.append(m2)

                m4 = "以下是得分详情："
                summary.append(m4)
                for sens in summary:
                    logging.info(sens)

                logging.info("### showed by time line ###")
                logging.info("               {}                     time                     {}".format(left_team, right_team))
                for text in list(time_line_texts.keys()):
                    logging.info(text)

                logging.info("####################################################")

                time2 = time.time()
                logging.info("time for this video:{}".format(time2 - time1))

                # SCORES[video_index] = score_record

                # with open('scores_3' + '.json', 'w') as fd:
                #     json.dump(SCORES, fd)

    # logging.info("total_num:{}   get:{}   miss:{}  more:{}".format(total_num, get, miss, more))


get_frames(video_dir=VIDEO_DIR)

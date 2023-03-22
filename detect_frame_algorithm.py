import sys

MAX_INT = sys.maxsize


# 检测函数：对某个事件进行检测，返回True或False
def check(i):
    if i % 2 == 0:
        return True
    else:
        return False


# 视频总帧数
n = 100000

# 帧率：表示1s有多少帧
fps = 25

# 检测数阈值：表示确定检测到某个事件所需要的多帧检测次数
# 在大字幕检测（进球得分事件检测）等事件中，因为信息会动态变化，此时这个阈值可以定为MAX_INT，即检测到字幕彻底消失为止
# 在红黄牌事件或换人事件中，可设置阈值以降低检测成本
find_nums_thresh = 8

# 未检测阈值：表示没有检测到某个事件后，确定可以恢复常态化检测状态，所需要的多帧检测次数
notfind_nums_thresh = 5

i = 0
count = n
find_nums = 0
notfind_nums = 0
find_flag = False
while i < n:
    # 对第i帧进行某种检测
    if check(i):
        find_flag = True
    else:
        find_flag = False

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
            i += 1
        # 如果检测数已达到阈值，可以确定发生了某个事件，则进行一长段跳帧，即5s的长度（可变）
        # 5s的长度，一般可以保证已检测到的信息的字幕消失
        # 如果有连续的包含新信息的字幕出现，5s也不会完全跳过
        else:
            i += 5 * fps
            find_nums = 0

    # 本次没有检测到
    else:
        # 只给未检测数加1，暂时不把检测数归零
        notfind_nums += 1

        # 常态化检测
        # 如果是未检测数为极大，则处于常态化检测中，每隔视频的1s（即fps帧）检测一次
        # 1s检测1次，属于很细粒度的检测了，因为字幕一般出现时间不会小于3s，可根据情况加大这个间隔
        if notfind_nums == MAX_INT:
            i += fps

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
            i += 2*fps
            notfind_nums = MAX_INT
            find_nums = 0



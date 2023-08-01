import motmetrics as mm  # 导入该库
import numpy as np

metrics = list(mm.metrics.motchallenge_metrics)  # 即支持的所有metrics的名字列表
"""
['idf1', 'idp', 'idr', 'recall', 'precision', 'num_unique_objects', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations', 'mota', 'motp', 'num_transfer', 'num_ascend', 'num_migrate']
"""

# gt_file="F:\dataset/19th_tomato\sequoia_mot/5th\gt/gt.txt"
gt_file="F:\dataset/19th2_tomato\sequoia_mot/5th\gt/gt.txt"
"""  文件格式如下
1,0,1255,50,71,119,1,1,1
2,0,1254,51,71,119,1,1,1
3,0,1253,52,71,119,1,1,1
...
"""

# ts_file="F:\dataset/19th_tomato\sequoia_mot/7th/tracks/img1.txt"
# ts_file='F:\dataset/19th2_tomato\sequoia_mot/9th/tracks/bytetrack/img1.txt'
# ts_file='F:\code\Yolov5_StrongSORT_OSNet-master/runs/track\exp49/tracks/img1.txt'
ts_file='F:\code\yolov8_tracking-master/runs/track\exp54/tracks/img1.txt'
"""  文件格式如下
1,1,1240.0,40.0,120.0,96.0,0.999998,-1,-1,-1
2,1,1237.0,43.0,119.0,96.0,0.999998,-1,-1,-1
3,1,1237.0,44.0,117.0,95.0,0.999998,-1,-1,-1
...
"""
print('gt_file='+str(gt_file))
print('ts_file='+str(ts_file))
acc = mm.MOTAccumulator(auto_id=True)  #创建accumulator

# 用第一帧填充该accumulator
acc.update(
    [1, 2],                     # Ground truth objects in this frame
    [1, 2, 3],                  # Detector hypotheses in this frame
    [
        [0.1, np.nan, 0.3],     # Distances from object 1 to hypotheses 1, 2, 3
        [0.5,  0.2,   0.3]      # Distances from object 2 to hypotheses 1, 2, 3
    ]
)

# 查看该帧的事件
# print(acc.events) # a pandas DataFrame containing all events
"""
                Type  OId HId    D
FrameId Event
0       0        RAW    1   1  0.1
        1        RAW    1   2  NaN
        2        RAW    1   3  0.3
        3        RAW    2   1  0.5
        4        RAW    2   2  0.2
        5        RAW    2   3  0.3
        6      MATCH    1   1  0.1
        7      MATCH    2   2  0.2
        8         FP  NaN   3  NaN
"""

# 只查看MOT事件，不查看RAW
# print(acc.mot_events) # a pandas DataFrame containing MOT only events
"""
                Type  OId HId    D
FrameId Event
0       6      MATCH    1   1  0.1
        7      MATCH    2   2  0.2
        8         FP  NaN   3  NaN
"""

# 继续填充下一帧
frameid = acc.update(
    [1, 2],  # GT
    [1],     # hypotheses
    [
        [0.2],
        [0.4]
    ]
)
# print(acc.mot_events.loc[frameid])
"""
        Type OId  HId    D
Event
2      MATCH   1    1  0.2
3       MISS   2  NaN  NaN
"""

# 继续填充下一帧
frameid = acc.update(
    [1, 2], # GT
    [1, 3], # hypotheses
    [
        [0.6, 0.2],
        [0.1, 0.6]
    ]
)
# print(acc.mot_events.loc[frameid])
"""
         Type OId HId    D
Event
4       MATCH   1   1  0.6
5      SWITCH   2   3  0.6
"""


gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)  # 读入GT
ts = mm.io.loadtxt(ts_file, fmt="mot15-2D")  # 读入自己生成的跟踪结果


acc=mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)  # 根据GT和自己的结果，生成accumulator，distth是距离阈值
mh = mm.metrics.create()

# # 打印单个accumulator
# summary = mh.compute(acc,
#                      metrics=['num_frames', 'mota', 'motp'], # 一个list，里面装的是想打印的一些度量
#                      name='acc') # 起个名
# print(summary)
# """
#      num_frames  mota  motp
# acc           3   0.5  0.34
# """

# # 打印多个accumulators
# summary = mh.compute_many([acc, acc.events.loc[0:1]], # 多个accumulators组成的list
#                           metrics=['num_frames', 'mota', 'motp'], 
#                           name=['full', 'part']) # 起个名
# print(summary)
# """
#       num_frames  mota      motp
# full           3   0.5  0.340000
# part           2   0.5  0.166667
# """

# # 自定义显示格式
# strsummary = mm.io.render_summary(
#     summary,
#     formatters={'mota' : '{:.2%}'.format},  # 将MOTA的格式改为百分数显示
#     namemap={'mota': 'MOTA', 'motp' : 'MOTP'}  # 将列名改为大写
# )
# print(strsummary)
# """
#       num_frames   MOTA      MOTP
# full           3 50.00%  0.340000
# part           2 50.00%  0.166667
# """

# mh模块中有内置的显示格式
summary = mh.compute_many([acc, acc.events.loc[0:1]],
                          metrics=mm.metrics.motchallenge_metrics,
                          names=['full', 'part'])

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
"""
      IDF1   IDP   IDR  Rcll  Prcn GT MT PT ML FP FN IDs  FM  MOTA  MOTP
full 83.3% 83.3% 83.3% 83.3% 83.3%  2  1  1  0  1  1   1   1 50.0% 0.340
part 75.0% 75.0% 75.0% 75.0% 75.0%  2  1  1  0  1  1   0   0 50.0% 0.167
"""


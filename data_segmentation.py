import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 对原始数据进行替换、筛选
data = pd.read_csv("data_processed/Wimbledon_with_victor_labels_2.csv")
data.loc[(data.serve_depth=='CTL'),'serve_depth'] = 1
data.loc[(data.serve_depth=='NCTL'),'serve_depth'] = 0
data.loc[(data.return_depth=='D'),'return_depth'] = 1
data.loc[(data.return_depth=='ND'),'return_depth'] = 0
data.loc[(data.p1_score=='AD'),'p1_score'] = 50
data.loc[(data.p2_score=='AD'),'p2_score'] = 50

data['p1_score'] = data['p1_score'].astype(int)
data['p2_score'] = data['p2_score'].astype(int)



# x1_l,x2_l,x3_l,x4_l,x5_l,x6_l,x7_l,x8_l,x9_l,x10_l,x11_l,x12_l,x13_l,x14_l,x15_l,x16_l,x17_l=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

label_ls = []
x1_l, x2_l, x3_l, x4_l, x5_l, x6_l, x7_l, x8_l, x9_l, x10_l, x11_l, x12_l, x13_l, x14_l, x15_l, x16_l, x17_l = [[] for _ in range(17)]

for match_id, setno, gameno,pointno in zip(data.match_id,data.set_no,data.game_no,data.point_no):
    match = data[data.match_id==match_id]
    set = match[match.set_no==setno]
    game = set[set.game_no==gameno]
    point = game[game.point_no==pointno]
    label = 1 if point['point_victor'].values[0] == 1 else 0
    label_ls.append(label)
    # x1为一个match中领先的set数
    x1 = point['p1_sets'].values[0] - point['p2_sets'].values[0]   
    # x2为 当前game中领先的分数
    x2 = point['p1_score'].values[0] - point['p2_score'].values[0]
    # x3为是否为发球者
    x3 = 1 if point['serve_no'].values[0] == 1 else 0
    # x4为当前game中是否落后
    x4 = 0 if x2<0 else 1
    # x1为一个set中赢得的game数
    x5 = point['p1_games'].values[0]   
    x6 = 1 if 1 in game['p1_ace'].values else 0
    x7 = 1 if 1 in game['p1_winner'].values else 0
    x8 = 1 if 1 in game['p1_double_fault'].values else 0
    x9 = 1 if 1 in game['p1_unf_err'].values else 0
    # x10 为网前得分与上网次数的比例
    x10 = game['p1_net_pt_won'].sum()/game['p1_net_pt'].sum() if game['p1_net_pt'].sum()!= 0 else 0
    # x11 为赢得破局点与破局点数的比例
    x11 = set['p1_break_pt_won'].sum()/set['p1_break_pt'].sum() if game['p1_break_pt'].sum()!= 0 else 0
    index = match.index.tolist().index(point.index.tolist()[0])
    # x12为本场match的总跑动里程
    x12 = match.iloc[ :index+1]['p1_distance_run'].sum()
    #  x13为近三个points的总跑动里程
    x13 = match.iloc[index-2:index+1]['p1_distance_run'].sum()
    #  x14为上一point的跑动里程
    x14 = point['p1_distance_run'].values[0]
    #  x15球速
    x15 = point['speed_mph'].values[0]
    # x16为一个回合的击球数
    x16 = point['rally_count'].values[0]
    # x17发球或回球深度
    x17 = point['serve_depth'].values[0] * x3 + point['return_depth'].values[0] * (1 - x3)




    x1_l.append(x1)
    x2_l.append(x2)
    x3_l.append(x3)
    x4_l.append(x4)
    x5_l.append(x5)
    x6_l.append(x6)
    x7_l.append(x7)
    x8_l.append(x8)
    x9_l.append(x9)
    x10_l.append(x10)
    x11_l.append(x11)
    x12_l.append(x12)
    x13_l.append(x13)
    x14_l.append(x14)
    x15_l.append(x15)
    x16_l.append(x16)
    x17_l.append(x17)

dataset = pd.DataFrame({'x1':x1_l,'x2':x2_l,'x3':x3_l,'x4':x4_l,'x5':x5_l,'x6':x6_l,'x7':x7_l,'x8':x8_l,'x9':x9_l,'x10':x10_l,'x11':x11_l,'x12':x12_l,'x13':x13_l,'x14':x14_l,'x15':x15_l,'x16':x16_l,'x17':x17_l,'labels':label_ls})
# print(dataset)

Normalization = MinMaxScaler()
columns = dataset.columns[:-1]
# print(columns)
Normalization.fit(dataset[columns].values)
dataset[columns] = Normalization.transform(dataset[columns].values)
dataset.to_csv("Standard match_2023-wimbledon-1701_labels_2.csv",index=False)
# dataset.to_excel("Standard dataset.xlsx",index=False)
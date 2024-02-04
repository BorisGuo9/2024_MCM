from data_process import read_csv_to_dict
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("data_processed/invert_Wimbledon.csv")
data.loc[(data.serve_depth=='CTL'),'serve_depth'] = 1
data.loc[(data.serve_depth=='NCTL'),'serve_depth'] = 0
data.loc[(data.return_depth=='D'),'return_depth'] = 1
data.loc[(data.return_depth=='ND'),'return_depth'] = 0
data.loc[(data.p1_score=='AD'),'p1_score'] = 50
data.loc[(data.p2_score=='AD'),'p2_score'] = 50
data['p1_score'] = data['p1_score'].astype(int)
data['p2_score'] = data['p2_score'].astype(int)

data['game_victor'] = data['game_victor'].replace(0, pd.NA)
data['game_victor'] = data['game_victor'].fillna(method='ffill')
data['set_victor'] = data['set_victor'].replace(0, pd.NA)
data['set_victor'] = data['set_victor'].fillna(method='ffill')

data['match_victor'] = data['match_victor'].replace({0: 1, 1: 1})


# label_proc(data,'set_victor','set_no')
df = pd.DataFrame(data)
df.to_csv("data_processed/invert_Wimbledon_labels.csv",index=False)


# data = read_csv_to_dict("match_2023-wimbledon-1701_labels.csv")




# def label_proc(data,label_vic,label_no):
#     no = 0
#     count = 0
#     for row in range(no, data.shape[0]):   #data.shape[0]
#         if row == 7283:
#             for index in range(no-1 , no+count):
#                     # print(no+count)
#                     data.loc[index, label_vic] = data[label_vic][no+count]
#                     # data[label_vic][index] = data[label_vic][no+count]
#         else:
#             if data[label_no][row] == data[label_no][no]:
#                 count+= 1
#             else:
#                 for index in range(no-1 , no+count-1):
#                     data.loc[index, label_vic] = data[label_vic][no+count-1]
#                 no = no+count+1
#                 count = 0

# label_proc(data,'game_victor','game_no')

# # label_proc(data,'set_victor','set_no')
# df = pd.DataFrame(data)
# df.to_csv("data_processed/invert_Wimbledon_labels.csv",index=False)




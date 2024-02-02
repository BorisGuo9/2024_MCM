# import pytesseract
# from PIL import Image

# # Load the image from file
# img = Image.open("data/seg.png")

# # Use tesseract to do OCR on the image
# text = pytesseract.image_to_string(img)
# file_path = "data/extracted_text.txt"

# with open(file_path, "w") as file:
#     file.write(text)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("data/Wimbledon_featured_matches.csv")
df.loc[(df.p1_score=='AD'),'p1_score'] = 55
df.loc[(df.p2_score=='AD'),'p2_score'] = 55
# df.loc[(df.speed_mph=='NA'),'speed_mph'] = 0
df['p1_score'] = df['p1_score'].astype(int)
df['p2_score'] = df['p2_score'].astype(int)
df.dropna(subset=['speed_mph'], inplace=True)

x1_ls,x2_ls,x3_ls,x4_ls,x5_ls,x6_ls,x7_ls,x8_ls,x9_ls,x10_ls,x11_ls,x12_ls,x13_ls,x14_ls,x15_ls,x16_ls=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
label_ls = []


for match_id, set_no, game_no,point_no in zip(df.match_id,df.set_no,df.game_no,df.point_no):
    match = df[df.match_id==match_id]
    set_ = match[match.set_no==set_no]
    game_ = set_[set_.game_no==game_no]
    point_ = game_[game_.point_no==point_no]

    x1 = point_['p1_games'].values[0]
    x2 = point_['p1_score'].values[0] - point_['p2_score'].values[0]
    x3 = 1 if point_['serve_no'].values[0] == 1 else 0
    x4 = 0 if x2<0 else 1
    x5 = point_['p1_sets'].values[0] - point_['p2_sets'].values[0]       
    x6 = 1 if 1 in game_['p1_ace'].values else 0
    x7 = 1 if 1 in game_['p1_winner'].values else 0
    x8 = 1 if 1 in game_['p1_double_fault'].values else 0
    x9 = 1 if 1 in game_['p1_unf_err'].values else 0
    x10 = game_['p1_net_pt_won'].sum()/game_['p1_net_pt'].sum() if game_['p1_net_pt'].sum()!= 0 else 0
    x11 = set_['p1_break_pt_won'].sum()/set_['p1_break_pt'].sum() if game_['p1_break_pt'].sum()!= 0 else 0
    index = match.index.tolist().index(point_.index.tolist()[0])
    x12 = match.iloc[ :index+1]['p1_distance_run'].sum()
    x13 = match.iloc[index-2:index+1]['p1_distance_run'].sum()
    x14 = point_['p1_distance_run'].values[0]
    x15 = point_['speed_mph'].values[0]
    x16 = x15*x3

    label = 1 if point_['point_victor'].values[0] == 1 else 0
    label_ls.append(label)
    x1_ls.append(x1)
    x2_ls.append(x2)
    x3_ls.append(x3)
    x4_ls.append(x4)
    x5_ls.append(x5)
    x6_ls.append(x6)
    x7_ls.append(x7)
    x8_ls.append(x8)
    x9_ls.append(x9)
    x10_ls.append(x10)
    x11_ls.append(x11)
    x12_ls.append(x12)
    x13_ls.append(x13)
    x14_ls.append(x14)
    x15_ls.append(x15)
    x16_ls.append(x16)

dataset = pd.DataFrame({'x1':x1_ls,'x2':x2_ls,'x3':x3_ls,'x3':x3_ls,'x4':x4_ls,'x5':x5_ls,'x6':x6_ls,'x7':x7_ls,'x8':x8_ls,'x9':x9_ls,'x10':x10_ls,'x11':x11_ls,'x12':x12_ls,'x13':x13_ls,'x14':x14_ls,'x15':x15_ls,'x16':x16_ls,'labels':label_ls})
# print(dataset)

scaler = MinMaxScaler()
columns = dataset.columns[:-1]
# print(columns)
scaler.fit(dataset[columns].values)
dataset[columns] = scaler.transform(dataset[columns].values)
dataset.to_csv("Standard dataset.csv",index=False)

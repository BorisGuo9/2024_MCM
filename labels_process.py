from data_process import read_csv_to_dict
import matplotlib.pyplot as plt
import pandas as pd
from fig_network import save_to_csv

data = pd.read_csv("data/Wimbledon_with_victor_labels.csv")
data.loc[(data.serve_depth=='CTL'),'serve_depth'] = 1
data.loc[(data.serve_depth=='NCTL'),'serve_depth'] = 0
data.loc[(data.return_depth=='D'),'return_depth'] = 1
data.loc[(data.return_depth=='ND'),'return_depth'] = 0
data.loc[(data.p1_score=='AD'),'p1_score'] = 50
data.loc[(data.p2_score=='AD'),'p2_score'] = 50
data['p1_score'] = data['p1_score'].astype(int)
data['p2_score'] = data['p2_score'].astype(int)

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

# print(data)
data = data.to_dict(orient='records')
for row in data[3:4]:  
    print(row)
    
def invert_tennis_scores(data):
    def update_victor(current, total_points, threshold):
        """ Helper function to determine the victor based on total points and threshold """
        if total_points[current - 1] >= threshold and (total_points[current - 1] - total_points[2 - current]) >= 2:
            return current
        return 0

    for point in data:
        # Initialize temporary counters with current values
        temp_p1_points_won, temp_p2_points_won = point['p1_points_won'], point['p2_points_won']
        temp_p1_games, temp_p2_games = point['p1_games'], point['p2_games']
        temp_p1_sets, temp_p2_sets = point['p1_sets'], point['p2_sets']

        # Reverse point_victor
        point_victor_reversed = 3 - point['point_victor']
        # point_victor_reversed = point['point_victor']

        # Update temporary points won
        if point_victor_reversed == 1:
            temp_p1_points_won += 1
        else:
            temp_p2_points_won += 1

        # Determine game victor
        game_victor = update_victor(point_victor_reversed, [temp_p1_points_won, temp_p2_points_won], 4)

        # Update temporary games won and reset points
        if game_victor:
            if game_victor == 1:
                temp_p1_games += 1
            else:
                temp_p2_games += 1
            temp_p1_points_won, temp_p2_points_won = 0, 0

        # Determine set victor
        set_threshold = 7 if point['set_no'] < 5 else 10
        set_victor = update_victor(game_victor, [temp_p1_games, temp_p2_games], set_threshold)

        # Update temporary sets won and reset games
        if set_victor:
            if set_victor == 1:
                temp_p1_sets += 1
            else:
                temp_p2_sets += 1
            temp_p1_games, temp_p2_games = 0, 0

        # Determine match victor
        match_victor = 0
        if temp_p1_sets >= 3 or temp_p2_sets >= 3:
            match_victor = 1 if temp_p1_sets > temp_p2_sets else 2

        # Update the point data with temporary victor values
        point['point_victor'] = point_victor_reversed
        point['game_victor'] = game_victor
        point['set_victor'] = set_victor
        point['match_victor'] = match_victor

    return data

data = invert_tennis_scores(data)
# for row in data[3:4]:  
#     print(row)
# 指定 CSV 文件的名称
# filename = 'data/truth_Wimbledon.csv'
filename = 'data/invert_Wimbledon.csv'

save_to_csv(data, filename)

print(f"数据已保存到 {filename}")
from data_process import read_csv_to_dict
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('momentum_data/momentum of match_2023-wimbledon-1303.csv')
# 读取第一列的所有元素
match_values_pos = data.iloc[:, 0]
match_values_neg = data.iloc[:, 4]
match_values = match_values_pos - match_values_neg

set_values_pos = data.iloc[:, 1]
set_values_neg = data.iloc[:, 5]
set_values = set_values_pos - set_values_neg

game_values_pos = data.iloc[:, 2]
game_values_neg = data.iloc[:, 6]
game_values = game_values_pos - game_values_neg

point_values_pos = data.iloc[:, 3]
point_values_neg = data.iloc[:, 7]
point_values = point_values_pos - point_values_neg

momentum = match_values *0.4 + set_values *0.3 + game_values*0.2 +point_values*0.1


plt.figure
plt.plot(momentum)
plt.show()




















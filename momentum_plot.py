from data_process import read_csv_to_dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def find_last_of_consecutive_sequences(numbers):
    # 空列表或只有一个元素的处理
    if numbers.size == 0 or len(numbers) == 1:
        return numbers

    # 找出连续序列中的最后一个数
    last_numbers = []
    for i in range(len(numbers) - 1):
        # 如果当前数的下一个数不是连续的
        if numbers[i] + 1 != numbers[i + 1]:
            last_numbers.append(numbers[i])

    # 添加列表中的最后一个数，因为它总是某个序列的最后一个
    last_numbers.append(numbers[-1])

    return last_numbers



A = np.array([[1,   1.33, 1.67, 2],  # match
              [0.75, 1,   1.33, 1.67],  # set
              [0.6,  0.75, 1,   1.33],  # game
              [0.5,  0.6,  0.75, 1]])   # point

# A = np.array([[1,   1.33, 1.67,  2],  # match
#               [0.6, 1,    1.15, 1.33],   # set
#               [0.5, 0.87, 1,    1.15],  # game
#               [0.75, 1.5, 1.2, 1]])    # point


# Calculate the eigenvector and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(A)

# Find the index of the maximum eigenvalue
max_index = np.argmax(eigenvalues)

# The corresponding eigenvector is the AHP weights
weights = eigenvectors[:, max_index].real

# Normalizing the weights
weights = weights / np.sum(weights)

print(weights)



data = pd.read_csv('momentum_data/momentum of match_2023-wimbledon-1701.csv')
match_data = pd.read_csv('data_positive/data_segment/match_2023-wimbledon-1701.csv')

# break_point segmentation
break_point_rows = match_data[match_data['p1_break_pt'] == 1]
break_points_ls = np.array(break_point_rows['point_no'])
break_points_ls = [x - 1 for x in break_points_ls]

# Tiebreak_point segmentation
Tiebreak_point_rows = match_data[((match_data['p1_games'] == 6) & (match_data['p2_games'] == 6)) | ((match_data['p1_games'] == 9) & (match_data['p2_games'] == 9))]
Tiebreak_point_ls = np.array(Tiebreak_point_rows['point_no'])
# print(filtered_rows_ls)
Tiebreak_point_ls = find_last_of_consecutive_sequences(Tiebreak_point_ls) 
Tiebreak_point_ls = [x - 1 for x in Tiebreak_point_ls]




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

leverage = []

for p in range(0, len(match_values)):
    if p in break_points_ls:
        # lev = match_values[p] * 0.2 + set_values[p] * 0.2 + game_values[p] * 0.2 +point_values[p] * 0.4
        lev = match_values[p] * 0 + set_values[p] * 0 + game_values[p] * 0 +point_values[p] * 1
    elif p in Tiebreak_point_ls:
        lev = match_values[p] * 0.01 + set_values[p] * 0.02 + game_values[p] * 0.02 +point_values[p] * 0.95
    else:
        lev = match_values[p] * 0.3 + set_values[p] * 0.2 + game_values[p] * 0.2 +point_values[p] * 0.3
        # lev = match_values[p] * 0.25 + set_values[p] * 0.25 + game_values[p] * 0.25 +point_values[p] * 0.25
    leverage.append(lev)

# leverage = match_values * weights[0] + set_values * weights[2] + game_values * [3] +point_values*0.25 * [1]
# leverage = match_values * 0.25 + set_values * 0.25 + game_values * 0.25 +point_values*0.25
# leverage = match_values * 0.2 + set_values * 0.3 + game_values * 0.2 +point_values*0.3



def exponential_smoothing(leverage, alpha):
    """
    Apply exponential smoothing to a given data list.
    
    Parameters:
    data (list or pd.Series): A list of data points.
    alpha (float): The smoothing factor.
    
    Returns:
    list: The list of smoothed values.
    """
    smoothed_values = []  # Starting the smoothed values list with the first data point
    for t in range(0, len(leverage)):
        num_values = 0
        den_values = 1
        for i in range(0 , t+1):
            num_values = leverage[i] * (1 - alpha)**(t-i) + num_values
            den_values = (1 - alpha)**(t-i) + den_values
        smoothed_value = num_values / den_values
        smoothed_values.append(smoothed_value)
    return smoothed_values

# Example data and alpha value, replace leverage with your actual data list or Series
# leverage = [your_data_here]
alpha = 0.32

# Calculate the smoothed values for the entire list
smoothed_values = exponential_smoothing(leverage, alpha)
df_1 = pd.DataFrame(leverage)
df_1.to_csv('leverage.csv')
df_2 = pd.DataFrame(smoothed_values)
df_2.to_csv('momentum.csv')


# Plot the smoothed values
plt.figure()

# plt.plot(leverage)
plt.plot(smoothed_values)

# 在break points上添加红色散点
for bp in break_points_ls:
    plt.scatter(bp, smoothed_values[bp], color='red')  # 'leverage[bp]' 获取该点的杠杆值


# 在Tiebreak points上添加蓝色散点
for tbp in Tiebreak_point_ls:
    plt.scatter(tbp, smoothed_values[tbp], color='blue')  # 'leverage[bp]' 获取该点的杠杆值


plt.show()




















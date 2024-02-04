import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv("data/Wimbledon_featured_matches.csv")

# 初始化变量
momentum_data = []  # 存储动量数据
window_size = 5  
player1_score_prev=0
player2_score_prev=0


# 遍历每个比赛点
for i in range(len(data)):
    # 获取当前比赛点的信息
    match_id = data.loc[i, 'match_id']
    elapsed_time = data.loc[i, 'elapsed_time']
    player1_score = data.loc[i, 'p1_points_won']
    player2_score = data.loc[i, 'p2_points_won']
    server = data.loc[i, 'server']


    # 计算得分差和动量
    score_diff = player1_score - player2_score
    momentum = score_diff / window_size
                # 考虑发球优势
    if server == 1:  # 如果是发球方
        momentum += 0.1 * (data.loc[i, 'p1_points_won'] - player1_score_prev)
    else:  # 如果是非发球方
        momentum -= 0.1 * (data.loc[i, 'p2_points_won'] - player2_score_prev)
    # 将动量数据添加到列表中
    momentum_data.append({'match_id': match_id, 'elapsed_time': elapsed_time, 'momentum': momentum})
    
    # 更新前一点的得分，用于下一个数据点的发球优势计算
    player1_score_prev = player1_score
    player2_score_prev = player2_score

# 将动量数据转换为DataFrame
momentum_df = pd.DataFrame(momentum_data)
print(momentum_df['momentum'][0])
# 可视化动量变化
plt.figure()
plt.plot(momentum_df['momentum'], label='Momentum', color='blue')
plt.title('Momentum in Tennis Match')
plt.xlabel('Elapsed Time (minutes)')
plt.show()
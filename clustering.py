from data_process import read_csv_to_dict
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from adjustText import adjust_text

csv_data = read_csv_to_dict('data/Wimbledon_featured_matches.csv')
# for row in csv_data[3:4]:  
#     print(row)

def preprocess_data():
    # 读取CSV数据
    csv_data = read_csv_to_dict('data/Wimbledon_featured_matches.csv')
    data = pd.DataFrame(csv_data)
    
    # 创建每位选手的特征集
    player_stats = defaultdict(lambda: defaultdict(list))
    
    # 遍历数据，聚集每位选手的统计信息
    for _, row in data.iterrows():
        player_stats[row['player1']]['points_won'].append(row['p1_points_won'])
        player_stats[row['player1']]['distance_run'].append(row['p1_distance_run'])
        player_stats[row['player2']]['points_won'].append(row['p2_points_won'])
        player_stats[row['player2']]['distance_run'].append(row['p2_distance_run'])
    
    # 将数据转换为适用于聚类的格式
    features = []
    names = []
    for player, stats in player_stats.items():
        names.append(player)
        features.append([
            np.mean(stats['points_won']),
            np.mean(stats['distance_run'])
        ])
    
    return names, features

def cluster_players(n_clusters=4):
    names, features = preprocess_data()
    
    # 特征标准化
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # 应用K-means聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(scaled_features)
    
    # 获取聚类标签
    labels = kmeans.labels_
    
    # 设置字体为Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    
    # 可视化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=labels, cmap='viridis')
    plt.title('Tennis Players Clustering')
    plt.xlabel('Average Points Won')
    plt.ylabel('Average Distance Run')

    # 创建一个列表来保存所有要调整的文本对象
    texts = []
    for i, name in enumerate(names):
        texts.append(plt.text(scaled_features[i, 0], scaled_features[i, 1], name))

    # 调用adjust_text函数，传入文本对象列表和散点图对象
    # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

    plt.show()

# 调用函数进行聚类分析和可视化
cluster_players()
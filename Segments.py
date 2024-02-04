from data_process import read_csv_to_dict
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from collections import defaultdict
# data = pd.read_csv("data_from_web/2023-wimbledon-points.csv")
# data.loc[(data.serve_depth=='CTL'),'serve_depth'] = 1
# data.loc[(data.serve_depth=='NCTL'),'serve_depth'] = 0
# data.loc[(data.return_depth=='D'),'return_depth'] = 1
# data.loc[(data.return_depth=='ND'),'return_depth'] = 0
# data.loc[(data.p1_score=='AD'),'p1_score'] = 50
# data.loc[(data.p2_score=='AD'),'p2_score'] = 50
# data['p1_score'] = data['p1_score'].astype(int)
# data['p2_score'] = data['p2_score'].astype(int)



def classify_and_save_data(data: pd.DataFrame, save_path: str):
# 按 match_id 分类数据
    classified_data = defaultdict(list)
    for index, row in data.iterrows():
        match_id = row['match_id']
        classified_data[match_id].append(row.to_dict())

        # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 为每个 match_id 创建并保存CSV文件
    for match_id, rows in classified_data.items():
        file_name = f"match_{match_id}.csv"
        file_path = os.path.join(save_path, file_name)

        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

# 使用函数
csv_data = pd.read_csv('data_processed/invert_Wimbledon_labels.csv')
classify_and_save_data(csv_data, 'data_segment')
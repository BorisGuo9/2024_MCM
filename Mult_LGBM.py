# 导入必要的库
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import MinMaxScaler
import os

# 准备数据（示例数据）
# 请替换成你自己的数据
data = pd.read_csv('data_processed/Standard Wimbledon_with_victor_labels_2.csv')
X = data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16']]
x = np.array(X)
# print(x.shape)
y = data['labels']


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# 构建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# 设置LightGBM参数
params = {
    'objective': 'binary',  # 二分类问题
    'metric': 'auc',  # 评估指标，可根据具体情况选择
    'boosting_type': 'gbdt',  # 提升类型
    'num_leaves': 50,  # 叶子节点数  31
    'learning_rate': 0.1,  # 学习率  0.05
    'feature_fraction': 0.9,  # 特征采样比例 0.9
    'bagging_fraction': 0.8,  # 数据采样比例 0.8
    'bagging_freq': 5,  # 数据采样频率  5
    'verbose': 0  # 控制输出信息   0
}


num_round = 400 # Point Model的迭代次数
# bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)
bst_point = lgb.train(params, train_data, num_round, valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)])


# # 在测试集上进行预测
# y_pred = bst_point.predict(X_test, num_iteration=bst_point.best_iteration)
# y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]


# Game Model


x17 = bst_point.predict(x, num_iteration=bst_point.best_iteration) 
x17 = np.array(x17)
x17 = x17.reshape(-1, 1)  # 将x17转换为一维数组
# print(x17)
# lowessed_point = lowess(np.arange(len(x17)), x17, frac = 0.05)
# lowessed_point = lowessed_point[:, 1]
# lowessed_point = np.array(lowessed_point)
# lowessed_point = lowessed_point.reshape(-1,1)
# print(np.shape(x))
# print(np.shape(x17))
# x = np.hstack((x, lowessed_point))
# print(np.shape(x))
x = np.hstack((x, x17))

# print(x)
data_labels = pd.read_csv('data_processed/Wimbledon_with_victor_labels_2.csv')
# for row in range(0,data_labels.shape[0]):
#     if data_labels['game_victor'] == 2:
#         data_labels['game_vicotr'] = 0
data_labels.loc[(data_labels.game_victor == 2),'game_victor'] = 0
data_labels.loc[(data_labels.set_victor == 2),'set_victor'] = 0
data_labels.loc[(data_labels.match_victor == 2),'match_victor'] = 0


y = data_labels['game_victor']





# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# 构建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

num_round = 400 # Game Model的迭代次数
# bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)
bst_game = lgb.train(params, train_data, num_round, valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)])


# Set Model
x18 = bst_game.predict(x, num_iteration=bst_game.best_iteration) 
# lowessed_game = lowess(np.arange(len(x18)), x18, frac = 0.1)
# lowessed_game = lowessed_game[:, 1]
# print(lowessed_game)

# lowessed_game = np.array(lowessed_game)
# lowessed_game = lowessed_game.reshape(-1,1)
x18 = np.array(x18)
x18 = x18.reshape(-1, 1)  # 将x17转换为一维数组
# x = np.hstack((x, lowessed_game))
# plt.figure()
# plt.plot(x18[1:100])
# plt.show()

x = np.hstack((x, x18))


y = data_labels['set_victor']

# print(x)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# 构建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


num_round = 800 # Set Model的迭代次数
# bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)
bst_set = lgb.train(params, train_data, num_round, valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)])

# Match Model
x19 = bst_set.predict(x, num_iteration=bst_set.best_iteration) 
# lowessed_set = lowess(np.arange(len(x19)), x19, frac = 0.01)
# lowessed_set = lowessed_set[:, 1]
# lowessed_set = np.array(lowessed_set)
# lowessed_set = lowessed_set.reshape(-1,1)

# x = np.hstack((x,lowessed_set))

# print(x19)
x19 = np.array(x19)
x19 = x19.reshape(-1, 1)  # 将x17转换为一维数组

x = np.hstack((x, x19))
y = data_labels['match_victor']



# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=30)

# 构建LightGBM数据集
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)


num_round = 800 # Set Model的迭代次数
# bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)
bst_match = lgb.train(params, train_data, num_round, valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)])

# 在测试集上进行预测
y_pred = bst_match.predict(X_test, num_iteration=bst_match.best_iteration)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred_binary)
report = classification_report(y_test, y_pred_binary)

# 打印结果
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')




def momentum_data(label_csv_file, standard_csv_file,output_folder):
    # 预测温网的一场比赛
    # single_match = pd.read_csv('data_positive/data_segment',label_csv_file)
    standard_csv = os.path.join('data_positive', standard_csv_file)
    single_match = pd.read_csv(standard_csv)
    print(132)
    X = single_match[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16']]
    x = np.array(X)
    label_csv = os.path.join('data_positive', label_csv_file)
    print(label_csv)
    single_match = pd.read_csv(label_csv)
    single_match.loc[(single_match.game_victor == 2),'game_victor'] = 0
    single_match.loc[(single_match.set_victor == 2),'set_victor'] = 0
    single_match.loc[(single_match.match_victor == 2),'match_victor'] = 0
    x17 = single_match['game_victor']
    x18 = single_match['set_victor']
    x19 = single_match['match_victor']
    x17 = np.array(x17)
    x18 = np.array(x18)
    x19 = np.array(x19)
    x17 = x17.reshape(-1, 1) 
    x18 = x18.reshape(-1, 1)
    x19 = x19.reshape(-1, 1) # 将x17转换为一维数组
    # x = np.hstack((x, x17))
    y_predicted_point = bst_point.predict(x, num_iteration=bst_match.best_iteration)
    x = np.hstack((x, x17))
    y_predicted_game = bst_game.predict(x, num_iteration=bst_match.best_iteration)
    x = np.hstack((x, x18))
    y_predicted_set = bst_set.predict(x, num_iteration=bst_match.best_iteration)
    x = np.hstack((x, x19))
    y_predicted_match = bst_match.predict(x, num_iteration=bst_match.best_iteration)
    y_predicted_match_pos = pd.DataFrame(y_predicted_match)
    y_predicted_set_pos = pd.DataFrame(y_predicted_set)
    y_predicted_game_pos = pd.DataFrame(y_predicted_game)
    y_predicted_point_pos = pd.DataFrame(y_predicted_point)

    # lowessed_set = lowess(np.arange(len(y_predicted)), y_predicted, frac = 0.2)
    # lowessed_set = lowessed_set[:, 1]
    # lowessed_set = np.array(lowessed_set)

    # 创建一个MinMaxScaler对象
    # scaler = MinMaxScaler()

    # 将lowessed_set进行归一化
    # normalized_set = scaler.fit_transform(lowessed_set.reshape(-1, 1))
    # los = pd.DataFrame(lowessed_set)
    # los.to_csv('lowessed_set.csv')
    # plt.figure()
    # plt.plot(normalized_set)
    # plt.plot(y_pred)
    # plt.show()

    # 预测温网的一场比赛,反转分数
    standard_csv = os.path.join('data_negative', standard_csv_file)
    single_match = pd.read_csv(standard_csv)
    # single_match = pd.read_csv('data_negative/data_segment',label_csv_file)
    X = single_match[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16']]
    x = np.array(X)
    # single_match = pd.read_csv('data_negative/Standard_data',standard_csv_file)
    label_csv = os.path.join('data_negative', label_csv_file)
    print(label_csv)
    single_match = pd.read_csv(label_csv)
    single_match.loc[(single_match.game_victor == 2),'game_victor'] = 0
    single_match.loc[(single_match.set_victor == 2),'set_victor'] = 0
    single_match.loc[(single_match.match_victor == 2),'match_victor'] = 0
    x17 = single_match['game_victor']
    x18 = single_match['set_victor']
    x19 = single_match['match_victor']
    x17 = np.array(x17)
    x18 = np.array(x18)
    x19 = np.array(x19)
    x17 = x17.reshape(-1, 1) 
    x18 = x18.reshape(-1, 1)
    x19 = x19.reshape(-1, 1) # 将x17转换为一维数组
    # x = np.hstack((x, x17))
    y_predicted_point = bst_point.predict(x, num_iteration=bst_match.best_iteration)
    x = np.hstack((x, x17))
    y_predicted_game = bst_game.predict(x, num_iteration=bst_match.best_iteration)
    x = np.hstack((x, x18))
    y_predicted_set = bst_set.predict(x, num_iteration=bst_match.best_iteration)
    x = np.hstack((x, x19))
    y_predicted_match = bst_match.predict(x, num_iteration=bst_match.best_iteration)
    y_predicted_match_neg = pd.DataFrame(y_predicted_match)
    y_predicted_set_neg = pd.DataFrame(y_predicted_set)
    y_predicted_game_neg = pd.DataFrame(y_predicted_game)
    y_predicted_point_neg = pd.DataFrame(y_predicted_point)
    new_filename = f"momentum of {os.path.basename(label_csv_file)}"
    output_file_path = os.path.join(output_folder, new_filename)
    print(output_file_path)
    result_df = pd.concat([y_predicted_match_pos, y_predicted_set_pos, y_predicted_game_pos, y_predicted_point_pos,y_predicted_match_neg, y_predicted_set_neg, y_predicted_game_neg, y_predicted_point_neg], axis=1)
    result_df.to_csv(output_file_path, index=False)
    
def process_all_csv_files(label_csvs_folder, standard_csvs_folder):
    # 遍历文件夹中的所有 CSV 文件
    output_folder = "momentum_data"
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data_path = os.path.join('data_positive', label_csvs_folder)
    for label_filename in os.listdir(data_path):
        if label_filename.endswith(".csv"):
            # 构建标签CSV文件的路径
            label_csv_file = os.path.join(label_csvs_folder, label_filename)

            # 构建相应的标准CSV文件的路径
            standard_filename = "Standard_" + label_filename
            standard_csv_file = os.path.join(standard_csvs_folder, standard_filename)


            # 处理标签和标准CSV文件
            print('dadadadacxsac')
            momentum_data(label_csv_file, standard_csv_file,output_folder)

# 调用处理所有 CSV 文件的函数，传入文件夹路径
# process_all_csv_files("data_segment", 'Standard_data')


# # 结果作图

# importance_values_gain = bst_game.feature_importance(importance_type='gain')
# importance_values_split = bst_game.feature_importance(importance_type='split')

# feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16','x17']
# dataset = pd.DataFrame(importance_values_gain)
# dataset.to_csv('importance_values_gain of Multi_LGBM',index = False)

# dataset = pd.DataFrame(importance_values_split)
# dataset.to_csv('importance_values_split of Multi_LGBM',index = False)

# df_importance = pd.DataFrame({'Feature': feature_names, 'Gain Importance': importance_values_gain, 'Split Importance': importance_values_split})
# # 排序特征按照Gain Importance降序排列
# df_importance = df_importance.sort_values(by='Gain Importance', ascending=False)

# plt.figure(figsize=(12, 6))
# sns.barplot(x='Gain Importance', y='Feature', data=df_importance, palette='viridis')
# plt.title('Feature Importance (Gain) - LightGBM Model')
# plt.xlabel('Gain Importance')
# plt.ylabel('Feature')
# plt.show()

# # 排序特征按照Gain Importance降序排列
# df_importance = df_importance.sort_values(by='Split Importance', ascending=False)

# plt.figure(figsize=(12, 6))
# sns.barplot(x='Split Importance', y='Feature', data=df_importance, palette='viridis')
# plt.title('Feature Importance (Split) - LightGBM Model')
# plt.xlabel('Split Importance')
# plt.ylabel('Feature')
# plt.show()

feature_annotations = {
    "x_1": "Number of games lead in the set",
    "x_2": "Scores lead in the current game",
    "x_3": "Whether the player is serving",
    "x_4": "Number of sets lead in the match",
    "x_5": "Player 1's aces",
    "x_6": "Player 1's winners",
    "x_7": "Player 1's double faults",
    "x_8": "Player 1's unforced errors",
    "x_9": "Ratio of net points won to total net points",
    "x_10": "Ratio of break points won to total break points",
    "x_11": "Total running distance in the match",
    "x_12": "Total running distance in the last three points",
    "x_13": "Running distance in the last point",
    "x_14": "Ball speed",
    "x_15": "Number of hits per rally",
    "x_16": "Serve or Return Depth",
    "x_17": "Probability of winning the point",
    "x_18": "Probability of winning the game",
    "x_19": "Probability of winning the set"
}
labels = [f"x_{i+1}" for i in range(16)]

importance_values_gain = bst_match.feature_importance(importance_type='gain')
importance_values_split = bst_match.feature_importance(importance_type='split')

feature_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16','x17','x18','x19']

# Create DataFrames for importance values
df_importance_gain = pd.DataFrame({'Feature': feature_names, 'Gain Importance': importance_values_gain})
df_importance_split = pd.DataFrame({'Feature': feature_names, 'Split Importance': importance_values_split})

# Save importance values to CSV files
df_importance_gain.to_csv('importance_values_gain_of_Multi_LGBM.csv', index=False)
df_importance_split.to_csv('importance_values_split_of_Multi_LGBM.csv', index=False)

# Sort features by Gain Importance
df_importance_split = df_importance_split[:16]
df_importance_gain = df_importance_gain[:16]
df_importance_gain = df_importance_gain.sort_values(by='Gain Importance', ascending=False)

print(df_importance_gain)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Gain Importance', y='Feature', data=df_importance_gain, palette='viridis')
plt.title('Feature Importance (Gain) - LightGBM Model')
plt.xlabel('Gain Importance')
plt.ylabel('Feature')

# Add labels to the bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=10, color='black')
colors = plt.cm.get_cmap('viridis', len(labels))

patch_list = [plt.Rectangle((0,0),1,1, color=colors(i), label=f'{label}: {feature_annotations[label]}') 
              for i, label in enumerate(labels)]
plt.legend(handles=patch_list, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
           fontsize='small', prop={'family': 'Times New Roman'})


plt.tight_layout()
plt.savefig('feature_importance_split_plot.png', dpi=300)
plt.show()

# Sort features by Split Importance
df_importance_split = df_importance_split.sort_values(by='Split Importance', ascending=False)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Split Importance', y='Feature', data=df_importance_split, palette='viridis')
plt.title('Feature Importance (Split) - LightGBM Model')
plt.xlabel('Split Importance')
plt.ylabel('Feature')

# Add labels to the bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                ha='left', va='center', fontsize=10, color='black')

colors = plt.cm.get_cmap('viridis', len(labels))

patch_list = [plt.Rectangle((0,0),1,1, color=colors(i), label=f'{label}: {feature_annotations[label]}') 
              for i, label in enumerate(labels)]
plt.legend(handles=patch_list, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., 
           fontsize='small', prop={'family': 'Times New Roman'})


plt.tight_layout()
plt.savefig('feature_importance_split_plot.png', dpi=300)
plt.show()



conf_mat = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='PuBuGn')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='coral', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='darkblue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



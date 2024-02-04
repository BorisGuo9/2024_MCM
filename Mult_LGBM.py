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

# 准备数据（示例数据）
# 请替换成你自己的数据
data = pd.read_csv('data_processed/Standard dataset_2.csv')
X = data[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17']]
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


x18 = bst_point.predict(x, num_iteration=bst_point.best_iteration) 
x18 = np.array(x18)
x18 = x18.reshape(-1, 1)  # 将x18转换为一维数组
# print(x18)
# lowessed_point = lowess(np.arange(len(x18)), x18, frac = 0.05)
# lowessed_point = lowessed_point[:, 1]
# lowessed_point = np.array(lowessed_point)
# lowessed_point = lowessed_point.reshape(-1,1)
# print(np.shape(x))
# print(np.shape(x18))
# x = np.hstack((x, lowessed_point))
# print(np.shape(x))
x = np.hstack((x, x18))

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
x19 = bst_game.predict(x, num_iteration=bst_game.best_iteration) 
# lowessed_game = lowess(np.arange(len(x19)), x19, frac = 0.1)
# lowessed_game = lowessed_game[:, 1]
# print(lowessed_game)

# lowessed_game = np.array(lowessed_game)
# lowessed_game = lowessed_game.reshape(-1,1)
x19 = np.array(x19)
x19 = x19.reshape(-1, 1)  # 将x18转换为一维数组
# x = np.hstack((x, lowessed_game))
# plt.figure()
# plt.plot(x19[1:100])
# plt.show()

x = np.hstack((x, x19))


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
x20 = bst_set.predict(x, num_iteration=bst_set.best_iteration) 
# lowessed_set = lowess(np.arange(len(x20)), x20, frac = 0.01)
# lowessed_set = lowessed_set[:, 1]
# lowessed_set = np.array(lowessed_set)
# lowessed_set = lowessed_set.reshape(-1,1)

# x = np.hstack((x,lowessed_set))

# print(x20)
x20 = np.array(x20)
x20 = x20.reshape(-1, 1)  # 将x18转换为一维数组

x = np.hstack((x, x20))
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





# 预测温网的一场比赛
single_match = pd.read_csv('Standard match_2023-wimbledon-1701_labels.csv')
X = single_match[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17']]
x = np.array(X)
single_match = pd.read_csv('match_2023-wimbledon-1701_labels.csv')
x18 = single_match['game_victor']
x19 = single_match['set_victor']
x20 = single_match['match_victor']
x18 = np.array(x18)
x19 = np.array(x19)
x20 = np.array(x20)
x18 = x18.reshape(-1, 1) 
x19 = x19.reshape(-1, 1)
x20 = x20.reshape(-1, 1) # 将x18转换为一维数组
# x = np.hstack((x, x18))
x = np.hstack((x, x18,x19,x20))
y_predicted = bst_match.predict(x, num_iteration=bst_match.best_iteration)
lowessed_set = lowess(np.arange(len(y_predicted)), y_predicted, frac = 0.2)
lowessed_set = lowessed_set[:, 1]
# lowessed_set = np.array(lowessed_set)

# 创建一个MinMaxScaler对象
scaler = MinMaxScaler()

# 将lowessed_set进行归一化
normalized_set = scaler.fit_transform(lowessed_set.reshape(-1, 1))
# los = pd.DataFrame(lowessed_set)
# los.to_csv('lowessed_set.csv')

plt.figure()
plt.plot(normalized_set)
# plt.plot(y_pred)
plt.show()


# 结果作图

importance_values_gain = bst_match.feature_importance(importance_type='gain')
importance_values_split = bst_match.feature_importance(importance_type='split')

dataset = pd.DataFrame(importance_values_gain)
dataset.to_csv('importance_values_gain of Multi_LGBM',index = False)

dataset = pd.DataFrame(importance_values_split)
dataset.to_csv('importance_values_split of Multi_LGBM',index = False)


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



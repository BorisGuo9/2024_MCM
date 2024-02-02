# 导入必要的库
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# 准备数据（示例数据）
# 请替换成你自己的数据
df = pd.read_csv('Standard dataset.csv')
X = df[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16']]
x = np.array(X)
# print(x)
y = df['labels']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


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
    'num_leaves': 31,  # 叶子节点数  31
    'learning_rate': 0.05,  # 学习率  0.05
    'feature_fraction': 0.9,  # 特征采样比例 0.9
    'bagging_fraction': 0.8,  # 数据采样比例 0.8
    'bagging_freq': 5,  # 数据采样频率  5
    'verbose': 0  # 控制输出信息   0
}



# 训练模型
num_round = 400 # 迭代次数
# bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)])


# 在测试集上进行预测
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]


# 评估模型性能
accuracy = accuracy_score(y_test, y_pred_binary)
report = classification_report(y_test, y_pred_binary)


# 打印结果
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')


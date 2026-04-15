import pandas as pd
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# 特征提取函数
def extract_features(time, flux):
    features = {}
    # 确保时间数据为浮点数格式
    time_str = time.strip('[]').split(',')
    flux_str = flux.strip('[]').split(',')
    time = np.array(time_str, dtype=float)
    flux = np.array(flux_str, dtype=float)

    lc = lk.LightCurve(time=time, flux=flux)
    periodogram = lc.to_periodogram(method='lombscargle', minimum_frequency=0.1, maximum_frequency=24)
    # 主要周期特征
    features['main_period'] = periodogram.period_at_max_power.value
    # 主要频率特征
    features['main_frequency'] = 1 / features['main_period']
    # 峰值功率特征
    features['max_power'] = periodogram.max_power.value
    # 峰值数量特征
    peaks = periodogram.power > (0.5 * periodogram.max_power)
    features['num_peaks'] = np.sum(peaks)
    return features

df = pd.read_csv("lightcurve_data.csv")

plt.figure(figsize=(10, 10))
plt.title('变星种类的分布')
df['classALeRCE'].value_counts().plot(kind='bar')
plt.savefig('分布.png')

time_data = df['time']
flux_data = df['flux']


features_list = [extract_features(time, flux) for time, flux in zip(time_data, flux_data)]
features_df = pd.DataFrame(features_list)
X = features_df.values

dic = {'E': 0, 'RRL': 1, 'QSO': 2, 'LPV': 3, 'AGN': 4, 'YSO': 5,
       'CEP': 6, 'SNIa': 7, 'Blazar': 8, 'DSCT': 9, 'Periodic-Other': 10,
       'CV/Nova': 11, 'SNII': 12, 'SLSN': 13, 'SNIbc': 14
       }
Y = df['classALeRCE'].map(dic).values


imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
# 平衡类别
data = pd.DataFrame(X)
data['label'] = Y
# 分离不同类别
dfs = [data[data['label'] == label] for label in data['label'].unique()]
# 重新采样以平衡类别
dfs_resampled = [resample(d, replace=True, n_samples=len(max(dfs, key=len)), random_state=42) for d in dfs]
df_balanced = pd.concat(dfs_resampled)
X_balanced = df_balanced.drop('label', axis=1).values
y_balanced = df_balanced['label'].values

scaler = StandardScaler()
X_balanced = scaler.fit_transform(X_balanced)

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
X_train.shape,X_test.shape,y_train.shape, y_test.shape

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
del df

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 使用GridSearchCV进行超参数调优
param_grid = {'C': [10, 100, 1000],
              'gamma': [100, 10],
              'kernel': ['rbf']
              }
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2,n_jobs=-1)
grid.fit(X_train, y_train)
# 最佳参数
print(f'Best parameters: {grid.best_params_}')
# 预测
y_pred = grid.predict(X_test)
# 评估模型
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

model_filename = 'best_svm_model.joblib'
joblib.dump(grid, model_filename)
print(f"模型已保存为 {model_filename}")


# 获取网格搜索结果
results = grid.cv_results_
param_C = results['param_C'].data
param_gamma = results['param_gamma'].data
mean_test_score = results['mean_test_score']

# 创建一个二维数组来存储不同参数组合的得分
C_values = np.unique(param_C)
gamma_values = np.unique(param_gamma)
score_matrix = np.zeros((len(C_values), len(gamma_values)))

for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        idx = np.where((param_C == C) & (param_gamma == gamma))[0]
        score_matrix[i, j] = mean_test_score[idx]

# 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(score_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=gamma_values, yticklabels=C_values)
plt.xlabel('Gamma')
plt.ylabel('C')
plt.title('Grid Search Results: Mean Test Score')
plt.show()

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dic.keys(), yticklabels=dic.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('混淆矩阵.png')
plt.show()
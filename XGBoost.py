import pandas as pd
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import xgboost as xgb

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

# 创建XGBoost分类器
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(dic),
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    verbosity=1,
    use_label_encoder=False,
    n_jobs=-1
)

# 训练XGBoost模型，并指定评估集
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric=["mlogloss", "merror"],
    early_stopping_rounds=10,
    verbose=True
)
# 打印模型参数
print(xgb_model.get_params())
# 预测
y_pred = xgb_model.predict(X_test)
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy:.2f}')

# 保存最优模型
model_filename = 'best_xgb_model.joblib'
joblib.dump(xgb_model, model_filename)
print(f"模型已保存为 {model_filename}")

# 可视化特征重要性
xgb.plot_importance(xgb_model, max_num_features=10)
plt.savefig('可视化特征重要性.png')
plt.show()

# 获取训练过程中的评估结果
results = xgb_model.evals_result()
# 可视化训练过程
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# 绘制log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Validation')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.savefig('XGBoost Log Loss.png')
plt.show()

# 绘制分类错误率
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='Train')
ax.plot(x_axis, results['validation_1']['merror'], label='Validation')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.savefig('XGBoost Classification Error.png')
plt.show()

# 绘制 accuracy 曲线
train_accuracy = [1 - e for e in results['validation_0']['merror']]
val_accuracy = [1 - e for e in results['validation_1']['merror']]

fig, ax = plt.subplots()
ax.plot(x_axis, train_accuracy, label='Train Accuracy')
ax.plot(x_axis, val_accuracy, label='Validation Accuracy')
ax.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('XGBoost Accuracy')
plt.savefig('XGBoost Accuracy.png')
plt.show()
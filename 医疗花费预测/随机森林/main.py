import warnings

import pandas

warnings.filterwarnings('ignore')
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn.metrics

df = pd.read_csv('train.csv')
df = df.dropna(how='all')
df.describe()
df.head()
df.corr()

corrMatrix = df.corr()
sns.set(font_scale=1.10)
plt.figure(figsize=(8, 8))
sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True, annot=True, cmap='viridis', linecolor="white")
plt.title('Correlation between video num and retain')

df['bmi_int'] = df['bmi'].apply(lambda x: int(x))
variables = ['sex', 'smoker', 'region', 'age', 'bmi_int', 'children']

print('建模与评估\n\n')

le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['region'] = le_region.fit_transform(df['region'])

variables = ['sex', 'smoker', 'region', 'age', 'bmi', 'children']

X = df[variables]
sc = StandardScaler()
X = sc.fit_transform(X)
Y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


regressor = RandomForestRegressor(n_estimators=1300)
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)

print('RandomForestRegressor evaluating result:')
print("Train MAE: ", sklearn.metrics.mean_absolute_error(y_train, y_train_pred))
print("Train RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_train, y_train_pred)))
print("Test MAE: ", sklearn.metrics.mean_absolute_error(y_test, y_test_pred))
print("Test RMSE: ", np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_test_pred)))

print('特征重要度排序\n')
importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

importance_list = []
for f in range(X.shape[1]):
    variable = variables[indices[f]]
    importance_list.append(variable)
    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))

plt.figure()
plt.title("feature importance")
plt.bar(importance_list, importances[indices],
        color="r", yerr=std[indices], align="center")
plt.show()

answer = []
file = pandas.read_csv('test_sample.csv')

print('在新数据上进行预测\n\n')

for index, row in file.iterrows():
    billy = [row['sex'], row['smoker'], row['region'], row['age'], row['bmi'], row['children']]
    billy[0] = le_sex.transform([billy[0]])[0]
    billy[1] = le_smoker.transform([billy[1]])[0]
    billy[2] = le_region.transform([billy[2]])[0]

    X = sc.transform([billy])

    cost_for_billy = regressor.predict(X)[0]
    info = {"age": row['age'], "sex": row['sex'], "bmi": row['bmi'], "children": row['children'],
            "smoker": row['smoker'], "region": row['region'], "charges": cost_for_billy}
    answer.append(info)

csvfile = pandas.DataFrame(answer, columns=["age", "sex", "bmi", "children", "smoker", "region", "charges"])
print(csvfile)
csvfile.to_csv("submission.csv", index=False)

from sklearn.datasets import load_boston
boston = load_boston()
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
X = boston.data
y = boston.target
num_training = int(0.7*len(X))

#分割数据集
X_train = X[:num_training]
y_train = y[:num_training]
X_test = X[num_training:]
y_test = y[num_training:]
reg = linear_model.Ridge(alpha = .5)

#训练模型
reg.fit(X_train,y_train)

#预测模型
y_pred = reg.predict(X_test)

#输出模型参数
print("系数",reg.coef_)
print(reg.coef_.shape)
print("常数项",reg.intercept_)

#计算均方误差
print("在测试集均方误差",mean_squared_error(y_test,y_pred))

#计算r2值
print("r2值",r2_score(y_test,y_pred))

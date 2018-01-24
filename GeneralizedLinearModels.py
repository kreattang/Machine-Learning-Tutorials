import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import linear_model,datasets
from sklearn.metrics import mean_squared_error,r2_score

#加载我们的训练数据
#使用sklearn提供的diabetes数据集

diabetes = datasets.load_diabetes()

#仅选择一列数据作为训练数据
diabetes_X = diabetes.data[:, np.newaxis, 2]
#y是标签
y = diabetes.target

#这里我们使用sklearn里的shuffle函数把数据顺序打乱
X,y = shuffle(diabetes_X,y,random_state=7)

#分割训练集和数据集
num_training = int(0.7*len(X))
X_train = X[:num_training]
y_train = y[:num_training]
X_test = X[num_training:]
y_test = y[num_training:]

#构建模型
reg = linear_model.LinearRegression()
#训练模型
reg.fit(X_train,y_train)
#预测模型
y_pred = reg.predict(X_test)

#输出模型参数
print("模型参数",reg.coef_)

#计算均方误差
print("均方误差",mean_squared_error(y_test,y_pred))

#计算r2值
print("r2值",r2_score(y_test,y_pred))
plt.figure()
plt.subplot(1,2,1)
plt.title('rew data')
plt.scatter(X_test,y_test,color='blue')
plt.subplot(1,2,2)
plt.title('linear model')
plt.scatter(X_test,y_test,color='blue')
plt.plot(X_test,y_pred,color='black',linewidth=4)
plt.show()

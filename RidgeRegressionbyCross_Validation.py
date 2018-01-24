from sklearn import linear_model
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
boston = load_boston()
X = boston.data
y = boston.target
num_training = int(0.7*len(X))
X_train = X[:num_training]
y_train = y[:num_training]
X_test = X[num_training:]
y_test = y[num_training:]
reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
reg.fit(X_train,y_train)
print("最优alpha",reg.alpha_)
y_pred = reg.predict(X_test)
print("在测试集均方误差",mean_squared_error(y_test,y_pred))

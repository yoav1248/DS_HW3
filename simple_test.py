import numpy as np
import matplotlib.pyplot as plt
from data import *
from knn import *

k = 1
n = 100
N = 10000

X_train = np.linspace(-1, 1, n).reshape(-1, 1)
y_train = add_noise(X_train ** 2)

X_test = np.linspace(-1, 1, N).reshape(-1, 1)
y_test_true = add_noise(X_test**2)

model = RegressionKnn(k)
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)


plt.plot(X_test, y_test_pred, label="predicted")
plt.plot(X_test, y_test_true, label="true")
plt.legend()
plt.show()
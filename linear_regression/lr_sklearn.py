import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

A = np.array([[1,2,3,5,6,8,9,10,13,15,16,18,19,20]]).T
b = np.array([[2,4,5,6,7,8,9,13,15,16,17,18,19,22]]).T

plt.plot(A,b,'ro')

lr = linear_model.LinearRegression()
lr.fit(A,b)

x0 = np.array([[1,20]]).T
y0 = lr.coef_*x0 + lr.intercept_

plt.plot(x0,y0)

plt.show()
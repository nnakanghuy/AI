import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
#random data
A = [1,2,3,5,6,8,9,10,13,15,16,18,19,20]
b = [13,14,12,9,8,7,4,5,9,12,15,17,18,22]

plt.plot(A,b,'ro')
#transpose
A = np.array([A]).T
b = np.array([b]).T

x_square = np.array([A[:,0]**2]).T
#combine A^2 and A and one
A = np.concatenate((x_square,A), axis=1)

lr = linear_model.LinearRegression()
lr.fit(A,b)

print(lr.coef_)
x0 = np.linspace(1,20,1000)
y0 = lr.coef_[0][0]*x0*x0 + lr.coef_[0][1]*x0 + lr.intercept_

plt.plot(x0,y0)

plt.show()

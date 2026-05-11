#b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
#A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.animation as animation
def cost(x):
    m = A.shape[0]
    return 0.5/m*np.linalg.norm(A.dot(x)-b,2)**2
def grad(x):
    m = A.shape[0]
    return 1/m * A.T.dot(A.dot(x)-b)
def gradient_descent(x_init,learning_rate,iteration):
    x_list = [x_init]
    for i in range(iteration):
        x_new = x_list[-1]-learning_rate*grad(x_list[-1])
        x_list.append(x_new)
        if(np.linalg.norm(grad(x_new))<1e-3):
            break
    return x_list

A = np.array([[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]).T
b = np.array([[2,5,7,9,11,16,19,23,22,29,29,35,37,40,46,42,39,31,30,28,20,15,10,6]]).T
fig = plt.figure("Gradient descent")
ax = plt.axes(xlim=(-5,35), ylim = (-10,60))
plt.plot(A,b,'ro')
x_square = np.array([A[:,0]**2]).T
A = np.concatenate((x_square,A), axis = 1)
#ve do thi = linear regression
lr = linear_model.LinearRegression()
lr.fit(A,b)

x0 = np.linspace(2,40,1000)
y0 = lr.coef_[0][0]*x0**2+lr.coef_[0][1]*x0+lr.intercept_
plt.plot(x0,y0,color = "green")
#(số_hàng, số_cột)
ones = np.ones((A.shape[0],1), dtype = np.int8)
A = np.concatenate((A,ones), axis =1)
#random
x0_init = np.array([[-2.1],[5.1],[-2.1]])
y0_init = x0_init[0][0]*x0**2+x0_init[1][0]*x0+x0_init[2][0]
plt.plot(x0,y0_init,color = "black", alpha = 0.3)


iteration = 70
learning_rate = 1e-6
x_list = gradient_descent(x0_init,learning_rate,iteration)
#draw list
for i in range(len(x_list)):
    y0_list = x_list[i][0]*x0**2+x_list[i][1]*x0+x_list[i][2]
    plt.plot(x0, y0_list,color = "black", alpha =0.3)

#draw animation
line , = ax.plot([],[], color = "blue")
def update(i):
    y0_gd = x_list[i][0][0]*x0**2+x_list[i][1][0]*x0+x_list[i][2][0]
    line.set_data(x0,y0_gd)
    return line,
iters = np.arange(1,len(x_list),1)
line_ani = animation.FuncAnimation(fig, update, iters, interval = 50, blit = True)

plt.show()
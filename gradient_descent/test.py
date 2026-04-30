#f(x)=x^2+5*sin(x)
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#ham f(x)
def cost(x):
    return x**2 + 5*np.sin(x)
#ham f'(x)
def grad(x):
    return 2*x + 5*np.cos(x)
#thuat toan gradient descent
def gradient_descent(learning_rate, x0):
    x = [x0]

    for i in range(100):
        x_new = x[-1] - learning_rate*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x,i)
#tao figure
# gs = GridSpec(nrows = 1, ncols = 2)
plt.figure(figsize = (8,6), label = "gradient descent")
# tao 2 chieu
x1, it1 = gradient_descent(0.05,-5)
x2, it2 = gradient_descent(0.1,5)
print(f"x1 = {x1[-1]}, cost = {cost(x1[-1])}, iteration = {it1}")
print(f"x2 = {x2[-1]}, cost = {cost(x2[-1])}, iteration = {it2}")
#ve hinh
x0 = np.linspace(-4,6,100)
y0 = x0**2 + 5*np.sin(x0)
plt.plot(x0,y0, color = "blue")
for i in range(it1):
    x1_dot = x1[i]
    y1_dot = x1_dot**2+5*np.sin(x1_dot)
    plt.plot(x1_dot,y1_dot,'ro')
for i in range(it2):
    x2_dot = x2[i]
    y2_dot = x2_dot**2+5*np.sin(x2_dot)
    plt.plot(x2_dot,y2_dot,'ro')
plt.show()
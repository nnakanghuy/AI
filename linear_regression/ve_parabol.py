import numpy as np
import matplotlib.pyplot as plt
#random data
A = [1,2,3,5,6,8,9,10,13,15,16,18,19,20]
b = [13,14,12,9,8,7,4,5,9,12,15,17,18,22]
#visualize data
plt.plot(A,b,'ro')
#transpose
A = np.array([A]).T
b = np.array([b]).T
#create vector one
ones = np.ones((14,1),dtype = np.int8)
#A^2
x_square = np.array([A[:,0]**2]).T
#combine A^2 and A and one
A = np.concatenate((x_square,A), axis=1)
A = np.concatenate((A,ones), axis = 1)


#use formular
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
print(x)
x0 = np.linspace(1,20,1000)
y0 = x[0][0]*x0*x0+x[1][0]*x0+x[2][0]
#draw
plt.plot(x0,y0)


plt.show()
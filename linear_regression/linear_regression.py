import numpy as np
import matplotlib.pyplot as plt
#random data
A = [1,2,3,5,6,8,9,10,13,15,16,18,19,20]
b = [2,4,5,6,7,8,9,13,15,16,17,18,19,22]
#visualize data
plt.plot(A,b,'ro')
#transpose
A = np.array([A]).T
b = np.array([b]).T
#create vector one
ones = np.ones((14,1),dtype = np.int8)
#combine A and ones
A = np.concatenate((A,ones), axis = 1)
print(A)
#use formular
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)

x0 = np.array([1,20]).T
y0 = x[0][0]*x0+x[1][0]
#draw
plt.plot(x0,y0)


plt.show()
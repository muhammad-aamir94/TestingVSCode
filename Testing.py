import numpy as np 
import pandas as pd 
import os
import cv2
import matplotlib.pyplot as pyplot
from sklearn import metrics

#a = np.array([0,1,2,3])
#res = np.expand_dims(a, axis=1)


#a = np.array(( [0,1], [2,3]))
#res= (np.sum(a, axis = 2))

#a = [ [0,1], [2,3]] 
#res= (np.expand_dims(a, axis = 2))
#print(a)
#print('/****************/')
#print(res)
#print(len(a))

#var = np.zeros([2,2])
#print(var)

img = cv2.imread("White_Cat.jpg")

#cv2.imshow("BN", img)
#h,w = img.shape[:2]
#print(h,w)
A = np.array(img)
print("Matrix A:\n{}, shape={}\n".format(A, A.shape))
B = np.dot(A,2)
print("Multiplication of A with 2:\n{}, shape={}".format(B, B.shape))
#cv2.waitKey(0)

#array = np.arange(27).reshape(3,3,3)
#s =array[3][4][2]
#print(s)
# A = np.array([[1,2,3],
#              [4,5, 6],
#              [7, 8, 9]])
# B = np.dot(A,10)
# print("Matrix A:\n{}, shape={}\n".format(A, A.shape))
# print("Multiplication of A with 10:\n{}, shape={}".format(B, B.shape))



import numpy as np
#两个一维向量生成向量点积的结果
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])
print("output:\n",np.dot(a, b))
#二维矩阵和一维向量(被当作一维矩阵)进行计算
a = np.random.randint(0,10, size = (5,5))
b = np.array([1,2,3,4,5])
print("the shape of a is " + str(a.shape))
print("the shape of b is " + str(b.shape))
print("output:\n",np.dot(a, b))
#两个二维矩阵进行矩阵乘法运算
a = np.random.randint(0, 10, size = (5, 5))
b = np.random.randint(0, 10, size = (5, 3))
print("the shape of a is " + str(a.shape))
print("the shape of b is " + str(b.shape))
print("output:\n",np.dot(a, b))

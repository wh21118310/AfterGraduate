import numpy as np
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print("竖直方向上堆叠:\n",np.vstack((arr1,arr2)))
print("水平方向上堆叠：\n",np.hstack((arr1,arr2)))

a1 = np.array([[1, 2], [3, 4], [5, 6]])
a2 = np.array([[7, 8], [9, 10], [11, 12]])
print("输出a1:\n",a1)
print("输出a2:\n",a2)
print("水平方向上堆叠：\n",np.hstack((a1,a2)))
